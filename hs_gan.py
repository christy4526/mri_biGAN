from dcgan import Generator, Discriminator
from config import Flags
from visdom import Visdom
from adni_dataset import get_dataloader
import torch
from torch import nn
from torch.nn import functional as F
from ignite.engine import Engine, Events, _prepare_batch
from ignite.metrics import RunningAverage, Accuracy
from torch.optim import Adam
from torchtool.summary import Image3D, Scalar, Image2D
from torchvision import datasets, transforms


class ZSampler(object):
    def __init__(self, module, size, device=None):
        self.module = module
        self.size = size
        self.device = device
        self.batch_size = size[0]
        self.fixed_noise = self.module(*self.size, device=self.device)

    @property
    def real_label(self):
        return torch.ones(self.batch_size, device=self.device)

    @property
    def fake_label(self):
        return torch.zeros(self.batch_size, device=self.device)

    def sample(self):
        return self.module(*self.size, device=self.device)


def create_gan_trainer(netG, netD, optimG, optimD, loss_fn, z_sampler, device=None,
                       non_blocking=False, prepare_batch=_prepare_batch):
    if device:
        netG.to(device)
        netD.to(device)

    real_labels = z_sampler.real_label
    fake_labels = z_sampler.fake_label

    def _update(engine, batch):
        netG.train()
        netD.train()
        optimG.zero_grad()
        optimD.zero_grad()

        real, _ = prepare_batch(batch, device=device,
                                non_blocking=non_blocking)

        netD.zero_grad()
        output = netD(real)
        LDreal = loss_fn(output, real_labels)
        Dx = output.mean().item()
        output_real = output.detach()

        LDreal.backward()

        noise = z_sampler.sample()
        fake = netG(noise)

        output = netD(fake.detach())
        LDfake = loss_fn(output, fake_labels)
        DGz1 = output.mean().item()

        LDfake.backward()

        LD = LDreal + LDfake
        optimD.step()

        netG.zero_grad()

        output = netD(fake)
        LG = loss_fn(output, real_labels)
        DGz2 = output.mean().item()
        output_fake = output.detach()

        LG.backward()
        optimG.step()

        return dict(LD=LD.item(), LG=LG.item(), Dx=Dx, DGz1=DGz1, DGz2=DGz2,
                    output_real=output_real, output_fake=output_fake)
    return Engine(_update)


def main():
    parser = Flags()
    parser.set_arguments()
    parser.add_argument('--z_dim', type=int, default=100)
    FG = parser.parse_args()

    vis = Visdom(port=FG.vis_port, env=FG.model)
    report = parser.report(end='<br>')
    vis.text(report, win='report f{}'.format(FG.cur_fold))

    transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = datasets.MNIST(root='./mnist', train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=FG.batch_size,
                                              worker_init_fn=lambda _: torch.initial_seed(),
                                              num_workers=5, shuffle=True, pin_memory=True, drop_last=True)
    # trainloader, _ = get_dataloader(FG.fold, FG.cur_fold, FG.data_root, FG.modality,
    #                                 labels=FG.labels, batch_size=FG.batch_size,
    #                                 balancing=FG.data_balancing)

    torch.cuda.set_device(FG.devices[0])
    device = torch.device(FG.devices[0])

    netG = nn.DataParallel(
        Generator(FG.ckpt_dir, FG.z_dim).weight_init(), device_ids=FG.devices)
    netD = nn.DataParallel(Discriminator(
        FG.ckpt_dir).weight_init(), device_ids=FG.devices)

    optimG = Adam(netG.parameters(), lr=FG.lr, amsgrad=True)
    optimD = Adam(netD.parameters(), lr=FG.lr, amsgrad=True)

    z_sampler = ZSampler(
        torch.randn, (FG.batch_size, FG.z_dim, 1, 1), device=device)
    trainer = create_gan_trainer(
        netG, netD, optimG, optimD, F.binary_cross_entropy,
        z_sampler, device=device, non_blocking=True)

    monitoring_metrics = ['LD', 'LG', 'Dx', 'DGz1', 'DGz2']
    RunningAverage(alpha=0.98, output_transform=lambda x: x['LD']).attach(
        trainer, 'LD')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['LG']).attach(
        trainer, 'LG')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['Dx']).attach(
        trainer, 'Dx')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['DGz1']).attach(
        trainer, 'DGz1')
    RunningAverage(alpha=0.98, output_transform=lambda x: x['DGz2']).attach(
        trainer, 'DGz2')
    real_rate = Accuracy()
    fake_rate = Accuracy()

    trackers = dict()
    for monitoring_metric in monitoring_metrics:
        trackers[monitoring_metric] = Scalar(
            vis, monitoring_metric, monitoring_metric, opts=dict(
                title=monitoring_metric, y_label=monitoring_metric,
                xlabel='epoch', showlegend=True))
    trackers['real_rate'] = Scalar(
        vis, 'real_rate', 'real_rate', opts=dict(
            title='real_rate', y_label='real_rate',
            ytickmin=0, ytickmax=1,
            xlabel='epoch', showlegend=True))
    trackers['fake_rate'] = Scalar(
        vis, 'fake_rate', 'fake_rate', opts=dict(
            title='fake_rate', y_label='fake_rate',
            ytickmin=0, ytickmax=1,
            xlabel='epoch', showlegend=True))
    fakeshow = Image2D(vis, 'fake')
    realshow = Image2D(vis, 'real')

    @trainer.on(Events.ITERATION_COMPLETED)
    def track_logs(engine):
        i = engine.state.iteration / len(trainloader)
        metrics = engine.state.metrics
        for key in metrics.keys():
            trackers[key](i, metrics[key])

        y_pred_real = (engine.state.output['output_real'] >= 0.5).long()
        y_pred_fake = (engine.state.output['output_fake'] < 0.5).long()
        real_rate.update((y_pred_real, z_sampler.real_label.long()))
        fake_rate.update((y_pred_fake, z_sampler.fake_label.long()))

    @trainer.on(Events.EPOCH_COMPLETED)
    def show_fake_example(engine):
        netG.eval()
        fake = netG(z_sampler.fixed_noise)
        fakeshow('fake_images', fake*0.5+0.5)
        realshow('real_images', engine.state.batch[0]*0.5+0.5)
        trackers['real_rate'](engine.state.epoch, real_rate.compute())
        trackers['fake_rate'](engine.state.epoch, fake_rate.compute())
        real_rate.reset()
        fake_rate.reset()

    trainer.run(trainloader, FG.num_epoch)


if __name__ == '__main__':
    main()

