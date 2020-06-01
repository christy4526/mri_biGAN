from classify_config import Flags
from models import Baseline, Baseline3D
from adni_dataset import get_dataloader, process_ninecrop_batch, mean_over_ninecrop
from torchtool import summary

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss

from visdom import Visdom
import torch
from torch.nn import functional as F

if __name__ == '__main__':
    parser = Flags()
    parser.set_arguments()
    FG = parser.parse_args()

    rf = str(FG.cur_fold)

    vis = Visdom(port=FG.vis_port, env=FG.model+'_fold'+rf)
    report = parser.report(end='<br>')
    vis.text(report, win='report f{}'.format(FG.cur_fold))

    torch.cuda.set_device(FG.devices[0])
    device = torch.device(FG.devices[0])

    net = Baseline(FG.ckpt_dir, len(FG.labels))
    # net = Baseline3D(FG.ckpt_dir, len(FG.labels))

    if len(FG.devices) > 1:
        net = torch.nn.DataParallel(net, device_ids=FG.devices)
        print(net.module)
    else:
        print(net)

    optimizer = Adam(net.parameters(), lr=FG.lr, weight_decay=FG.l2_decay)
    scheduler = ExponentialLR(optimizer, gamma=FG.lr_gamma)

    trainloader, testloader = get_dataloader(
        k=FG.fold, cur_fold=FG.cur_fold, modality=FG.modality, axis=FG.axis,
        labels=FG.labels, batch_size=FG.batch_size)

    trainer = create_supervised_trainer(
        net, optimizer, F.cross_entropy,
        device=device, non_blocking=True)

    evaluator = create_supervised_evaluator(
        net, metrics={'accuracy': Accuracy(mean_over_ninecrop),
                      'loss': Loss(F.cross_entropy, mean_over_ninecrop)},
        device=device, non_blocking=True,
        prepare_batch=process_ninecrop_batch)

    lr_tracker = summary.Scalar(
        vis, 'lr', 'lr', opts=dict(
            ylabel='lr', xlabel='epoch',
            ytickmin=0, ytickmax=FG.lr,
            title='learning rate', showlegend=True))

    train_loss_tracker = summary.Scalar(
        vis, 'loss', 'train', opts=dict(
            ylabel='loss', xlabel='epoch',
            ytickmin=0, ytickmax=1,
            title='loss', showlegend=True))

    train_avg_loss_tracker = summary.Scalar(
        vis, 'avg_loss', 'train', opts=dict(
            ylabel='average loss', xlabel='epoch',
            ytickmin=0, ytickmax=1,
            title='average_loss', showlegend=True))

    train_acc_tracker = summary.Scalar(
        vis, 'accuracy', 'train', opts=dict(
            ylabel='accuracy', xlabel='epoch',
            ytickmin=0, ytickmax=1,
            title='accuracy', showlegend=True))

    test_avg_loss_tracker = summary.Scalar(
        vis, 'avg_loss', 'test', opts=dict(
            ylabel='average loss', xlabel='epoch',
            ytickmin=0, ytickmax=1,
            title='average_loss', showlegend=True))

    test_acc_tracker = summary.Scalar(
        vis, 'accuracy', 'test', opts=dict(
            ylabel='accuracy', xlabel='epoch',
            ytickmin=0, ytickmax=1,
            title='accuracy', showlegend=True))

    best_acc_tracker = summary.Scalar(
        vis, 'accuracy', 'best', opts=dict(
            ylabel='accuracy', xlabel='epoch',
            ytickmin=0, ytickmax=1,
            title='accuracy', showlegend=True))

    @evaluator.on(Events.ITERATION_COMPLETED)
    def transform_ninecrop_output(engine):
        output, target = engine.state.output

        if output.size(0) != target.size(0):
            n = target.size(0)
            npatches = output.size(0) // n
            output = output.view(n, npatches, *output.shape[1:])
            output = torch.mean(output, dim=1)

        engine.state.output = (output, target)

    @trainer.on(Events.EPOCH_STARTED)
    def learning_rate(engine):
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        lr_tracker(engine.state.epoch, lr)

    @trainer.on(Events.ITERATION_COMPLETED)
    def training_loss(engine):
        i = engine.state.iteration/len(trainloader)
        train_loss_tracker(i, engine.state.output)

    @trainer.on(Events.EPOCH_COMPLETED)
    def training_epoch_result(engine):
        evaluator.run(trainloader)

        epoch = engine.state.epoch
        metrics = evaluator.state.metrics

        acc = metrics['accuracy']
        avg_loss = metrics['loss']

        train_avg_loss_tracker(epoch, avg_loss)
        train_acc_tracker(epoch, acc)

    @trainer.on(Events.EPOCH_COMPLETED)
    def test(engine):
        evaluator.run(testloader)

        epoch = engine.state.epoch
        metrics = evaluator.state.metrics

        acc = metrics['accuracy']
        avg_loss = metrics['loss']

        module = net.module if len(FG.devices) > 1 else net
        module.save_if_best(epoch, optimizer, acc, avg_loss)
        if epoch % FG.save_term == 0:
            module.save(epoch, optimizer, acc, avg_loss)

        test_avg_loss_tracker(epoch, avg_loss)
        test_acc_tracker(epoch, acc)
        best_acc_tracker(epoch, module.best_performance)

    trainer.run(trainloader, max_epochs=FG.num_epoch)
    vis.save([vis.env])
    print('Finish the '+str(FG.cur_fold)+' training')

    parser.save()
