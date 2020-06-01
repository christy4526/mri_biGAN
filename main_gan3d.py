from __future__ import absolute_import, division, print_function, unicode_literals
import os

# visualizes
from tqdm import tqdm
from visdom import Visdom
import time

# deep learning framework
import torch
import torch.nn as nn
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from models import infoGenerator3D, infoDiscriminator3D

import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.utils

# project packages
from config import GAN_parser, argument_report
from dataset import ADNIDataset, Trainset, fold_split
from transforms import Normalize, ToWoldCoordinateSystem, ToTensor, Pad
from utils import save_checkpoint, print_model_parameters, ScoreReport, SimpleTimer
from summary import Scalar, Image3D
import itertools
import numpy as np
import imageio
import shutil
import argparse



if __name__ == '__main__':
    FG = GAN_parser()
    vis = Visdom(port=FG.vis_port, env=str(FG.vis_env))
    vis.text(argument_report(FG, end='<br>'), win='config')

    # torch setting
    device = torch.device('cuda:{}'.format(FG.devices[0]))
    torch.cuda.set_device(FG.devices[0])
    timer = SimpleTimer()

    #save dir setting
    save_dir = str(FG.vis_env)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #visdom printer setting
    printers = dict(
        lr = Scalar(vis, 'lr', opts=dict(
            showlegend=True, title='lr', ytickmin=0, ytinkmax=2.0)),
        D_loss = Scalar(vis, 'D_loss', opts=dict(
            showlegend=True, title='D loss', ytickmin=0, ytinkmax=2.0)),
        G_loss = Scalar(vis, 'G_loss', opts=dict(
            showlegend=True, title='G loss', ytickmin=0, ytinkmax=10)),
        AC_loss = Scalar(vis, 'AC_loss', opts=dict(
            showlegend=True, title='AC loss', ytickmin=0, ytinkmax=10)),
        info_loss = Scalar(vis, 'info_loss', opts=dict(
            showlegend=True, title='info_loss', ytickmin=0, ytinkmax=10)),
        DG_z1 = Scalar(vis, 'DG_z1', opts=dict(
            showlegend=True, title='DG_z1', ytickmin=0, ytinkmax=2.0)),
        DG_z2 = Scalar(vis, 'DG_z2', opts=dict(
            showlegend=True, title='DG_z2', ytickmin=0, ytinkmax=2.0)),
        D_x = Scalar(vis, 'D_x', opts=dict(
            showlegend=True, title='D_x', ytickmin=0, ytinkmax=2.0)),
        inputs0 = Image3D(vis, 'inputs0'),
        inputs1 = Image3D(vis, 'inputs1'),
        fake0 = Image3D(vis, 'fake0'),
        fake1 = Image3D(vis, 'fake1'),
        outputs0 = Image3D(vis, 'outputs0'),
        outputs1 = Image3D(vis, 'outputs1'))

    # dataset setting
    x, y = Trainset(FG)
    # x, y, train_idx, test_idx, ratio = fold_split(FG)
    # transform = Compose([ToFloatTensor(), Normalize(0.5,0.5)])
    # trainset = ADNIDataset2D(FG, x, y, transform=transform)
    transform=Compose([ToWoldCoordinateSystem(), ToTensor(), Pad(1,0,1,0,1,0), Normalize(0.5,0.5)])
    trainset = ADNIDataset(FG, x, y, transform=transform)
    trainloader = DataLoader(trainset, batch_size=FG.batch_size,
                             shuffle=True, pin_memory=True)
    # trainset = ADNIDataset2D(FG, x[train_idx], y[train_idx], transform=transform)
    # testset = ADNIDataset2D(FG, x[test_idx], y[test_idx], transform=transform)
    # trainloader = DataLoader(trainset, batch_size=FG.batch_size, shuffle=True,
    #                          pin_memory=True, num_workers=4)
    # testloader = DataLoader(testset, batch_size=FG.batch_size, shuffle=True,
    #                         num_workers=4, pin_memory=True)

    # models
    D = infoDiscriminator3D(FG.c_code).to('cuda:{}'.format(FG.devices[0]))
    G = infoGenerator3D(FG.z_dim, FG.c_code).to('cuda:{}'.format(FG.devices[0]))

    if len(FG.devices) != 1:
        D = torch.nn.DataParallel(D, FG.devices)
        G = torch.nn.DataParallel(G, FG.devices)
    if FG.load_kpt:
        D.load_state_dict(torch.load(os.path.join(FG.checkpoint_root, save_dir, 'D.pth')))
        G.load_state_dict(torch.load(os.path.join(FG.checkpoint_root, save_dir, 'G.pth')))

    # optimizer
    optimizerD = optim.Adam(D.parameters(), lr=FG.lrD*5, betas=(FG.beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=FG.lrG, betas=(FG.beta1, 0.999))
    optimizerinfo = optim.Adam(itertools.chain(G.parameters(), D.parameters()),
                               lr=FG.lrD, betas=(FG.beta1, 0.999))
    optimizerAC = optim.Adam(itertools.chain(G.parameters(), D.parameters()),
                               lr=FG.lrD, betas=(FG.beta1, 0.999))
    schedularD = ExponentialLR(optimizerD, FG.lr_gamma)
    schedularG = ExponentialLR(optimizerG, FG.lr_gamma)
    schedularinfo = ExponentialLR(optimizerinfo, FG.lr_gamma)
    schedularinfo = ExponentialLR(optimizerAC, FG.lr_gamma)

    # loss function
    bce_loss = nn.BCELoss().to('cuda:{}'.format(FG.devices[0]))
    mse_loss = nn.MSELoss().to('cuda:{}'.format(FG.devices[0]))
    ce_loss = nn.CrossEntropyLoss().to('cuda:{}'.format(FG.devices[0]))

    # fixed noise & condition
    c_code = FG.c_code  # gaussian distribution (e.g. rotation, thickness)
    d_code = len(FG.labels)
    fixed_z = torch.rand((FG.batch_size, FG.z_dim)).type(torch.FloatTensor).cuda(device, non_blocking=True)
    fixed_c = torch.from_numpy(np.random.uniform(-1, 1, size=(FG.batch_size,\
                           c_code))).float().cuda(device, non_blocking=True)
    fixed_d = torch.torch.from_numpy(np.random.multinomial(1, d_code*[float(1.0/d_code)],
                              size=[FG.batch_size])).float().cuda(device, non_blocking=True)

    def D_loss(real, fake, y_real, y_fake, real_label, fake_label):
        output_real, _, d_D = D(real)
        lossD_real = bce_loss(output_real, y_real)
        lossAC_real = ce_loss(d_D, torch.max(real_label,1)[1])
        D_x = output_real.data.mean()

        output_fake, _, d_D = D(fake)
        lossD_fake = bce_loss(output_fake, y_fake)
        lossAC_fake = ce_loss(d_D, torch.max(fake_label,1)[1])
        D_G_z1 = output_fake.data.mean()

        loss_D = lossD_real + lossD_fake
        loss_AC = lossAC_real + lossAC_fake
        return loss_D, loss_AC, D_x, D_G_z1

    def G_loss(fake, c, y_real):
        output, c_D, _ = D(fake)
        lossG = bce_loss(output, y_real)
        lossinfo = mse_loss(c_D, c)
        D_G_z2 = output.data.mean()
        return lossG, lossinfo, D_G_z2

    # start training
    D.train()
    for epoch in range(FG.num_epochs):
        printers['lr']('D', epoch, optimizerD.param_groups[0]['lr'])
        printers['lr']('G',  epoch, optimizerG.param_groups[0]['lr'])
        printers['lr']('info', epoch, optimizerinfo.param_groups[0]['lr'])
        timer.tic()
        #lr schedular
        if (epoch+1)%100 == 0:
            schedularD.step()
            schedularG.step()
            schedularinfo.step()

        G.train()
        for step, data in enumerate(trainloader):
            D.zero_grad()
            G.zero_grad()
            x = data['image'].float().cuda(device, non_blocking=True)
            y = data['target'].float().cuda(device, non_blocking=True)
            batch_size = x.size(0)

            # set veriables
            z = torch.rand(batch_size, FG.z_dim).float().cuda(device, non_blocking=True)
            c = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size,\
                                   c_code))).float().cuda(device, non_blocking=True)
            d = torch.from_numpy(np.random.multinomial(1, d_code*[float(1.0/d_code)],
                                      size=[batch_size])).float().cuda(device, non_blocking=True)
            C = torch.cat([c, d],1).cuda(device, non_blocking=True)
            y_real = torch.ones(batch_size,1).cuda(device, non_blocking=True)
            y_fake = torch.zeros(batch_size,1).cuda(device, non_blocking=True)
            label = torch.zeros(batch_size, 2).float().cuda(device, non_blocking=True)
            for i in range(batch_size):
                if y[i] == 0:
                    label[i,0]=1
                else:
                    label[i,1]=1

            printers['inputs0']('input0', (x*0.5+0.5)[[0],:,:,:,:])

            # D networks
            optimizerD.zero_grad()
            fake = G(z, C)
            printers['outputs0']('train_fake', fake[[0],:,:,:,:])
            lossD, lossAC, D_x, D_G_z1 = D_loss(x, fake, y_real, y_fake, label, d)

            lossD.backward(retain_graph=True)
            optimizerD.step()
            lossAC.backward(retain_graph=True)
            optimizerAC.step()

            # G network
            optimizerG.zero_grad()
            optimizerinfo.zero_grad()
            lossG, lossinfo, D_G_z2 = G_loss(fake, c, y_real)

            lossG.backward(retain_graph=True)
            optimizerG.step()
            lossinfo.backward()
            optimizerinfo.step()

        printers['D_loss']('train', epoch+step/len(trainloader), lossD)
        printers['G_loss']('train', epoch+step/len(trainloader), lossG)
        printers['AC_loss']('train', epoch+step/len(trainloader), lossAC)
        printers['info_loss']('train', epoch+step/len(trainloader), lossinfo)
        printers['DG_z1']('3d', epoch+step/len(trainloader), D_G_z1)
        printers['DG_z2']('3d', epoch+step/len(trainloader), D_G_z2)
        printers['D_x']('3d', epoch+step/len(trainloader), D_x)

        # G.eval & check fake image
        if (epoch+1)%10 == 0:
            G.eval()
            fixed_C = torch.cat([fixed_c, fixed_d],1).cuda(device, non_blocking=True)
            fake = G(fixed_z, fixed_C)
            printers['fake0']('vla_fake', (fake*0.5+0.5)[[0],:,:,:,:])
            if ((epoch+1) % 50 == 0):
                save_printer0 = Image3D(vis, 'fake_e'+str(epoch+1))
                save_printer0('fake_e'+str(epoch+1), (fake*0.5+0.5)[[0],:,:,:,:])

        print("Epoch %d> DL: %.4f, GL: %.4f,  IL: %.4f, ACL: %4f" %
             (epoch, lossD.item(),lossG.item(),lossinfo.item(),lossAC.item()))
        result_dict = {"DL":lossD,"GL":lossG,"IL":lossinfo,"ACL":lossAC}
        # do checkpointing
        torch.save(G.state_dict(), '%s/G.pth' % (save_dir))
        torch.save(D.state_dict(), '%s/D.pth' % (save_dir))

        timer.toc()
        print('Time elapse {}h {}m {}s'.format(*timer.total()))
        vis.save([vis.env])
