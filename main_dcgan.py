from __future__ import absolute_import, division, print_function, unicode_literals
import os

# visualizes
from tqdm import tqdm
from visdom import Visdom
import time

# deep learning framework
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Lambda
from torch.utils.data import DataLoader
from models import dcGenerator2D, dcDiscriminator2D
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.datasets as dset
import torchvision.utils

# project packages
from config import GAN_parser, argument_report
from dataset import ADNIDataset, ADNIDataset2D, Trainset, fold_split
from transforms import ToWoldCoordinateSystem, ToTensor, ToFloatTensor, Normalize, Pad
from utils import print_model_parameters, ScoreReport, SimpleTimer, loss_plot
from utils import save_checkpoint
from summary import Scalar, Image3D
import itertools
import numpy as np
import imageio
import shutil
import argparse

#from utils import logging
from torch.autograd import Variable
from sklearn import metrics
from ori_utils import logging


if __name__ == '__main__':
    FG = GAN_parser()
    vis = Visdom(port=FG.vis_port, env=str(FG.vis_env))
    vis.text(argument_report(FG, end='<br>'), win='config')

    save_dir = str(FG.vis_env)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # torch setting
    device = torch.device('cuda:{}'.format(FG.devices[0]))
    torch.cuda.set_device(FG.devices[0])
    timer = SimpleTimer()

    printers = dict(
        lr = Scalar(vis, 'lr', opts=dict(
            showlegend=True, title='lr', ytickmin=0, ytinkmax=2.0)),
        D_loss = Scalar(vis, 'D_loss', opts=dict(
            showlegend=True, title='D loss', ytickmin=0, ytinkmax=2.0)),
        G_loss = Scalar(vis, 'G_loss', opts=dict(
            showlegend=True, title='G loss', ytickmin=0, ytinkmax=10)),
        DG_z1 = Scalar(vis, 'DG_z1', opts=dict(
            showlegend=True, title='DG_z1', ytickmin=0, ytinkmax=2.0)),
        DG_z2 = Scalar(vis, 'DG_z2', opts=dict(
            showlegend=True, title='DG_z2', ytickmin=0, ytinkmax=2.0)),
        D_x = Scalar(vis, 'D_x', opts=dict(
            showlegend=True, title='D_x', ytickmin=0, ytinkmax=2.0)),
        inputs = Image3D(vis, 'inputs'),
        fake = Image3D(vis, 'fake'),
        valid = Image3D(vis, 'valid'),
        outputs = Image3D(vis, 'outputs'),
        outputs2 = Image3D(vis, 'outputs2'))

    x, y = Trainset(FG)      # x = image, y=target
    transform=Compose([ToFloatTensor(), Normalize(0.5,0.5)])
    trainset = ADNIDataset2D(FG, x, y, transform=transform)
    trainloader = DataLoader(trainset, batch_size=FG.batch_size,
                             shuffle=True, pin_memory=True)

    D = dcDiscriminator2D(FG).to('cuda:{}'.format(FG.devices[0]))  # discriminator net D(x, z)
    G = dcGenerator2D(FG.z_dim).to('cuda:{}'.format(FG.devices[0]))  # generator net (decoder) G(x|z)

    if FG.load_ckpt:
        D.load_state_dict(torch.load(os.path.join(FG.checkpoint_root, save_dir, 'D.pth')))
        G.load_state_dict(torch.load(os.path.join(FG.checkpoint_root, save_dir, 'G.pth')))
    if len(FG.devices) != 1:
        G = torch.nn.DataParallel(G, FG.devices)
        D = torch.nn.DataParallel(D, FG.devices)

    optimizerD = optim.Adam(D.parameters(), lr=FG.lrD, betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=FG.lrG, betas=(0.5, 0.999))
    schedularD=torch.optim.lr_scheduler.ExponentialLR(optimizerD, FG.lr_gamma)
    schedularG=torch.optim.lr_scheduler.ExponentialLR(optimizerG, FG.lr_gamma)

    bce_loss = nn.BCELoss().to('cuda:{}'.format(FG.devices[0]))
    fixed_z = torch.rand((FG.batch_size, FG.z_dim)).float().cuda(device, non_blocking=True)

    for epoch in range(FG.num_epochs):
        printers['lr']('D', epoch, optimizerD.param_groups[0]['lr'])
        printers['lr']('G',  epoch, optimizerG.param_groups[0]['lr'])
        timer.tic()

        if (epoch+1)%100 == 0:
            schedularD.step()
            schedularG.step()
        D.train(True)
        G.train(True)
        for step, data in enumerate(trainloader):
            x = data['image']
            y = data['target']
            batch_size = x.size(0)  # batch_size <= FG.batch_size
            D.zero_grad()
            G.zero_grad()

            z = torch.rand(batch_size, FG.z_dim).float().cuda(device, non_blocking=True)
            y_fake = torch.zeros(batch_size).float().cuda(device, non_blocking=True)
            y_real = torch.ones(batch_size).float().cuda(device, non_blocking=True)

            x = x.unsqueeze(dim=1).float().cuda(device, non_blocking=True)
            printers['inputs']('input', (x*0.5+0.5)[0,:,:,:])

            """         Update D network            """
            optimizerD.zero_grad()
            # lossD, D_x, D_G_z1 = D_loss(x, fake.detach(), y_real, y_fake)
            output_real = D(x)
            lossD_real = bce_loss(output_real, y_real)
            lossD_real.backward(retain_graph=True)
            D_x = output_real.data.mean()

            fake = G(z)
            output_fake = D(fake.detach())
            lossD_fake = bce_loss(output_real, y_fake)
            lossD_fake.backward(retain_graph=True)
            D_G_z1 = output_fake.data.mean()
            lossD = lossD_real + lossD_fake
            # lossD.backward()
            optimizerD.step()

            """         Update G network            """
            optimizerG.zero_grad()
            # lossG, fake, D_G_z2 = G_loss(z, y_real)
            output = D(fake)
            lossG = bce_loss(output, y_real)
            lossG.backward()
            D_G_z2 = output.data.mean()
            printers['outputs']('x_fake', fake[0,:,:,:])

            optimizerG.step()

        printers['D_loss']('D_loss', epoch+step/len(trainloader), lossD)
        printers['G_loss']('G_loss', epoch+step/len(trainloader), lossG)
        printers['D_x']('D_x', epoch+step/len(trainloader), D_x)
        printers['DG_z1']('DG_z1', epoch+step/len(trainloader), D_G_z1)
        printers['DG_z2']('DG_z2', epoch+step/len(trainloader), D_G_z2)

        if ((epoch+1) % 10 == 0):
            G.eval()
            fake = G(fixed_z)
            printers['fake']('fake', (fake*0.5+0.5)[0,:,:,:])
            if ((epoch+1) % 50 == 0):
                save_printer = Image3D(vis, 'fake_e'+str(epoch+1))
                save_printer('fake_e'+str(epoch+1), (fake*0.5+0.5)[0,:,:,:])

        print("Epoch %d > DL: %.4f, GL: %.4f,  D_x: %.4f, DG_z1: %.4f DG_z2: %4f" %
             (epoch, lossD.item(),lossG.item(),D_x.item(),D_G_z1.item(),D_G_z2.item()))
        result_dict = {"DL":lossD,"GL":lossG,"D_x":D_x,"D_G_z1":D_G_z1,"D_G_z2":D_G_z2}
        # do checkpointing
        torch.save(G.state_dict(), '%s/G.pth' % (save_dir))
        torch.save(D.state_dict(), '%s/D.pth' % (save_dir))

        timer.toc()
        print('Time elapse {}h {}m {}s'.format(*timer.total()))
        vis.save([vis.env])
