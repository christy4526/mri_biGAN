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
from models import mGenerator, mDiscriminator, mE
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.datasets as dset
import torchvision.utils

# project packages
from config import GAN_parser, argument_report
from dataset import ADNIDataset, ADNIDataset2D, Trainset, fold_split
from transforms import ToFloatTensor, Normalize, ToWoldCoordinateSystem, ToTensor, Pad
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

from ignite.engine import Engine, Events, _prepare_batch
from ignite.metrics import RunningAverage, Accuracy


if __name__ == '__main__':
    FG = GAN_parser()
    if FG.clean_ckpt:
        shutil.rmtree(FG.checkpoint_root)
    if not os.path.exists(FG.checkpoint_root):
        os.makedirs(FG.checkpoint_root, exist_ok=True)
    logger = logging.Logger(FG.checkpoint_root)
    FG.seed = 1
    torch.manual_seed(FG.seed)
    torch.cuda.manual_seed(FG.seed)
    cudnn.benchmark = True
    EPS = 1e-12

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
        AC_loss = Scalar(vis, 'AC_loss', opts=dict(
            showlegend=True, title='AC loss', ytickmin=0, ytinkmax=10)),
        info_loss = Scalar(vis, 'info_loss', opts=dict(
            showlegend=True, title='info_loss', ytickmin=0, ytinkmax=10)),
        acc = Scalar(vis, 'Accuracy', opts=dict(
            showlegend=True, title='Accuracy', ytickmin=0, ytinkmax=2.0)),
        DG_z1 = Scalar(vis, 'DG_z1', opts=dict(
            showlegend=True, title='DG_z1', ytickmin=0, ytinkmax=2.0)),
        DG_z2 = Scalar(vis, 'DG_z2', opts=dict(
            showlegend=True, title='DG_z2', ytickmin=0, ytinkmax=2.0)),
        D_x = Scalar(vis, 'D_x', opts=dict(
            showlegend=True, title='D_x', ytickmin=0, ytinkmax=2.0)),
        ofake = Scalar(vis, 'Fake output', opts=dict(
            showlegend=True, title='Fake output', ytickmin=0, ytinkmax=2.0)),
        oreal = Scalar(vis, 'Real output', opts=dict(
            showlegend=True, title='Real output', ytickmin=0, ytinkmax=2.0)),
        inputs = Image3D(vis, 'inputs'),
        fake = Image3D(vis, 'fake'),
        valid = Image3D(vis, 'valid'),
        outputs = Image3D(vis, 'outputs'),
        outputs2 = Image3D(vis, 'outputs2'))

    # x, y = Trainset(FG)      # x = image, y=target
    x, y, train_idx, test_idx, ratio = fold_split(FG)
    # transform=Compose([ToFloatTensor(), Normalize(0.5,0.5)])
    # trainset = ADNIDataset2D(FG, x[train_idx], y[train_idx], transform=transform)
    # testset = ADNIDataset2D(FG, x[test_idx], y[test_idx], transform=transform)

    transform=Compose([ToWoldCoordinateSystem(), ToTensor(), Pad(1,0,1,0,1,0), Normalize(0.5,0.5)])
    trainset = ADNIDataset(FG, x[train_idx], y[train_idx], transform=transform)
    testset = ADNIDataset(FG, x[test_idx], y[test_idx], transform=transform)

    trainloader = DataLoader(trainset, batch_size=FG.batch_size, shuffle=True,
                             pin_memory=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=FG.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    # trainset = ADNIDataset2D(FG, x, y, transform=transform)
    # trainloader = DataLoader(trainset, batch_size=FG.batch_size,
    #                          shuffle=True, pin_memory=True)

    D = mDiscriminator(FG, FG.c_code).to('cuda:{}'.format(FG.devices[0]))  # discriminator net D(x, z)
    G = mGenerator(FG.z_dim, FG.c_code, FG.axis).to('cuda:{}'.format(FG.devices[0]))  # generator net (decoder) G(x|z)
    E = mE(FG).to('cuda:{}'.format(FG.devices[0]))  # inference net (encoder) Q(z|x)

    if FG.load_ckpt:
        D.load_state_dict(torch.load(os.path.join(FG.checkpoint_root, save_dir, 'D.pth')))
        G.load_state_dict(torch.load(os.path.join(FG.checkpoint_root, save_dir, 'G.pth')))
        E.load_state_dict(torch.load(os.path.join(FG.checkpoint_root, save_dir, 'E.pth')))
    if len(FG.devices) != 1:
        G = torch.nn.DataParallel(G, FG.devices)
        D = torch.nn.DataParallel(D, FG.devices)
        E = torch.nn.DataParallel(E, FG.devices)


    optimizerD = optim.Adam(D.parameters(), lr=FG.lr_adam*5, betas=(FG.beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=FG.lr_adam, betas=(FG.beta1, 0.999))
    optimizerinfo = optim.Adam(itertools.chain(G.parameters(), D.parameters()),
                               lr=FG.lr_adam, betas=(FG.beta1, 0.999))
    optimizerAC = optim.Adam(itertools.chain(G.parameters(), E.parameters()),
                            lr=FG.lrE, betas=(FG.beta1, 0.999))

    schedularD = ExponentialLR(optimizerD, FG.lr_gamma)
    schedularG = ExponentialLR(optimizerG, FG.lr_gamma)
    schedularinfo = ExponentialLR(optimizerinfo, FG.lr_gamma)
    schedularAC = ExponentialLR(optimizerAC, FG.lr_gamma)

    bce_loss = nn.BCELoss().to('cuda:{}'.format(FG.devices[0]))
    mse_loss = nn.MSELoss().to('cuda:{}'.format(FG.devices[0]))
    ce_loss = nn.CrossEntropyLoss().to('cuda:{}'.format(FG.devices[0]))

    def D_loss(real, fake, y_real, y_fake):
        output_real, _ = D(real)
        lossD_real = bce_loss(output_real, y_real)
        output_fake, _ = D(fake)
        lossD_fake = bce_loss(output_real, y_fake)
        loss_D = lossD_real + lossD_fake
        D_x = output_real.data.mean()
        D_G_z1 = output_fake.data.mean()
        return loss_D, D_x, D_G_z1

    def G_loss(z, c, axis, y_real, name):
        fake = G(z, c, axis)
        printers['outputs'](name, fake[0,:,:,:])
        output, c_D = D(fake)
        loss_G = bce_loss(output, y_real)
        loss_info = mse_loss(c_D, c)
        D_G_z2 = output.data.mean()
        return loss_G, loss_info, D_G_z2

    def E_loss(real, fake, y_real, y_fake):
        output = E(real)
        ac_real = ce_loss(output, y_real)
        output = E(fake)
        ac_fake = ce_loss(output, y_fake)
        return ac_real+ac_fake


    d_code = torch.tensor([0,1])
    c_code = FG.c_code  # gaussian distribution (e.g. rotation, thickness)
    fixed_z = torch.rand((FG.batch_size, FG.z_dim)).type(torch.FloatTensor).cuda(device, non_blocking=True)
    fixed_c = torch.from_numpy(np.random.uniform(-1, 1, size=(FG.batch_size,\
                           c_code))).type(torch.FloatTensor).cuda(device, non_blocking=True)
    fixed_class = torch.from_numpy(np.random.uniform(0, 1, size=(FG.batch_size,\
                           1))).type(torch.FloatTensor).cuda(device, non_blocking=True)

    train_scores = ScoreReport()
    valid_scores = ScoreReport()
    max_acc = 0
    min_loss = 0
    for epoch in range(FG.num_epochs):
        stats = logging.Statistics(['loss_D', 'loss_G'])
        printers['lr']('D', epoch, optimizerD.param_groups[0]['lr'])
        printers['lr']('G',  epoch, optimizerG.param_groups[0]['lr'])
        printers['lr']('info', epoch, optimizerinfo.param_groups[0]['lr'])
        printers['lr']('AC', epoch, optimizerAC.param_groups[0]['lr'])
        timer.tic()

        if (epoch+1)%100 == 0:
            schedularD.step()
            schedularG.step()
            schedularinfo.step()
            schedularAC.step()
        D.train()
        G.train()
        E.train()
        train_scores.clear()

        real_rate = torch.zeros(10)
        fake_rate = torch.zeros(10)
        for step, data in enumerate(trainloader):
            D.zero_grad()
            G.zero_grad()
            E.zero_grad()
            x = data['image']
            y = data['target']
            batch_size = x.size(0)  # batch_size <= FG.batch_size
            label = y.float().cuda(device, non_blocking=True)
            y_real = torch.ones(batch_size).cuda(device, non_blocking=True)
            y_fake = torch.zeros(batch_size).cuda(device, non_blocking=True)

            # x = x.unsqueeze(dim=1).float().cuda(device, non_blocking=True)
            inputs = (x*0.5)+0.5
            printers['inputs']('input', inputs)

            z = torch.rand(batch_size, FG.z_dim).float().cuda(device, non_blocking=True)
            c = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size,\
                                   c_code))).float().cuda(device, non_blocking=True)
            class_fake = torch.from_numpy(np.random.uniform(0, 1, size=(batch_size,\
                                   1))).float().cuda(device, non_blocking=True)
            C = torch.cat([c, class_fake],1).cuda(device, non_blocking=True)

            """         D network         """
            optimizerD.zero_grad()
            loss_D_0, D_x_0, D_G_z1_0 = D_loss(x[:,:,42,:,:], G(z,C,0), y_real, y_fake)
            loss_D_1, D_x_1, D_G_z1_1 = D_loss(x[:,:,:,48,:], G(z,C,1), y_real, y_fake)
            loss_D = loss_D_0 + loss_D_1
            loss_D.backward(retain_graph=True)
            optimizerD.step()

            """         G network         """
            optimizerG.zero_grad()
            loss_G_0, loss_info_0, D_G_z2_0 = G_loss(z, C, 0, y_real, 'x_fake_0')
            loss_G_1, loss_info_1, D_G_z2_1 = G_loss(z, C, 1, y_real, 'x_fake_1')
            loss_G = loss_G_0 + loss_G_1
            loss_info = loss_info_0 + loss_info_1
            loss_G.backward(retain_graph=True)
            optimizerG.step()
            loss_info.backward(retain_graph=True)
            optimizerinfo.step()

            """         E network         """
            optimizerAC.zero_grad()
            loss_AC_0 = E_loss(x[:,42,:,:], fake, label, class_fake)
            loss_AC_1 = E_loss(x[:,:,48,:], fake, label, class_fake)
            loss_AC = loss_AC_0+loss_AC_1

            loss_AC.backward()
            optimizerAC.step()

        printers['D_loss']('train', epoch+step/len(trainloader), loss_D)
        printers['G_loss']('train', epoch+step/len(trainloader), loss_G)
        printers['info_loss']('train', epoch+step/len(trainloader), loss_info)
        printers['AC_loss']('ac_real', epoch+step/len(trainloader), AC_real)
        printers['AC_loss']('ac_fake', epoch+step/len(trainloader), AC_fake)
        printers['AC_loss']('train', epoch+step/len(trainloader), loss_AC)
        printers['ofake']('train', epoch+step/len(trainloader), torch.mean(fake_rate))
        printers['oreal']('train', epoch+step/len(trainloader), torch.mean(real_rate))
        printers['DG_z1']('DG_z1', epoch+step/len(trainloader), D_G_z1)
        printers['DG_z2']('DG_z2', epoch+step/len(trainloader), D_G_z2)
        printers['D_x']('D_x', epoch+step/len(trainloader), D_x)


        train_acc = train_scores.accuracy
        # printers['acc']('train', epoch+i/len(trainloader), train_acc)
        print("Epoch: [%2d] D_loss: %.4f, G_loss: %.4f, info_loss: %.4f, info_AC: %.4f" %
             ((epoch + 1), loss_D.item(), loss_G.item(), loss_info.item(), loss_AC.item()))

        if (epoch+1)%10 == 0:
            G.eval()
            D.eval()
            E.eval()
            # valid_scores.clear()
            for j, data in enumerate(testloader):
                valid_x = data['image']
                valid_y = data['target']
                batch_size = valid_x.size(0)

                valid_z = torch.rand(batch_size, FG.z_dim).float().cuda(device, non_blocking=True)
                valid_c = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size,\
                                       c_code))).float().cuda(device, non_blocking=True)
                valid_class = torch.from_numpy(np.random.uniform(0, 1, size=(batch_size,\
                                       1))).float().cuda(device, non_blocking=True)
                valid_label = valid_y.float().cuda(device, non_blocking=True)
                valid_y_real = torch.ones(batch_size).cuda(device, non_blocking=True)
                valid_y_fake = torch.zeros(batch_size).cuda(device, non_blocking=True)
                valid_C = torch.cat([valid_c, valid_class],1).cuda(device, non_blocking=True)

                # D network
                valid_output_real, _ = D(valid_x[:,42,:,:])
                valid_fake = G(valid_z, valid_C, 0)
                valid_output_fake, valid_c_D = D(valid_fake)
                valid_loss_D_0 = BCE_loss(valid_output_real, valid_y_real)+BCE_loss(valid_output_fake, valid_y_fake)
                valid_loss_G_0 = BCE_loss(valid_output_fake, valid_y_real)
                valid_loss_info_0 = MSE_loss(valid_c_D, valid_c)

                valid_output_real, _ = D(valid_x[:,:,48,:])
                valid_fake2 = G(valid_z, valid_C, 1)
                valid_output_fake, valid_c_D = D(valid_fake2)
                valid_loss_D_1 = BCE_loss(valid_output_real, valid_y_real)+BCE_loss(valid_output_fake, valid_y_fake2)
                valid_loss_G_1 = BCE_loss(valid_output_fake, valid_y_real)

                valid_loss_G = valid_loss_G_0+valid_loss_G_1
                valid_loss_info_1 = MSE_loss(valid_c_D, valid_c)
                valid_loss_info = valid_loss_info_0 + valid_loss_info_1

                valid_fake = G(valid_z, valid_C, 0)
                printers['valid']('valid_fake', (valid_fake*0.5+0.5)[0,:,:,:])
                valid_fake = G(valid_z, valid_C)
                printers['valid']('valid_fake', (valid_fake*0.5+0.5)[0,:,:,:])

                if (epoch+1)%50 == 0:
                    valid_printer_save = Image3D(vis, 'valid_fake_'+str(epoch+1))
                    valid_printer_save('valid_fake_'+str(epoch+1), (valid_fake*0.5+0.5)[0,:,:,:])

                # E network
                valid_x = valid_x.unsqueeze(dim=1).float().cuda(device, non_blocking=True)
                valid_ac_r = F.softmax(E(valid_x), dim=1)
                valid_yp = torch.argmax(valid_ac_r, dim=1).float()

                valid_ac_f = F.softmax(E(valid_fake), dim=1)
                valid_yp_f = torch.zeros((batch_size, 1)).float().cuda(device, non_blocking=True)
                for k in range(batch_size):
                    valid_yp_f[k,:] = valid_ac_f[k].max()

                ac_r = E(x[:,42,:,:])
                AC_real = CELoss(ac_r, label)

                ac_f = E(fake)
                AC_fake = CELoss(ac_f, class_fake)
                loss_AC_0 = AC_real+AC_fake

                ac_r = E(x[:,:,48,:])
                AC_real = CELoss(ac_r, label)

                ac_f = E(fake2)
                AC_fake = CELoss(ac_f, class_fake)
                loss_AC_1 = AC_real+AC_fake

                loss_AC = loss_AC_0+loss_AC_1

                # valid_loss_AC = BCEWithLogitsLoss(valid_yp, valid_label)+BCEWithLogitsLoss(valid_yp_f, valid_class)
                valid_loss_AC = BCEWithLogitsLoss(valid_yp, valid_label)

            printers['D_loss']('valid', epoch+step/len(testloader), valid_loss_D)
            printers['G_loss']('valid', epoch+step/len(testloader), valid_loss_G)
            printers['AC_loss']('valid', epoch+step/len(testloader), valid_loss_AC)
            printers['info_loss']('valid', epoch+step/len(testloader), valid_loss_info)
            # valid_acc = valid_scores.accuracy
            #printers['acc']('valid', epoch+i/len(testloader), valid_acc)

            # if valid_acc > max_acc:
            #     min_loss = valid_loss_AC
            #     max_acc = valid_acc
            #     training_state = dict(
            #         epoch=epoch, best_score=dict(loss=min_loss, acc=max_acc),
            #         #state_dict=AC.module.state_dict(),
            #         optimizer_state_dict=optimizerAC.state_dict())
            #     #save_checkpoint(FG, FG.model, training_state, is_best=True)
            #     fname = FG.vis_env
            #     save_checkpoint(FG, fname, training_state, is_best=True)
            # result_dict = {"D_loss":loss_D, "G_loss":loss_G, "AC_loss":loss_AC}
            result_dict = {"D_loss":loss_D, "G_loss":loss_G, "info_loss":loss_info,
                            "AC_loss":loss_AC}


        if ((epoch+1) % 10 == 0):
            with torch.no_grad():
                torch.save(D.state_dict(), '%s/D_%d.pth' % (save_dir, epoch+1))
                torch.save(G.state_dict(), '%s/G_%d.pth' % (save_dir, epoch+1))
                torch.save(E.state_dict(), '%s/E_%d.pth' % (save_dir, epoch+1))
        timer.toc()
        print('Time elapse {}h {}m {}s'.format(*timer.total()))
        vis.save([vis.env])
        time.sleep(0.5)
    # np.save('stat.npy', stats)
