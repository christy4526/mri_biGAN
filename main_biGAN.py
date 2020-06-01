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
from models import Generator, Discriminator, E
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.datasets as dset
import torchvision.utils

# project packages
from config import GAN_parser, argument_report
from dataset import ADNIDataset, ADNIDataset2D, Trainset, fold_split
from transforms import ToFloatTensor, Normalize
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
        G_loss = Scalar(vis, 'GE_loss', opts=dict(
            showlegend=True, title='GE loss', ytickmin=0, ytinkmax=10)),
        AC_loss = Scalar(vis, 'AC_loss', opts=dict(
            showlegend=True, title='AC loss', ytickmin=0, ytinkmax=10)),
        info_loss = Scalar(vis, 'info_loss', opts=dict(
            showlegend=True, title='info_loss', ytickmin=0, ytinkmax=10)),
        acc = Scalar(vis, 'Accuracy', opts=dict(
            showlegend=True, title='Accuracy', ytickmin=0, ytinkmax=2.0)),
        ofake = Scalar(vis, 'Fake output', opts=dict(
            showlegend=True, title='Fake output', ytickmin=0, ytinkmax=2.0)),
        oreal = Scalar(vis, 'Real output', opts=dict(
            showlegend=True, title='Real output', ytickmin=0, ytinkmax=2.0)),
        inputs = Image3D(vis, 'inputs'),
        fake = Image3D(vis, 'fake'),
        valid = Image3D(vis, 'valid'),
        outputs = Image3D(vis, 'outputs'),
        outputs2 = Image3D(vis, 'outputs2'))

    x, y = Trainset(FG)      # x = image, y=target
    # x, y, train_idx, test_idx, ratio = fold_split(FG)
    if FG.gm:
        transform=Compose([ToFloatTensor()])
    else :
        transform=Compose([ToFloatTensor(), Normalize(0.5,0.5)])

    # trainset = ADNIDataset2D(FG, x[train_idx], y[train_idx], transform=transform)
    # testset = ADNIDataset2D(FG, x[test_idx], y[test_idx], transform=transform)
    #
    # trainloader = DataLoader(trainset, batch_size=FG.batch_size, shuffle=True,
    #                          pin_memory=True, num_workers=4)
    # testloader = DataLoader(testset, batch_size=FG.batch_size, shuffle=True,
    #                         num_workers=4, pin_memory=True)
    trainset = ADNIDataset2D(FG, x, y, transform=transform)
    trainloader = DataLoader(trainset, batch_size=FG.batch_size,
                             shuffle=True, pin_memory=True)

    D = Discriminator(FG).to('cuda:{}'.format(FG.devices[0]))  # discriminator net D(x, z)
    G = Generator(FG.z_dim, FG.c_code, FG.axis).to('cuda:{}'.format(FG.devices[0]))  # generator net (decoder) G(x|z)
    E = E(FG).to('cuda:{}'.format(FG.devices[0]))  # inference net (encoder) Q(z|x)

    if FG.load_ckpt:
        D.load_state_dict(torch.load(os.path.join(FG.checkpoint_root, save_dir, 'D.pth')))
        G.load_state_dict(torch.load(os.path.join(FG.checkpoint_root, save_dir, 'G.pth')))
        E.load_state_dict(torch.load(os.path.join(FG.checkpoint_root, save_dir, 'E.pth')))
    if len(FG.devices) != 1:
        G = torch.nn.DataParallel(G, FG.devices)
        D = torch.nn.DataParallel(D, FG.devices)
        E = torch.nn.DataParallel(E, FG.devices)

    if FG.wasserstein:
        optimizerD = optim.RMSprop(D.parameters(), lr=FG.lr_rmsprop)
        optimizerGE = optim.RMSprop(itertools.chain(G.parameters(), E.parameters()), lr=FG.lr_rmsprop)
        optimizerinfo = optim.RMSprop(itertools.chain(E.parameters(),G.parameters()),lr=FG.lr_rmsprop)
    else:
        optimizerD = optim.Adam(D.parameters(), lr=FG.lr_adam*5, betas=(FG.beta1, 0.999))
        optimizerGE = optim.Adam(itertools.chain(G.parameters(), E.parameters()),
                                  lr=FG.lr_adam, betas=(FG.beta1, 0.999))
        optimizerinfo = optim.Adam(itertools.chain(G.parameters(), E.parameters()),
                                   lr=FG.lr_adam, betas=(FG.beta1, 0.999))
        optimizerAC = optim.Adam(E.parameters(), lr=FG.lr_adam*0.1, betas=(FG.beta1, 0.999))

    schedularD = ExponentialLR(optimizerD, FG.lr_gamma)
    schedularGE = ExponentialLR(optimizerGE, FG.lr_gamma)
    schedularinfo = ExponentialLR(optimizerinfo, FG.lr_gamma)
    schedularAC = ExponentialLR(optimizerAC, FG.lr_gamma)

    BCEWithLogitsLoss = nn.BCEWithLogitsLoss().to('cuda:{}'.format(FG.devices[0]))
    BCE_loss = nn.BCELoss().to('cuda:{}'.format(FG.devices[0]))
    MSE_loss = nn.MSELoss().to('cuda:{}'.format(FG.devices[0]))

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
        stats = logging.Statistics(['loss_D', 'loss_GE'])
        printers['lr']('D', epoch, optimizerD.param_groups[0]['lr'])
        printers['lr']('GE',  epoch, optimizerGE.param_groups[0]['lr'])
        printers['lr']('info', epoch, optimizerinfo.param_groups[0]['lr'])
        # printers['lr']('AC', epoch, optimizerAC.param_groups[0]['lr'])
        timer.tic()

        if (epoch+1)%100 == 0:
            schedularD.step()
            schedularGE.step()
            schedularinfo.step()
            #schedularAC.step()
        D.train()
        G.train()
        E.train()
        train_scores.clear()

        real_rate = torch.zeros(10)
        fake_rate = torch.zeros(10)
        for step, data in enumerate(trainloader):
            x = data['image']
            y = data['target']
            batch_size = x.size(0)  # batch_size <= FG.batch_size
            D.zero_grad()
            G.zero_grad()
            E.zero_grad()
            y_real = torch.ones(batch_size).cuda(device, non_blocking=True)
            y_fake = torch.zeros(batch_size).cuda(device, non_blocking=True)

            x = x.unsqueeze(dim=1)
            inputs = (x*0.5)+0.5
            printers['inputs']('input', inputs[0,:,:,:])

            """        G network          """
            z_fake = torch.rand(batch_size, FG.z_dim).type(torch.FloatTensor)
            c_fake = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size,\
                                   c_code))).type(torch.FloatTensor)

            class_fake = torch.from_numpy(np.random.uniform(0, 1, size=(batch_size,\
                                   1))).type(torch.FloatTensor)
            C = torch.cat([c_fake, class_fake],1)
            z_fake, c_fake, C = z_fake.cuda(device, non_blocking=True),\
                            c_fake.cuda(device, non_blocking=True),\
                            C.cuda(device, non_blocking=True)

            x_fake_0 = G(z_fake, c_fake, FG.axis)
            # x_fake_0 = G(z_fake, c_fake, d_code[0], FG.axis)
            # x_fake_1 = G(z_fake, c_fake, d_code[1], FG.axis)
            #x_fake = G(z_fake, C, FG.axis)
            printers['outputs']('x_0_fake', x_fake_0[0,:,:,:])
            # printers['outputs2']('x_1_fake', x_fake_1[0,:,:,:])

            """         E network         """
            #x_real = x.type(torch.FloatTensor).cuda(device, non_blocking=True)
            x_real_0 = x.cuda(device, non_blocking=True)
            z_real_0, c_real_0 = E(x_real_0)
            # z_real_1, c_real_1 = E(x_real[1])

            """         D network         """
            # print(x_fake.shape, x_real.shape)
            output_fake_0 = D(x_fake_0, z_fake)
            output_real_0 = D(x_real_0, z_real_0)
            # output_fake_1 = D(x_fake_1, z_fake_1)
            # output_real_1 = D(x_real_1, z_real_1)

            real_rate[step] = torch.mean(output_real_0)
            fake_rate[step] = torch.mean(output_fake_0)
            print('Real : ', real_rate[step].item(), ' Fake : ', fake_rate[step].item())

            loss_D_0 = -torch.mean(torch.log(output_real_0+EPS)+torch.log(1-output_fake_0+EPS))
            loss_GE_0 = -torch.mean(torch.log(1-output_real_0+EPS)+torch.log(output_fake_0+EPS))
            # loss_D_0 = BCE_loss(output_real_0, y_real) + BCE_loss(output_fake_0, y_fake)
            # loss_GE_0 = BCE_loss(output_fake_0, y_real) + BCE_loss(output_real_0, y_fake)

            # Loss_D_1 = -torch.mean(torch.log(output_real_1+EPS)+torch.log(1-output_fake_1+EPS))
            # Loss_GE_1 = -torch.mean(torch.log(1-output_real_1+EPS)+torch.log(output_fake_1+EPS))

            # loss_D = loss_D_0 + loss_D_1
            # loss_GE = loss_GE_0 + loss_GE_1

            loss_D = loss_D_0
            loss_GE = loss_GE_0

            _, c_fake_E_0 = E(x_fake_0)
            c_fake_E_0 = c_fake_E_0[:, :FG.c_code]
            # _, c_fake_E_1 = E(x_fake_1)
            # c_fake_E_1 = c_fake_E_1[:, :FG.c_code]
            # y_class = y.type(torch.FloatTensor).cuda(device, non_blocking=True)
            loss_info = MSE_loss(c_fake_E_0, c_fake)

            loss_D.backward(retain_graph=True)
            optimizerD.step()
            loss_GE.backward(retain_graph=True)
            optimizerGE.step()
            loss_info.backward()
            optimizerinfo.step()

            #y_class = y.reshape(batch_size,1).type(torch.FloatTensor).cuda(device, non_blocking=True)
            #loss_AC = BCEWithLogitsLoss(c_real, y_class)
            # loss_AC.backward()
            # optimizerAC.step()
            # train_scores.update_true(y)
            # train_scores.update_score(score)

            if FG.wasserstein:
                for p in D.parameters():
                    p.data.clamp_(-FG.clamp, FG.clamp)

            # if (epoch+1)%10 == 0:
            #     torchvision.utils.save_image(x, '%s/real_samples.png'%save_dir)
            #     f_C = torch.cat([fixed_c, fixed_class],1)
            #     f_C = f_C.cuda(device, non_blocking=True)
            #     # fake = G(fixed_z[12:22], fixed_c[12:22], FG.axis)
            #     fake = G(fixed_z[:batch_size], f_C[:batch_size], FG.axis)
            #     fake = (fake*0.5)+0.5
            #     torchvision.utils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png'%(save_dir, epoch))

        printers['D_loss']('train', epoch+step/len(trainloader), loss_D)
        printers['G_loss']('train', epoch+step/len(trainloader), loss_GE)
        printers['info_loss']('train', epoch+step/len(trainloader), loss_info)
        printers['ofake']('train', epoch+step/len(trainloader), torch.mean(fake_rate))
        printers['oreal']('train', epoch+step/len(trainloader), torch.mean(real_rate))


        train_acc = train_scores.accuracy
        # printers['acc']('train', epoch+i/len(trainloader), train_acc)
        print("Epoch: [%2d] D_loss: %.4f, G_loss: %.4f, info_loss: %.4f" %
             ((epoch + 1), loss_D.item(), loss_GE.item(), loss_info.item()))

        if (epoch+1)%10 == 0:
            G.eval()
            D.eval()
            E.eval()
            """      G network     """
            C = torch.cat([fixed_c, fixed_class],1)
            valid_x_fake = G(fixed_z, fixed_c, FG.axis)
            # valid_x_fake = G(fixed_z, C, FG.axis)
            fake = (valid_x_fake*0.5)+0.5
            printers['fake']('fake', fake[0,:,:,:])
            if ((epoch+1) % 50 == 0):
                saver = Image3D(vis, 'output_'+str(epoch+1))
                saver('output_'+str(epoch+1), fake[0,:,:,:])

            valid_scores.clear()

            # for j, data in enumerate(testloader):
            #     valid_x = data['image']
            #     valid_y = data['target']
            #
            #     # G network
            #     valid_z_fake = torch.rand(valid_x.shape[0], FG.z_dim).type(torch.FloatTensor).cuda(device, non_blocking=True)
            #     valid_c_fake = torch.from_numpy(np.random.uniform(-1, 1, size=(valid_x.shape[0],\
            #                            FG.c_code))).type(torch.FloatTensor).cuda(device, non_blocking=True)
            #     valid_x_fake = G(valid_z_fake, valid_c_fake, FG.axis)
            #     valid_fake = (valid_x_fake*0.5)+0.5
            #     printers['valid']('valid_fake', valid_fake[0,:,:,:])
            #
            #     if (epoch+1)%50 == 0:
            #         valid_printer_save = Image3D(vis, 'valid_fake_'+str(epoch+1))
            #         valid_printer_save('valid_fake_'+str(epoch+1), valid_fake[0,:,:,:])
            #
            #     # E network
            #     valid_x = valid_x.unsqueeze(dim=1)
            #     valid_x_real = valid_x.type(torch.FloatTensor).cuda(device, non_blocking=True)
            #     ac_real  = torch.zeros((valid_x.shape[0], 2)).type(torch.FloatTensor).cuda(device, non_blocking=True)
            #     for k in range(valid_x.shape[0]):
            #         if valid_y[k] == 0:
            #             ac_real[k][0] = 1
            #         else:
            #             ac_real[k][1] = 1
            #
            #     valid_ac_real  = ac_real.cuda(device, non_blocking=True)
            #     valid_z_real, valid_c_real = E(valid_x_real)
            #
            #     # D network
            #     valid_D_fake = D(valid_x_fake, valid_z_fake)
            #     valid_D_real = D(valid_x_real, valid_z_real)
            #
            #     # loss & back propagation
            #     valid_loss_D = -torch.mean(torch.log(valid_D_real+EPS)+torch.log(1-valid_D_fake+EPS))
            #     valid_loss_GE = -torch.mean(torch.log(valid_D_fake+EPS)+torch.log(1-valid_D_real+EPS))
            #
            #     valid_z_E, valid_c_E = E(valid_fake)
            #     valid_loss_info = MSE_loss(valid_c_E, valid_c_fake)

                # valid_y_class = valid_y.reshape(valid_x.shape[0],1).type(torch.FloatTensor).cuda(device, non_blocking=True)
                # valid_loss_AC = BCEWithLogitsLoss(valid_c_real, valid_y_class)
                # valid_scores.update_true(valid_y)
                # valid_scores.update_score(score)

            # printers['D_loss']('valid', epoch+i/len(testloader), valid_loss_D)
            # printers['G_loss']('valid', epoch+i/len(testloader), valid_loss_GE)
            # # printers['AC_loss']('valid', epoch+i/len(testloader), valid_loss_AC)
            # printers['info_loss']('valid', epoch+i/len(testloader), valid_loss_info)
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
            # result_dict = {"D_loss":loss_D, "G_loss":loss_GE, "AC_loss":loss_AC}
            # result_dict = {"D_loss":loss_D, "G_loss":loss_GE, "info_loss":loss_info}
            #                 "AC_loss":loss_AC}


        if ((epoch+1) % 10 == 0):
            with torch.no_grad():
                torch.save(D.state_dict(), '%s/D_%d.pth' % (save_dir, epoch+1))
                torch.save(G.state_dict(), '%s/G_%d.pth' % (save_dir, epoch+1))
                torch.save(E.state_dict(), '%s/E_%d.pth' % (save_dir, epoch+1))
        # torch.save(D.state_dict(), os.path.join(save_dir, 'D.pth'))
        # torch.save(G.state_dict(), os.path.join(save_dir, 'G.pth'))
        # torch.save(E.state_dict(), os.path.join(save_dir, 'E.pth'))
        timer.toc()
        print('Time elapse {}h {}m {}s'.format(*timer.total()))
        vis.save([vis.env])
        time.sleep(0.5)
    # np.save('stat.npy', stats)
