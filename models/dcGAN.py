from __future__ import print_function, division, absolute_import

import torch
import torchvision
import torch.nn as nn
from config import train_args, argument_report


class dcGenerator(nn.Module):
    def __init__(self, z_dim=128):
        super(dcGenerator, self).__init__()
        self.z_dim = z_dim

        """
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, 2, 0, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(True))
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(True))
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(16, 1, 4, 2, 1, bias=False),
            nn.Tanh())
        """
        self.layer1 = nn.Linear(self.z_dim, 256*5*6*5)
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 3, 2, 0, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True))
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, 2, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(True))
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, 2, 2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(True))
        self.layer6 = nn.Sequential(
            nn.ConvTranspose3d(16, 1, 1, 1, 0, bias=False),
            nn.Tanh())
        """
        self.input_dim = FG.z
        self.discrete_code = FG.d_code
        self.continuous_code = FG.c_code
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim+self.discrete_code+self.continuous_code, 512),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256*5*6*5))

        self.layer1 = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, 3, 2, 0, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, 2, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(True))
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, 2, 2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(True))
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(16, 1, 1, 1, 0, bias=False),
            nn.ReLU(True),
            nn.Tanh())
        """

    def forward(self, x):
        #cont_code, dist_code = FG.c_code, FG.d_code
        #x = torch.cat([x, cont_code, dist_code],1)
        #print(x.shape, cont_code.shape, dist_code.shape)

        # x = self.fc1(x)
        # # print('Gf1: ', x.shape)
        # x = self.fc2(x)
        # print('Gf2: ', x.shape)
        # x = x.view(-1,256,5,6,5)


        x = self.layer1(x)
        x = x.view(-1,256,5,6,5)
        # print('G1: ', x.shape)
        x = self.layer2(x)
        # print('G2: ', x.shape)
        x = self.layer3(x)
        # print('G3: ', x.shape)
        x = self.layer4(x)
        # print('G4: ', x.shape)
        x = self.layer5(x)
        # print('G5: ', x.shape)
        x = self.layer6(x)
        #x = torch.clamp(x, min=0, max=1)
        # print('G6: ', x.shape)
        # exit()
        """
        x = self.layer7(x)
        # print('G7: ', x.shape)
        x = self.layer8(x)
        print('G8: ', x.shape)
        x = self.layer9(x)
        print('G9: ', x.shape)
        """
        return x


class dcDiscriminator(nn.Module):
    def __init__(self, FG):
        super(dcDiscriminator, self).__init__()
        """
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv3d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv3d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer5 = nn.Sequential(
            nn.Conv3d(128, 1, 4, 2, 0, bias=False),
            nn.Sigmoid())
        """
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, 2, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer5 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer6 = nn.Sequential(
            nn.Conv3d(256, 1, 3, 2, 0, bias=False),
            nn.Sigmoid())
        """
        self.input_dim = 1
        self.output_dim = 1
        self.input_size = 79
        self.discrete_code = FG.d_code   # categorical distribution (i.e. label)
        self.continuous_code = FG.c_code # gaussian distribution (e.g. rotation, thickness)
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, 2, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer5 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True))
        self.fc1 = nn.Sequential(
            nn.Linear(256*3*3*3, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2))
        self.fc2 = nn.Sequential(
            nn.Linear(512,self.output_dim+self.continuous_code+self.discrete_code),
            # nn.Sigmoid(),
        )
        """
    def forward(self, x):
        # print('Di: ', x.shape)
        x = self.layer1(x)
        # print('D1: ', x.shape)
        x = self.layer2(x)
        # print('D2: ', x.shape)
        x = self.layer3(x)
        # print('D3: ', x.shape)
        x = self.layer4(x)
        # print('D4: ', x.shape)
        x = self.layer5(x)
        # print('D5: ', x.shape)
        x = self.layer6(x)
        # print('D6: ', x.shape)
        """
        x = self.layer7(x)
        print('D7: ', x.shape)
        x = self.layer8(x)
        print('D8: ', x.shape)
        x = self.layer9(x)
        print('D9: ', x.shape)
        """
        return x.view(-1, 1).squeeze(1)


class dcGenerator2D(nn.Module):
    def __init__(self, z_dim=128):
        super(dcGenerator2D, self).__init__()
        self.z_dim = z_dim

        self.layer1 = nn.Sequential(
            nn.Linear(self.z_dim, 128*6*5, bias=False),
            nn.BatchNorm1d(128*6*5),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True))
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True))
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True))
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, 1, 1, 0, bias=False),
            nn.Tanh())

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), 128,6,5)
        # print('G1: ', x.shape)
        x = self.layer2(x)
        # print('G2: ', x.shape)
        x = self.layer3(x)
        # print('G3: ', x.shape)
        x = self.layer4(x)
        # print('G4: ', x.shape)
        x = self.layer5(x)
        # print('G5: ', x.shape)
        x = self.layer6(x)
        # print('G6: ', x.shape)
        # exit()
        return x


class dcDiscriminator2D(nn.Module):
    def __init__(self, FG):
        super(dcDiscriminator2D, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 1, 4, 2, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        # print('Di: ', x.shape)
        x = self.layer1(x)
        # print('D1: ', x.shape)
        x = self.layer2(x)
        # print('D2: ', x.shape)
        x = self.layer3(x)
        # print('D3: ', x.shape)
        x = self.layer4(x)
        # print('D4: ', x.shape)
        x = self.layer5(x)
        # print('D5: ', x.shape)
        x = self.layer6(x)
        # print('D6: ', x.shape)
        return x.view(-1, 1).squeeze(1)
