from __future__ import print_function, division, absolute_import
import torch
import torchvision
import torch.nn as nn


class infoGenerator2D(nn.Module):
    def __init__(self, z_dim=128, c_code=4, d_code=2):
        super(infoGenerator2D, self).__init__()
        self.z = z_dim
        self.c = c_code
        self.d = d_code

        self.fc1 = nn.Sequential(
            # nn.Linear(self.z+self.c, 256, bias=False),
            nn.Linear(self.z+self.c+self.d, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True))

        self.fc2_up = nn.Sequential(
            nn.Linear(256, 128*6*5, bias=False),
            nn.BatchNorm1d(128*6*5),
            nn.ReLU(True))
        self.fc2_front = nn.Sequential(
            nn.Linear(256, 128*5*5, bias=False),
            nn.BatchNorm1d(128*5*5),
            nn.ReLU(True))

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True))
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True))
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, 1, bias=False),
            nn.Tanh())

    def forward(self, z, c, axis):
        x = torch.cat([z,c],1)
        x = self.fc1(x)
        if axis == 0:
            x = self.fc2_up(x)
            x = x.view(x.size(0), 128, 6, 5)
        elif axis == 1:
            x = self.fc2_front(x)
            x = x.view(x.size(0), 128, 5, 5)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class infoDiscriminator2D(nn.Module):
    def __init__(self, c_code=3, d_code=2):
        super(infoDiscriminator2D, self).__init__()
        self.output_dim = 1
        self.c = c_code
        self.d = d_code

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, 4, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, 4, stride=2,bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, 4, stride=2,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2,  bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))
        self.fc1 = nn.Sequential(
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True))
        # self.fc2 = nn.Linear(256, self.output_dim+self.c, bias=False)
        self.fc2 = nn.Linear(256, self.output_dim+self.c+self.d, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        x_D = torch.sigmoid(x[:, :self.output_dim])
        # c_D = x[:, self.output_dim:]
        c_D = x[:, self.output_dim:self.output_dim+self.c]
        d_D = x[:, self.output_dim+self.c:]
        return x_D, c_D, d_D


class infoGenerator3D(nn.Module):
    def __init__(self, z_dim=128, c_code=4, d_code=2):
        super(infoGenerator3D, self).__init__()
        self.z = z_dim
        self.c = c_code
        self.d = d_code

        self.fc1 = nn.Sequential(
            # nn.Linear(self.z+self.c, 256, bias=False),
            nn.Linear(self.z+self.c+self.d, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True))
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128*5*6*5, bias=False),
            nn.BatchNorm1d(128*5*6*5),
            nn.ReLU(True))
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(True))
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(True))
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(8, 1, 1, bias=False),
            nn.Tanh())

    def forward(self, z, c):
        x = torch.cat([z,c],1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 128, 5, 6, 5)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x



class infoDiscriminator3D(nn.Module):
    def __init__(self, c_code=3, d_code=2):
        super(infoDiscriminator3D, self).__init__()
        self.output_dim = 1
        self.c = c_code
        self.d = d_code

        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 8, 4, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv3d(8, 16, 4, stride=2,bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv3d(16, 32, 4, stride=2,bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv3d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer5 = nn.Sequential(
            nn.Conv3d(64, 128, 4, stride=2,  bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True))
        self.fc1 = nn.Sequential(
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True))
        # self.fc2 = nn.Linear(256, self.output_dim+self.c, bias=False)
        self.fc2 = nn.Linear(256, self.output_dim+self.c+self.d, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        x_D = torch.sigmoid(x[:, :self.output_dim])
        # c_D = x[:, self.output_dim:]
        c_D = x[:, self.output_dim:self.output_dim+self.c]
        d_D = x[:, self.output_dim+self.c:]
        return x_D, c_D, d_D
