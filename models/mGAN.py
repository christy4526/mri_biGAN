from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtool.networks import NetworkBase
from torchtool.layers.conv import Conv2d, Conv3d


class Block(nn.Module):
    def __init__(self, _in, _out):
        super(Block, self).__init__()
        self.stride = _out // _in
        assert self.stride in (1, 2)

        self.conv1 = Conv2d(_in, _out, 3, self.stride, 1)
        self.conv2 = Conv2d(_in, _out, 3, 1, 1, activation=None)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)

        if self.stride == 1:
            x += residual

        x = F.relu(x, inplace=True)
        return x


class BBlock(nn.Module):
    def __init__(self, _in, _m, _out):
        super(BBlock, self).__init__()
        self.stride = _out // _in
        assert self.stride in (1, 2)

        self.sq = Conv2d(_in, _m, 1, 1, 0)
        self.conv = Conv2d(_m, _m, 3, self.stride, 1)
        self.ex = Conv2d(_m, _out, 1, 1, 0, activation=None)

    def forward(self, x):
        residual = x
        x = self.sq(x)
        x = self.conv(x)
        x = self.ex(x)

        if self.stride == 1:
            x += residual

        x = F.relu(x, inplace=True)
        return x


class ConvT2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, dilation=1,
                 batch_norm=True, activation=None):
        super(ConvT2d, self).__init__()
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                              stride, padding, output_padding, groups, not batch_norm, dilation)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation

    def forward(self, x):
        x = self.convT(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 batch_norm=True, activation=None):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias=not batch_norm)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class mGenerator(nn.Module):
    """ Generator net (decoder, fake) P(x|z) """
    def __init__(self, z_dim=128, c_code=4, axis=1):
        super(mGenerator, self).__init__()
        self.vsize = 0
        self.out_dim = 1
        self.axis = axis
        self.z = z_dim
        self.c = c_code

        if self.axis == 0 or self.axis == 2:
            self.vsize = 6*5
        elif self.axis ==1:
            self.vsize = 5*5

        self.fc1 = nn.Sequential(
            nn.Linear(self.z+self.c+1, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256*self.vsize, bias=False),
            nn.BatchNorm1d(256*self.vsize),
            nn.ReLU(inplace=True))

        self.layer1 = ConvT2d(256, 128, 4, stride=2, padding=1, batch_norm=True,
                              activation=nn.ReLU(inplace=True))
        self.layer2 = ConvT2d(128, 64, 4, stride=2, padding=1, batch_norm=True,
                              activation=nn.ReLU(inplace=True))
        self.layer3 = ConvT2d(64, 32, 4, stride=2, padding=1, batch_norm=True,
                              activation=nn.ReLU(inplace=True))
        self.layer4 = ConvT2d(32, 16, 4, stride=2, padding=1, batch_norm=True,
                              activation=nn.ReLU(inplace=True))
        self.layer5 = ConvT2d(16, 8, 4, stride=2, padding=1, batch_norm=True,
                              activation=nn.ReLU(inplace=True))
        self.layer6 = ConvT2d(8, 8, 1, batch_norm=True, activation=nn.ReLU(inplace=True))
        self.layer7 = ConvT2d(8, 8, 1, batch_norm=True, activation=nn.ReLU(inplace=True))
        self.layer8 = ConvT2d(8, 8, 1, batch_norm=True, activation=nn.ReLU(inplace=True))
        self.layer9 = ConvT2d(8, self.out_dim, 1, batch_norm=False, activation=nn.Tanh())

        self.layer6_79 = ConvT2d(16, 8, 1, batch_norm=True, activation=nn.ReLU(inplace=True))
        self.layer9_79 = ConvT2d(16, self.out_dim, 1, batch_norm=False, activation=nn.Tanh())
    #self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, FG.std)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, FG.std)
                m.bias.data.zero_()

    def forward(self, z, c, axis):
        x = torch.cat([z, c], 1)
        x = self.fc1(x)
        x = self.fc2(x)
        if axis == 0:
            x = x.view(-1,256, 6, 5)
        elif axis == 1:
            x = x.view(-1,256, 5, 5)
        elif axis == 2:
            x = x.view(-1,256, 5, 6)
        x = self.layer1(x)
        # print('G1', x.shape)
        x = self.layer2(x)
        # print('G2', x.shape)
        x = self.layer3(x)
        # print('G3', x.shape)
        x = self.layer4(x)
        # print('G4', x.shape)
        # x = self.layer5(x)
        # x = self.layer6(x)
        # x = self.layer6_79(x)
        # print('G6', x.shape)
        # x = self.layer7(x)
        # x = self.layer8(x)
        # x = self.layer9(x)
        # print('G9', x.shape)
        x = self.layer9_79(x)
        return x

class mE(NetworkBase):
    def __init__(self, ckpt_dir, num_outputs=2):
        super(mE, self).__init__(ckpt_dir)
        self.conv1_1 = Conv2d(1, 32, 3, 2, 1)
        self.conv1_2 = Conv2d(32, 32, 3, 1, 1)
        self.mp1 = nn.MaxPool2d(2, 2)

        self.block2_1 = BBlock(32, 32, 64)
        self.block2_2 = BBlock(64, 32, 64)

        self.block3_1 = BBlock(64, 32, 128)
        self.block3_2 = BBlock(128, 32, 128)

        self.block4 = BBlock(128, 64, 128)
        self.block5 = BBlock(128, 64, 128)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print(x.shape, type(x))
        x = self.conv1_1(x)
        # print(x.shape, type(x))
        x = self.conv1_2(x)
        x = self.mp1(x)

        x = self.block2_1(x)
        x = self.block2_2(x)

        x = self.block3_1(x)
        x = self.block3_2(x)

        x = self.block4(x)
        x = self.block5(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class mDiscriminator(nn.Module):
    """ Discriminator net D(x, yc) """
    def __init__(self, FG, c_code):
        super(mDiscriminator, self).__init__()
        self.FG = FG
        self.c_code = c_code
        self.output_dim = 1

        self.inference_x = nn.Sequential(
            Conv2d(1, 8, 4, stride=2, batch_norm=False,
                   activation=nn.LeakyReLU(0.2, inplace=False)),
            nn.Dropout2d(p=FG.dropout),
            Conv2d(8, 16, 4, stride=2, batch_norm=True,
                   activation=nn.LeakyReLU(0.2, inplace=False)),
            nn.Dropout2d(p=FG.dropout),
            Conv2d(16, 32, 4, stride=2, batch_norm=True,
                   activation=nn.LeakyReLU(0.2, inplace=False)),
            nn.Dropout2d(p=FG.dropout),
            Conv2d(32, 64, 4, stride=2, batch_norm=True,
                   activation=nn.LeakyReLU(0.2, inplace=False)),
            nn.Dropout2d(p=FG.dropout),
            Conv2d(64, 128, 4, stride=2, padding=1, batch_norm=True,
                   activation=nn.LeakyReLU(0.2, inplace=False)),
            nn.Dropout2d(p=FG.dropout))

        self.fc = nn.Sequential(
            Conv2d(128, 256, 1, stride=2, batch_norm=False, activation=nn.LeakyReLU(0.2, inplace=False)),
            nn.Dropout2d(p=FG.dropout),
            Conv2d(256, 256, 1, batch_norm=False, activation=nn.LeakyReLU(0.2, inplace=False)),
            nn.Dropout2d(p=FG.dropout),
            nn.Conv2d(256, self.output_dim+self.c_code, 1, bias=True))

    #self.reset_parameters()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, self.FG.std)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, self.FG.std)
                m.bias.data.zero_()

    def forward(self, x):
        print('Di:', x.shape)
        output_x = self.inference_x(x)
        print('Dc:', output_x.shape)
        output = self.fc(output_x)
        print('Df:', output.shape)
        exit()
        output = output.squeeze()
        if output.dim() == 1:
            output = output.unsqueeze(dim=0)
        ac = torch.sigmoid(output[:, self.output_dim])
        cont = output[:, self.output_dim:]
        # print(ac.shape, cont.shape)
        return ac, cont


""""""""""""""""""""" Generator layer """""""""""""""""""""
""" 95x79 image """
# self.layer1 = nn.Sequential(
#   # input dim: z_dim x 1 x 1
#   nn.ConvTranspose2d(self.FG.z_dim, 256, 3, stride=2, padding=0, bias=False),
#   nn.BatchNorm2d(256),
#   nn.ReLU(FG.slope, inplace=True))
# self.layer1 = nn.Sequential(
#   nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, output_padding=1, bias=False),
#   nn.BatchNorm2d(128),
#   nn.ReLU(inplace=True))
# self.layer2 = nn.Sequential(
#   nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1,bias=False),
#   nn.BatchNorm2d(64),
#   nn.ReLU(inplace=True))
# self.layer3 = nn.Sequential(
#   nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=1, bias=False),
#   nn.BatchNorm2d(32),
#   nn.ReLU(inplace=True))
# self.layer4 = nn.Sequential(
#   nn.ConvTranspose2d(32, 16, 4, stride=2, output_padding=1,  bias=False),
#   nn.BatchNorm2d(16),
#   nn.ReLU(inplace=True))
# self.layer5 = nn.Sequential(
#   nn.ConvTranspose2d(16, 8, 4, stride=2, bias=False),
#   nn.BatchNorm2d(8),
#   nn.ReLU(inplace=True))
# self.layer6 = nn.Sequential(
#   nn.ConvTranspose2d(8, 8, 1, stride=1,bias=True),
#   nn.BatchNorm2d(8),
#   nn.ReLU(inplace=True))
# self.layer7 = nn.Sequential(
#   nn.ConvTranspose2d(8, self.out_dim, 1, stride=1,bias=True),
#   nn.Tanh())
""" 188x156 image """
# self.layer1 = nn.Sequential(
#   nn.ConvTranspose2d(256, 128, 4, stride=2, padding=0, bias=False),
#   nn.BatchNorm2d(128),
#   nn.ReLU(inplace=True))
# self.layer2 = nn.Sequential(
#   nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1,bias=False),
#   nn.BatchNorm2d(64),
#   nn.ReLU(inplace=True))
# self.layer3 = nn.Sequential(
#   nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
#   nn.BatchNorm2d(32),
#   nn.ReLU(inplace=True))
# self.layer4 = nn.Sequential(
#   nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
#   nn.BatchNorm2d(16),
#   nn.ReLU(inplace=True))
# self.layer5 = nn.Sequential(
#   nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, bias=False),
#   nn.BatchNorm2d(8),
#   nn.ReLU(inplace=True))
# self.layer6 = nn.Sequential(
#   nn.ConvTranspose2d(8, 8, 1, stride=1, padding=1, bias=True),
#   nn.BatchNorm2d(8),
#   nn.ReLU(inplace=True))
# self.layer7 = nn.Sequential(
#   nn.ConvTranspose2d(8, self.out_dim, 1, stride=1, padding=1, bias=True),
#   nn.Tanh())
