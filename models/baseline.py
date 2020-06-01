from __future__ import print_function, division, absolute_import

from torch import nn
from torch.nn import functional as F
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


class BBlock3D(nn.Module):
    def __init__(self, _in, _m, _out):
        super(BBlock3D, self).__init__()
        self.stride = _out // _in
        assert self.stride in (1, 2)

        self.sq = Conv3d(_in, _m, 1, 1, 0)
        self.conv = Conv3d(_m, _m, 3, self.stride, 1)
        self.ex = Conv3d(_m, _out, 1, 1, 0, activation=None)

    def forward(self, x):
        residual = x
        x = self.sq(x)
        x = self.conv(x)
        x = self.ex(x)

        if self.stride == 1:
            x += residual

        x = F.relu(x, inplace=True)
        return x


class Baseline(NetworkBase):
    def __init__(self, ckpt_dir, num_outputs=2):
        super(Baseline, self).__init__(ckpt_dir)
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
        print(x.shape, type(x))
        x = self.conv1_1(x)
        print(x.shape, type(x))
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


class Baseline3D(NetworkBase):
    def __init__(self, ckpt_dir, num_outputs=2):
        super(Baseline3D, self).__init__(ckpt_dir)
        self.conv1_1 = Conv3d(1, 32, 3, 2, 1)
        self.conv1_2 = Conv3d(32, 32, 3, 1, 1)
        self.mp1 = nn.MaxPool3d(2, 2)

        self.block2_1 = BBlock3D(32, 32, 64)
        self.block2_2 = BBlock3D(64, 32, 64)

        self.block3_1 = BBlock3D(64, 32, 128)
        self.block3_2 = BBlock3D(128, 32, 128)

        self.block4 = BBlock3D(128, 64, 128)
        self.block5 = BBlock3D(128, 64, 128)

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print(x.shape, type(x))
        # exit()
        x = self.conv1_1(x)
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
