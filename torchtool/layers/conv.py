from torch import nn


class ConvT3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, dilation=1,
                 batch_norm=True, activation=nn.ReLU(inplace=True)):
        super(ConvT3d, self).__init__()
        self.convT = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, padding,
            output_padding, groups, not batch_norm, dilation)
        self.bn = nn.BatchNorm3d(out_channels) if batch_norm else None
        self.activation = activation

    def forward(self, x):
        x = self.convT(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)

        return x


class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 batch_norm=True, activation=nn.ReLU(inplace=True)):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias=not batch_norm)
        self.bn = nn.BatchNorm3d(out_channels) if batch_norm else None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)

        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 batch_norm=True, activation=nn.ReLU(inplace=True)):
        super(Conv2d, self).__init__()
        bias = bias or not batch_norm
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)

        return x
