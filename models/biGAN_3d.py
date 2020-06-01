from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn


class ConvT3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, dilation=1,
                 batch_norm=True, activation=None):
        super(ConvT3d, self).__init__()
        self.convT = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                              stride, padding, output_padding, groups, not batch_norm, dilation)
        self.batch_norm = nn.BatchNorm3d(out_channels) if batch_norm else None

        self.activation = activation

    def forward(self, x):
        x = self.convT(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 batch_norm=True, activation=None):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias=not batch_norm)
        self.batch_norm = nn.BatchNorm3d(out_channels) if batch_norm else None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ResNetBlockT(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, output_padding=1):
        super(ResNetBlockT, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.upsample = None

        self.convT1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 4, 1, 0),
            nn.ReLU(True))
        self.convT2 = nn.ConvTranspose3d(in_channels, out_channels, 4, 1, padding,
                              output_padding=output_padding)
        # if stride != 1 or in_channels != out_channels:
        #     self.upsample = nn.ConvTranspose3d(in_channels, out_channels, 1, stride, 1,
        #                           output_padding=output_padding)
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, 1, stride, padding,
                              output_padding=output_padding)
    def forward(self, x):
        residual = x
        out = self.convT1(x)
        out = self.convT2(out)
        # if self.upsample is not None:
        #     residual = self.upsample(x)
        if residual.size(2) != out.size(2):
            residual = self.upsample(x)
        out += residual
        out = self.activation(out)
        return out


class Generator_3D(nn.Module):
    """ Generator net (decoder, fake) P(x|z) """
    def __init__(self, FG):
        super(Generator_3D, self).__init__()
        self.FG = FG
        self.vsize = 6*5*5
        self.out_dim = 1

        self.fc1 = nn.Sequential(
            nn.Linear(self.FG.z_dim+self.FG.c_code, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128*self.vsize, bias=False),
            nn.BatchNorm1d(128*self.vsize),
            nn.ReLU(inplace=True))

        self.layer1 = ConvT3d(128, 128, 4, stride=2, padding=1, batch_norm=True,
                              activation=nn.ReLU(inplace=True))
        self.layer2 = ConvT3d(128, 64, 4, stride=2, padding=1, batch_norm=True,
                              activation=nn.ReLU(inplace=True))
        self.layer3 = ConvT3d(64, 32, 4, stride=2, padding=1, batch_norm=True,
                              activation=nn.ReLU(inplace=True))
        self.layer4 = ConvT3d(32, 16, 4, stride=2, padding=1, batch_norm=True,
                              activation=nn.ReLU(inplace=True))
        self.layer5 = ConvT3d(16, 8, 4, stride=2, padding=1, batch_norm=True,
                              activation=nn.ReLU(inplace=True))
        self.layer6 = ConvT3d(8, 8, 1, batch_norm=True, activation=nn.ReLU(inplace=True))
        self.layer6_79 = ConvT3d(16, 8, 1, batch_norm=True, activation=nn.ReLU(inplace=True))
        self.layer7 = ConvT3d(8, 8, 1, batch_norm=True, activation=nn.ReLU(inplace=True))
        self.layer8 = ConvT3d(8, 8, 1, batch_norm=True, activation=nn.ReLU(inplace=True))
        self.layer9 = ConvT3d(8, self.out_dim, 1, batch_norm=False, activation=nn.Tanh())
        self.layer9_79 = ConvT3d(16, self.out_dim, 1, batch_norm=False, activation=nn.Tanh())
    #self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                m.weight.data.normal_(0.0, self.FG.std)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.normal_(1.0, self.FG.std)
                m.bias.data.zero_()

    def forward(self, z, cont_code):
        x = torch.cat([z, cont_code], 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1,128,5,6,5)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)
        # x = self.layer6(x)
        #x = self.layer7(x)
        #x = self.layer8(x)
        x = self.layer6_79(x)
        x = self.layer9(x)

        # x = self.layer9_79(x)
        return x


class E_3D(nn.Module):
    """ Inference net (encoder, real) E(z|x) """
    def __init__(self, FG):
        super(E_3D, self).__init__()
        self.FG = FG
        self.in_dim = 1

        self.layer1 = Conv3d(1, 8, 3, stride=2, batch_norm=True,
                             activation=nn.LeakyReLU(FG.slope, inplace=False))
        self.layer2 =  Conv3d(8, 16, 3, stride=2, batch_norm=True,
                             activation=nn.LeakyReLU(FG.slope, inplace=False))
        self.layer3 =  Conv3d(16, 32, 3, stride=2, batch_norm=True,
                             activation=nn.LeakyReLU(FG.slope, inplace=False))
        self.layer4 =  Conv3d(32, 64, 3, stride=2, batch_norm=True,
                             activation=nn.LeakyReLU(FG.slope, inplace=False))
        self.layer5 =  Conv3d(64, 128, 3, stride=2, batch_norm=True,
                             activation=nn.LeakyReLU(FG.slope, inplace=False))
        self.layer6 =  Conv3d(128, 128, 1, batch_norm=True,
                             activation=nn.LeakyReLU(FG.slope, inplace=False))
        self.layer6_79 = Conv3d(64, 128, 1, batch_norm=True,
                             activation=nn.LeakyReLU(FG.slope, inplace=False))
        self.layer7 = Conv3d(128, 128, 1, batch_norm=True,
                             activation=nn.LeakyReLU(FG.slope, inplace=False))
        self.layer8 = Conv3d(128, 128, 1, batch_norm=True,
                             activation=nn.LeakyReLU(FG.slope, inplace=False))
        self.layer9 = Conv3d(128, 256, 1, batch_norm=False, activation=None)

        self.fc1 = nn.Sequential(
            nn.Linear(256*4*5*4, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(FG.slope, inplace=True))
        self.fc2 = nn.Linear(256, self.FG.z_dim+self.FG.c_code, bias=False)
        #self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0.0, self.FG.std)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.normal_(1.0, self.FG.std)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)
        # x = self.layer6(x)
        #x = self.layer7(x)
        #x = self.layer8(x)
        x = self.layer6_79(x)
        x = self.layer9(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x_E = x[:, :self.FG.z_dim]
        c_E = x[:, self.FG.z_dim:]
        return x_E, c_E


class Discriminator_3D(nn.Module):
    """ Discriminator net D(x, z) """
    def __init__(self, FG):
        super(Discriminator_3D, self).__init__()
        self.FG = FG

        self.inference_x = nn.Sequential(
            Conv3d(1, 8, 5, stride=2, batch_norm=False,
                   activation=nn.LeakyReLU(FG.std, inplace=False)),
            nn.Dropout3d(p=FG.dropout))#,
        self.inference_x2 = nn.Sequential(
            Conv3d(8, 16, 4, stride=2, batch_norm=True,
                   activation=nn.LeakyReLU(FG.std, inplace=False)),
            nn.Dropout3d(p=FG.dropout))#,
        self.inference_x3 = nn.Sequential(
            Conv3d(16, 32, 4, stride=2, batch_norm=True,
                   activation=nn.LeakyReLU(FG.std, inplace=False)),
            nn.Dropout3d(p=FG.dropout))#,
        self.inference_x4 = nn.Sequential(
            Conv3d(32, 64, 4, stride=2, batch_norm=True,
                   activation=nn.LeakyReLU(FG.std, inplace=False)),
            nn.Dropout3d(p=FG.dropout))#,
        self.inference_x5 = nn.Sequential(
            Conv3d(64, 128, 4, stride=2, padding=1, batch_norm=True,
                   activation=nn.LeakyReLU(FG.std, inplace=False)),
            nn.Dropout3d(p=FG.dropout))#,
        self.inference_x6 = nn.Sequential(
            Conv3d(128, 128, 4, stride=2, batch_norm=True,
                   activation=nn.LeakyReLU(FG.std, inplace=False)),
            nn.Dropout3d(p=FG.dropout))
        self.inference_x6_79 = nn.Sequential(
            Conv3d(128, 128, 1, stride=2, batch_norm=True,
                   activation=nn.LeakyReLU(FG.std, inplace=False)),
            nn.Dropout3d(p=FG.dropout))

        self.inference_z = nn.Sequential(
            Conv3d(self.FG.z_dim, 128, 1, batch_norm=False, activation=nn.LeakyReLU(FG.std, inplace=False)),
            nn.Dropout3d(p=FG.dropout),
            Conv3d(128, 128, 1, batch_norm=False, activation=nn.LeakyReLU(FG.std, inplace=False)),
            nn.Dropout3d(p=FG.dropout))

        self.inference_joint = nn.Sequential(
            Conv3d(256, 256, 1, batch_norm=False, activation=nn.LeakyReLU(FG.std, inplace=False)),
            nn.Dropout3d(p=FG.dropout),
            Conv3d(256, 256, 1, batch_norm=False, activation=nn.LeakyReLU(FG.std, inplace=False)),
            nn.Dropout3d(p=FG.dropout),
            nn.Conv3d(256, 1, 1, stride=1, bias=True))
        if not FG.wasserstein:
            self.inference_joint.add_module('sigmoid', nn.Sigmoid())

    #self.reset_parameters()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0.0, self.FG.std)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.normal_(1.0, self.FG.std)
                m.bias.data.zero_()

    def forward(self, x, z):
        output_x = self.inference_x(x)
        output_x = self.inference_x2(output_x)
        output_x = self.inference_x3(output_x)
        output_x = self.inference_x4(output_x)
        output_x = self.inference_x5(output_x)
        output_x = self.inference_x6_79(output_x)
        #output_x = self.inference_x6(output_x)
        # print(output_x.shape)
        # exit()
        z = z.reshape(z.size(0), z.size(1), 1, 1, 1) #[b, 128, 1, 1]
        output_z = self.inference_z(z) #[b, 256, 1, 1]
        output = self.inference_joint(torch.cat((output_x, output_z), 1))
        return output


"""    Generator    """
# self.layer1 = ConvT3d(128,128,4,stride=2, padding=1, batch_norm=True,
#                       activation=nn.ReLU(inplace=True))
# self.layer2 = ConvT3d(128,64,4,stride=2, padding=1, batch_norm=True,
#                       activation=nn.ReLU(inplace=True))
# self.layer3 = ConvT3d(64,32,4,stride=2, padding=1, batch_norm=True,
#                       activation=nn.ReLU(inplace=True))
# self.layer4 = ConvT3d(32,16,4,stride=2, padding=1, batch_norm=True,
#                       activation=nn.ReLU(inplace=True))
# self.layer5 = ConvT3d(16,8,4,stride=2, padding=1, batch_norm=True,
#                       activation=nn.ReLU(inplace=True))
# self.layer6 = ConvT3d(8,8,1,stride=1, padding=0, batch_norm=True,
#                       activation=nn.ReLU(inplace=True))
# self.layer7 = ConvT3d(8,8,1,stride=1, padding=0, batch_norm=True,
#                       activation=nn.ReLU(inplace=True))
# self.layer8 = ConvT3d(8,8,1,stride=1, padding=0, batch_norm=True,
#                       activation=nn.ReLU(inplace=True))
# self.layer9 = ConvT3d(8, self.out_dim, 1, batch_norm=False, activation=nn.Tanh())

"""    E    """
# self.layer1 = Conv3d(1, 8, 3, stride=2, batch_norm=True,
#                      activation=nn.LeakyReLU(FG.slope, inplace=False))
# self.layer2 =  Conv3d(8, 16, 3, stride=2, batch_norm=True,
#                      activation=nn.LeakyReLU(FG.slope, inplace=False))
# self.layer3 =  Conv3d(16, 32, 3, stride=2, batch_norm=True,
#                      activation=nn.LeakyReLU(FG.slope, inplace=False))
# self.layer4 =  Conv3d(32, 64, 3, stride=2, batch_norm=True,
#                      activation=nn.LeakyReLU(FG.slope, inplace=False))
# self.layer5 =  Conv3d(64, 128, 3, stride=2, batch_norm=True,
#                      activation=nn.LeakyReLU(FG.slope, inplace=False))
# self.layer6 =  Conv3d(128, 128, 1, batch_norm=True,
#                      activation=nn.LeakyReLU(FG.slope, inplace=False))
# self.layer7 = Conv3d(128, 128, 1, batch_norm=True,
#                      activation=nn.LeakyReLU(FG.slope, inplace=False))
# self.layer8 = Conv3d(128, 128, 1, batch_norm=True,
#                      activation=nn.LeakyReLU(FG.slope, inplace=False))
# self.layer9 = Conv3d(128, 128, 1, batch_norm=False, activation=None)

""" 95x79 image """
# self.layer1 = nn.Sequential(
#   # input dim: z_dim x 1 x 1
#   nn.ConvTranspose3d(self.FG.z_dim, 256, 3, stride=2, padding=0, bias=False),
#   nn.BatchNorm3d(256),
#   nn.ReLU(FG.slope, inplace=True))
# self.layer1 = nn.Sequential(
#   nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1, output_padding=1, bias=False),
#   nn.BatchNorm3d(128),
#   nn.ReLU(inplace=True))
# self.layer2 = nn.Sequential(
#   nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1,bias=False),
#   nn.BatchNorm3d(64),
#   nn.ReLU(inplace=True))
# self.layer3 = nn.Sequential(
#   nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1, output_padding=1, bias=False),
#   nn.BatchNorm3d(32),
#   nn.ReLU(inplace=True))
# self.layer4 = nn.Sequential(
#   nn.ConvTranspose3d(32, 16, 4, stride=2, output_padding=1,  bias=False),
#   nn.BatchNorm3d(16),
#   nn.ReLU(inplace=True))
# self.layer5 = nn.Sequential(
#   nn.ConvTranspose3d(16, 8, 4, stride=2, bias=False),
#   nn.BatchNorm3d(8),
#   nn.ReLU(inplace=True))
# self.layer6 = nn.Sequential(
#   nn.ConvTranspose3d(8, 8, 1, stride=1,bias=True),
#   nn.BatchNorm3d(8),
#   nn.ReLU(inplace=True))
# self.layer7 = nn.Sequential(
#   nn.ConvTranspose3d(8, self.out_dim, 1, stride=1,bias=True),
#   nn.Tanh())
""" 188x156 image """
# self.layer1 = nn.Sequential(
#   nn.ConvTranspose3d(256, 128, 4, stride=2, padding=0, bias=False),
#   nn.BatchNorm3d(128),
#   nn.ReLU(inplace=True))
# self.layer2 = nn.Sequential(
#   nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1,bias=False),
#   nn.BatchNorm3d(64),
#   nn.ReLU(inplace=True))
# self.layer3 = nn.Sequential(
#   nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1, bias=False),
#   nn.BatchNorm3d(32),
#   nn.ReLU(inplace=True))
# self.layer4 = nn.Sequential(
#   nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1, bias=False),
#   nn.BatchNorm3d(16),
#   nn.ReLU(inplace=True))
# self.layer5 = nn.Sequential(
#   nn.ConvTranspose3d(16, 8, 4, stride=2, padding=1, bias=False),
#   nn.BatchNorm3d(8),
#   nn.ReLU(inplace=True))
# self.layer6 = nn.Sequential(
#   nn.ConvTranspose3d(8, 8, 1, stride=1, padding=1, bias=True),
#   nn.BatchNorm3d(8),
#   nn.ReLU(inplace=True))
# self.layer7 = nn.Sequential(
#   nn.ConvTranspose3d(8, self.out_dim, 1, stride=1, padding=1, bias=True),
#   nn.Tanh())
