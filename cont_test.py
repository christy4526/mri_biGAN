import os

import torch
from torch.nn import functional as F
from torchvision.transforms import Compose, Lambda

from models import Baseline, Generator
#from config import biGAN_parser
from classify_config import Flags
from hs_transforms import FiveCrop, ToTensor

import time
from visdom import Visdom
from summary import Image3D

import itertools
import numpy as np

if __name__ == '__main__':
    parser = Flags()
    parser.set_arguments()
    FG = parser.parse_args()
    c_code, axis, z_dim = FG.c_code, FG.axis, FG.z_dim
    device = torch.device(FG.devices[0])
    torch.cuda.set_device(FG.devices[0])

    nets = []
    for i in range(FG.fold):
        parser.configure('cur_fold', i)
        parser.configure('ckpt_dir')
        FG = parser.load()
        net = Baseline(FG.ckpt_dir, len(FG.labels))
        net.to(device)
        net.load(epoch=None, optimizer=None, is_best=True)
        net.eval()
        nets += [net]

    #G = Generator(FG)
    G = torch.nn.DataParallel(Generator(z_dim, c_code, axis))

    # state_dict = torch.load(os.path.join('BiGAN-info-c4-f', 'G.pth'), 'cpu')
    state_dict = torch.load(os.path.join('157-G8', 'G.pth'), 'cpu')
    G.load_state_dict(state_dict)
    G.to(device)
    G.eval()

    if axis == 1:
        ns = ((160, 160), (112, 112))
    elif axis == 0:
        ns = ((192, 160), (144, 112))
    t = Compose([
            FiveCrop((160, 160), (112, 112)),
            Lambda(lambda patches: torch.stack(
                [patch for patch in patches]))])

    uniform = torch.distributions.Uniform(-1, 1)
    vis = Visdom(port=10002, env='1-256')

    def process_batch(batch, device=None, non_blocking=False):
        x, y = batch
        return (x.to(device, non_blocking=non_blocking),
                y.to(device, non_blocking=non_blocking))

    def get_confidence(net, x):
        output = net(x)
        # pricess five crop outputs into one
        # mean over five crops
        output = output.view(1, 5, *output.shape[1:])
        output = output.mean(dim=1)
        output = F.softmax(output, dim=1)
        return F.softmax(output, dim=1)

    def get_ensambled_y_preds(outputs):
        ensambled_ypred = torch.zeros(1).to(device)

        for output in outputs:
            # print(torch.argmax(output, dim=1).float(), output)
            ensambled_ypred += torch.argmax(output, dim=1).float()

        # ones = torch.ones(1)
        # zeros = torch.zeros(1)
        # ypred=0

        if ensambled_ypred >= 3:
            ypred = torch.ones(1).to(device)

        else:
            ypred = torch.zeros(1).to(device)
        return ypred

    result = []
    torch.set_grad_enabled(False)
    z = torch.rand(1, z_dim)
    count=0
    for cc in itertools.product(torch.linspace(1, -1, 10), repeat=c_code):
        c = torch.stack(cc).float()
        z = z.to(device)
        c = c.unsqueeze(dim=0).to(device)

        im = G(z, c, axis)
        im = im.detach().cpu()
        im = im.squeeze(dim=1)
        x = t(im)
        batch_size = x.size(0)
        npatches = x.size(1)
        x = x.to(device)
        #print(x.shape)

        # classification
        outputs = list(map(lambda net: get_confidence(net, x), nets))
        y_pred = get_ensambled_y_preds(outputs)

        # confidence = ...
        confidence = torch.zeros(1).cuda(device, non_blocking=True)
        temp=torch.zeros(5,2).cuda(device, non_blocking=True)
        for j in range(len(nets)):
            temp[j] += outputs[j][0]
        if y_pred.item() == 0:
            confidence = temp[:,0].max()
        else:
            confidence = 1-temp[:,1].max()
        #print(confidence.item())
        print(y_pred.item())

        c=c[0]
        #print((confidence.item(),)+c[0])
        result.append((confidence.item(),)+cc)

        title = 'c'+str(count)+'>' + str(c.squeeze().tolist())
        im = im*0.5+0.5
        ip = Image3D(vis, title)
        # print(im.shape)
        ip(title, im[:,:,:])
        # vis.image(im.squeeze(), win='si', opts=dict(title=title))

        # time.sleep(0.3)
        # input()
        vis.save([vis.env])

    np.savetxt('Result-256-1.txt', result, delimiter=',')
