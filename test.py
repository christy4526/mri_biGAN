import os
from models import Baseline
from adni_dataset import transform_presets
import torch
from classify_config import Flags
from torch.nn import functional as F
from torchvision.transforms import Compose, Lambda
from hs_transforms import RandomCrop, FiveCrop, ToTensor, Normalize
from biGAN import Generator
from summary import Image3D
from visdom import Visdom
import numpy as np


if __name__ == '__main__':
    parser = Flags()
    parser.set_arguments()
    FG = parser.parse_args()
    torch.cuda.set_device(FG.devices[0])
    device = torch.device(FG.devices[0])

    nets = []
    c_code = 4
    axis = 1
    z_dim = 128
    G = Generator(z_dim, c_code, axis).to('cuda:{}'.format(FG.devices[0]))
    if len(FG.devices) != 1:
        G = torch.nn.DataParallel(G, FG.devices)
    if FG.axis == 1:
        G.load_state_dict(torch.load(os.path.join('BiGAN-info-c4-f', 'G.pth')))
    else:
        G.load_state_dict(torch.load(os.path.join('BiGAN-info-c4-u', 'G.pth')))

    for i in range(FG.fold):
        parser.configure('cur_fold', i)
        parser.configure('ckpt_dir')
        FG = parser.load()
        net = Baseline(FG.ckpt_dir, len(FG.labels))
        net.to(device)
        net.load(epoch=None, optimizer=None, is_best=True)
        net.eval()
        nets += [net]

    if FG.axis == 1:
        ns = ((160, 160), (112, 112))
    elif FG.axis == 0:
        ns = ((192, 160), (144, 112))

    transform = transform_presets(FG.axis, mode='mri_five_crop')

    # manual test with manual images or a image
    # this is just a example
    # images = torch.stack(
        # [transform(torch.randn((1,)+ns[0])) for _ in range(batch_size)])
    # print('1',images.shape)
    # process five crop patches into batch
    # images = images.view(batch_size*images.size(1), *images.shape[2:])
    # print('2',images.shape)
    sample_num = 25
    batch_size = sample_num
    targets = torch.Tensor(torch.zeros(batch_size)).long()

    def process_batch(batch, device=None, non_blocking=False):
        x, y = batch
        return (x.to(device, non_blocking=non_blocking),
                y.to(device, non_blocking=non_blocking))

    # x, targets = process_batch((images, targets), device)
    # print('3',x.shape)

    def get_confidence(net, x):
        output = net(x)
        # pricess five crop outputs into one
        # mean over five crops
        output = output.view(batch_size, 5, *output.shape[1:])
        output = output.mean(dim=1)
        output = F.softmax(output, dim=1)
        return F.softmax(output, dim=1)

    # outputs = list(map(lambda net: get_confidence(net, x), nets))
    # print(outputs)

    def get_ensambled_y_preds(outputs):
        ensambled_ypred = torch.zeros_like(targets)

        for output in outputs:
            ensambled_ypred += torch.argmax(output, dim=1)

        ones = torch.ones_like(targets)
        zeros = torch.zeros_like(targets)
        ensambled_ypred = torch.where(ensambled_ypred >= 3,
                                      ones, zeros)
        return ensambled_ypred
    # y_pred = get_ensambled_y_preds(outputs)
    # print(y_pred)
    # exit()
    """#################### MINK ######################"""
    G.eval()
    FG = parser.parse_args()
    gi_transforms = Compose([
        FiveCrop(*ns),
        Lambda(lambda patches: torch.stack([patch
            for patch in patches]))])
    c_code = FG.c_code
    temp_c = torch.linspace(-1, 1, 5)
    C = torch.zeros((sample_num, FG.c_code)).cuda(device, non_blocking=True)

    num = len(temp_c)**FG.c_code#*10
    save_confi = torch.zeros(num,c_code+1)
    save_gi = torch.zeros(1)
    if FG.axis == 0:
        save_gi = torch.zeros(num,1,192,160)
    else:
        save_gi = torch.zeros(num,1,160,160)

    Z = torch.zeros(sample_num, FG.z_dim)
    z2 = torch.rand(1, FG.z_dim)
    for i in range(sample_num):
        #Z[i] = torch.rand(1, FG.z_dim)
        Z[i] = z2
    Z = Z.cuda(device, non_blocking=True)
    # for t1 in range(num):
    for t1 in range(len(temp_c)):
        C[:, 0] = temp_c[t1]
        for t2 in range(len(temp_c)):
            C[:, 1] = temp_c[t2]
            for t3 in range(5):
                for t in range(5):
                    C[t3*5+t, 2] = temp_c[t3]
            for s in range(sample_num):
                C[s, 3] = temp_c[s%5]

            vis = Visdom(port=FG.vis_port, env='Test-c4-'+str(FG.axis))
            gi_c = G(Z, C, FG.axis)

            for s in range(sample_num):
                name = 'target:'+str(C[s, 0].item())+':'+str(C[s, 1].item())
                # name = 'target:'+str(C[s, 0].item())+':'+str(C[s, 1].item())+\
                #     ':'+str(C[s, 2].item())+':'+str(C[s, 3].item())
                save_printer = Image3D(vis, name)
                save_printer(name, ((gi_c*0.5)+0.5)[s,:,:,:])

            gi = gi_c.squeeze() #.cuda(device, non_blocking=True)
            gi = gi_transforms(gi)
            gi = gi.view(gi.size(0)*gi.size(1), 1,*gi.shape[2:])

            x, targets = process_batch((gi, targets), device)
            outputs = list(map(lambda net: get_confidence(net, x), nets))
            y_pred = get_ensambled_y_preds(outputs)

            out_max = torch.zeros(sample_num, 1).cuda(device, non_blocking=True)
            for i in range(sample_num):
                temp=torch.zeros(5,2).cuda(device, non_blocking=True)
                for j in range(len(nets)):
                    temp[j] += outputs[j][i]
                    # print(outputs[j][i])
                if y_pred[i] == 0:
                    out_max[i] = temp[:,0].max()
                else:
                    out_max[i] = 1-temp[:,1].max()
                #print(out_max[i])
            # exit()
            for i in range(sample_num):
                for j in range(c_code):
                    save_confi[t1*125+t2*25+i][j] = C[i,j-1].item()
                    #save_confi[t1*25+i][j] = C[i,j].item()
                save_confi[t1*125+t2*25+i][c_code] = out_max[i].item()
                save_gi[t1*125+t2*25+i] = ((gi_c*0.5)+0.5)[i]
            # exit()

    confi = [0.9, 0.8, 0.7, 0.6, 0.5]
    # for k in range(sample_num*num):
    for k in range(625):
        # print(save_confi[k][0].item(),save_confi[k][1].item(),save_confi[k][2].item())
        print(save_confi[k][0].item(),save_confi[k][1].item(),save_confi[k][2].item(),
            save_confi[k][3].item(),save_confi[k][4].item())

        if save_confi[k][c_code] >= 0.5:
            for j in range(5):
                if save_confi[k][c_code].item() >= confi[j]:
                    vis = Visdom(port=10002, env=str(FG.axis)+'_'+str(FG.z_dim)+'-c4-result_AD>'+str(confi[j]))
                    sname = str(save_confi[k][c_code].item())+':'+str(save_confi[k][1].item())+','+\
                        str(save_confi[k][2].item())+','+str(save_confi[k][3].item())+\
                        ','+str(save_confi[k][4].item())
                    # sname = str(save_confi[k][c_code].item())+':'+str(save_confi[k][0].item())+','+\
                    #     str(save_confi[k][1].item())
                    save_image = Image3D(vis, 'AD_'+sname)
                    save_image(sname, save_gi[k])
                    break
                else:
                    continue
        else:
            for j in range(5):
                if 1-save_confi[k][c_code].item() >= confi[j]:
                    # sname = str(1-save_confi[k][c_code].item())+':'+str(save_confi[k][0].item())+','+\
                    #     str(save_confi[k][1].item())
                    vis = Visdom(port=10002, env=str(FG.axis)+'_'+str(FG.z_dim)+'-c4-result_NC>'+str(confi[j]))
                    sname = str(1-save_confi[k][c_code].item())+':'+str(save_confi[k][1].item())+','+\
                        str(save_confi[k][2].item())+','+str(save_confi[k][3].item())+\
                        ','+str(save_confi[k][4].item())
                    save_image = Image3D(vis, 'NC_'+sname)
                    save_image(sname, save_gi[k])
                    break
                else:
                    continue
