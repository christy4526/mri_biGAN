from __future__ import absolute_import, division, print_function
import argparse
import os

def argument_report(arg, end='\n'):
    d = arg.__dict__
    keys = d.keys()
    report = '{:15}    {}'.format('running_fold', d['running_fold'])+end
    report += '{:15}    {}'.format('memo', d['memo'])+end
    for k in sorted(keys):
        if k == 'running_fold' or k == 'memo':
            continue
        report += '{:15}    {}'.format(k, d[k])+end
    return report


def _base_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--checkpoint_root', type=str, default='checkpoint')
    parser.add_argument('--devices', type=int, nargs='+', default=(0,1,2,3))
    parser.add_argument('--labels', type=str, nargs='+', default=('AD','CN'))
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--running_fold', type=int, default=0)
    parser.add_argument('--memo', type=str, default='')
    parser.add_argument('--vis_env', type=str, default='infoGAN')
    parser.add_argument('--vis_port', type=int, default=10002)
    parser.add_argument('--subject_ids_path', type=str,
                        default=os.path.join('data', 'subject_ids.pkl'))
    parser.add_argument('--diagnosis_path', type=str,
                        default=os.path.join('data', 'diagnosis.pkl'))
    # parser.add_argument('--load_pkl', type=str, default ='false')
    # parser.add_argument('--gm', type=str, default ='false')
    parser.add_argument('--gm', action='store_true')
    ## usage python filename.py --gm
    ## > print(FG.gm)
    ## # True

    return parser


def train_args():
    parser = argparse.ArgumentParser(parents=[_base_parser()])
    parser.add_argument('--num_epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--l2_decay', type=float, default=0.0005)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--lr_gamma', type=float, default=0.999)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--z', type=int, default=128)
    parser.add_argument('--d_code', type=int, nargs='+', default=(40,42,44))
    parser.add_argument('--c_code', type=int, default=2)
    parser.add_argument('--axis', type=int, default=0)
    parser.add_argument('--isize', type=int, default=79)
    parser.add_argument('--SUPERVISED', type=str, default='True')
    parser.add_argument('--lr_adam', type=float, default=1e-4)
    parser.add_argument('--std', type=float, default=0.02, help='for weight')

    args = parser.parse_args()
    return args

def GAN_parser():
    parser = argparse.ArgumentParser(parents=[_base_parser()])
    #parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=20)
    # logging
    parser.add_argument('--clean_ckpt', type=bool, default=False)
    parser.add_argument('--load_kpt', type=bool, default=False)
    parser.add_argument('--D3', type=bool, default=False)
    parser.add_argument('--print_every', type=int, default=50)
    # hyperparameters
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--lr_adam', type=float, default=1e-4)
    parser.add_argument('--lrD', type=float, default=2e-4)
    parser.add_argument('--lrG', type=float, default=2e-4)
    parser.add_argument('--lrE', type=float, default=0.0001)
    parser.add_argument('--lr_rmsprop', type=float, default=5e-5)
    parser.add_argument('--beta1', type=float, default=0.5, help='for adam')
    parser.add_argument('--slope', type=float, default=1e-2, help='for leaky ReLU')
    parser.add_argument('--std', type=float, default=0.02, help='for weight')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--clamp', type=float, default=1e-2)
    parser.add_argument('--wasserstein', type=bool, default=False)

    parser.add_argument('--lr_gamma', type=float, default=0.999)
    parser.add_argument('--d_code', type=int, nargs='+', default=(42))
    parser.add_argument('--c_code', type=int, default=3)
    parser.add_argument('--axis', type=int, default=0)
    parser.add_argument('--isize', type=int, default=79)


    args = parser.parse_args()
    return args
