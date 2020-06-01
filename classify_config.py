from __future__ import print_function, division, absolute_import
import argparse
import os
import pickle


class Flags(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(Flags, self).__init__(*args, **kwargs)

    def set_arguments(self):
        self.add_argument('model', type=str)
        self.add_argument(
            '--devices', nargs='+', type=int, default=[0, 1])

        self.add_argument('--ckpt_root', type=str, default='checkpoint')
        self.add_argument('--ckpt_dir', type=str, default=None)

        self.add_argument('--fold', type=int, default=5)
        self.add_argument('--cur_fold', type=int, default=0)

        self.add_argument('--vis_port', type=int, default=10002)
        self.add_argument('--vis_env', type=str, default='test')
        self.add_argument('--memo', type=str, default='')

        self.add_argument('--axis', type=int, default=1)
        self.add_argument('--z_dim', type=int, default=128)
        self.add_argument('--c_code', type=int, default=4)
        self.add_argument('--modality', type=str, default='MRI_0')
        self.add_argument(
            '--labels', nargs='+', type=str, default=['AD', 'CN'])

        self.add_argument('--num_epoch', type=int, default=350)
        self.add_argument('--batch_size', type=int, default=40)

        self.add_argument('--l2_decay', type=float, default=0.0005)
        self.add_argument('--l1_decay', type=float, default=None)

        self.add_argument('--lr', type=float, default=0.001)
        self.add_argument('--lr_gamma', type=float, default=0.985)

        self.add_argument('--save_term', type=int, default=10)
        self.add_argument('--nmid_layers', type=int, default=2)
        self.add_argument('--last_layer', default='fconv_gap',
                          choices=['fconv_gap', 'gap_linear'])

        self.args = None

    def parse_args(self, *args, **kwargs):
        args = super(Flags, self).parse_args(*args, **kwargs)
        args.labels = tuple(args.labels)
        args.ckpt_dir = os.path.join(
            args.ckpt_root, args.model, 'f'+str(args.cur_fold))

        self.args = args
        return args

    def report(self, end='\n'):
        assert self.args is not None, 'Argument not parsed'
        d = self.parse_args().__dict__
        report = '{:15}      {}'.format('cur_fold', d['cur_fold'])+end
        report += '{:15}      {}'.format('memo', d['memo'])+end
        keys = d.keys()

        for k in sorted(keys):
            report += '{:15}      {}'.format(k, d[k])+end
        return report

    def save(self):
        assert self.args is not None, 'Argument not parsed'

        file_path = os.path.join(self.args.ckpt_dir, 'config.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(self.args, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        assert self.args is not None, 'Argument not parsed'
        self.args.ckpt_dir = os.path.join(
            self.args.ckpt_root, self.args.model, 'f'+str(self.args.cur_fold))

        file_path = os.path.join(self.args.ckpt_dir, 'config.pkl')
        with open(file_path, 'rb') as f:
            args = pickle.load(f)

        self.args = args
        return args

    def configure(self, key, *args):
        if key == 'ckpt_dir':
            self.args.ckpt_dir = os.path.join(
                self.args.ckpt_root, self.args.model, 'f'+str(self.args.cur_fold))
        elif key == 'cur_fold':
            if len(args) == 1:
                self.args.__dict__[key] = args[0]
            else:
                self.args.__dict__[key] = args
