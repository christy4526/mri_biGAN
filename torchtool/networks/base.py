import torch
import numpy
import os
from collections import OrderedDict as odict


def calc_nparams(module):
    nparams = odict()

    for k in module._modules:
        for param in module._modules[k].parameters():
            try:
                nparams[k] += numpy.prod(param.shape)
            except KeyError:
                nparams[k] = numpy.prod(param.shape)

    nparams['total'] = sum([nparams[k] for k in nparams.keys()])
    return nparams


class NetworkBase(torch.nn.Module):
    def __init__(self, chkp_dir):
        super(NetworkBase, self).__init__()
        self.chkp_dir = chkp_dir
        self.best_performance = 0
        self.minimum_loss = 1e+199

    def _before_save(self, is_best=False):
        folders = self.chkp_dir.split(os.path.sep)

        accum = ''
        for folder in folders:
            if folder in ('', os.path.curdir):
                continue
            accum = os.path.join(accum, folder)
            if not os.path.exists(accum):
                os.mkdir(accum)

    def save(self, epoch, optimizer, performance, loss, is_best=False):
        self._before_save()

        trainig_state = dict(
            epoch=epoch,
            model=self.state_dict(),
            optimizer=optimizer.state_dict(),
            best_performance=self.best_performance,
            minimum_loss=self.minimum_loss,
            performance=performance,
            loss=loss)

        if is_best:
            fname = os.path.join(self.chkp_dir, 'best.pth.tar')
        else:
            fname = os.path.join(self.chkp_dir, str(epoch)+'.pth.tar')

        torch.save(trainig_state, fname)

    def save_if_best(self, epoch, optimizer, performance, loss,
                     mode='performance'):
        assert mode in ['performance', 'loss', 'both']
        self._before_save()

        if mode == 'performance':
            if self.best_performance <= performance:
                self.best_performance = performance
                self.minimum_loss = loss

                self.save(epoch, optimizer, performance, loss, is_best=True)

        elif mode == 'loss':
            if self.minimum_loss >= loss:
                self.best_performance = performance
                self.minimum_loss = loss

                self.save(epoch, optimizer, performance, loss, is_best=True)

        elif mode == 'both':
            if self.best_performance <= performance and \
                    self.minimum_loss + 1e-3 >= loss:
                self.best_performance = performance
                self.minimum_loss = loss

                self.save(epoch, optimizer, performance, loss, is_best=True)

    def load(self, epoch, optimizer=None, is_best=False):
        if not is_best:
            file = os.path.join(self.chkp_dir, str(epoch)+'.pth.tar')
        else:
            file = os.path.join(self.chkp_dir, 'best.pth.tar')

        state = torch.load(file, lambda s, l: s)

        optimizer_state = state.pop('optimizer')
        self.load_state_dict(state.pop('model'))
        if optimizer is not None:
            optimizer.load_state_dict(optimizer_state)

        return state

    def __str__(self):
        string = super(NetworkBase, self).__str__()
        string += '\n# of Parameters\n'
        nparams = calc_nparams(self)
        for k in nparams.keys():
            string += k + ': ' + str(nparams[k])+'\n'
        return string
