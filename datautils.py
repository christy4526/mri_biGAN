from __future__ import print_function, division, absolute_import

import os
import time
import pickle
import random

import numpy as np
import nibabel as nib
import torch

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import StratifiedKFold

from torchvision.transforms import Compose, Lambda
from .transforms import RandomCrop, NineCrop, ToTensor, Normalize

# **aware**
from ignite._utils import convert_tensor


def pickle_load(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data


def kfold_split(k=5, cur_fold=0,
                modality='MRI',
                labels=('CN', 'AD'),
                subject_ids_file=os.path.join(
                    'adni_dataset', 'subject_ids.pkl'),
                diagnosis_file=os.path.join(
                    'adni_dataset', 'diagnosis.pkl'),
                seed=1234, verbose=True):
    labels = tuple(labels)
    assert labels in [('CN', 'AD'), ('CN', 'MCI', 'AD')]
    subject_ids = pickle_load(subject_ids_file)
    diagnosis = pickle_load(diagnosis_file)

    X = np.array([sid
                  for sid in subject_ids[modality.lower()]
                  if diagnosis[sid] in labels])
    y = np.array([diagnosis[x] for x in X])

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    train, test = list(skf.split(X, y))[cur_fold]

    if verbose:
        print('Train ', tuple(
            zip(labels, [len(X[train][y[train] == label]) for label in labels])))
        print('Test ', tuple(
            zip(labels, [len(X[test][y[test] == label]) for label in labels])))

    train = list(zip(X[train], y[train]))
    test = list(zip(X[test], y[test]))
    return train, test


class AbstractDataset(Dataset):
    def __init__(self, root, data, labels, transform=None, get_sid=False, pstfx='.nii'):
        self.root = root
        self.abstract_data = data
        self.labels = labels
        self.transform = transform
        self.get_sid = get_sid

        self.pstfx = pstfx

    def load(self, sid, diagnosis):
        if self.pstfx == '.nii':
            path = os.path.join(self.root, sid+self.pstfx)
            proxy_img = nib.load(path)
            im = np.asarray(proxy_img.dataobj)[np.newaxis]

        elif self.pstfx == '.pth' or self.pstfx == '.pth.tar':
            # TODO pth
            raise NotImplementedError
        else:
            raise NotImplementedError(
                'Unrecognized postfix `{}`'.format(self.pstfx))

        target = self.labels.index(diagnosis)

        return im, target

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class ADNIClassDataset(AbstractDataset):
    def __init__(self, root, data, labels, class_idx, transform=None, get_sid=False, pstfx='.nii'):
        super(ADNIClassDataset, self).__init__(
            root, self._rectify(data, labels, class_idx), labels, transform, get_sid, pstfx)

    def _rectify(self, data, labels, class_idx):
        target_label = labels[class_idx]
        return [(sid, diag) for sid, diag in data if diag == target_label]

    def __len__(self):
        return len(self.abstract_data)

    def __getitem__(self, idx):
        sid, diag = self.abstract_data[idx]
        image, target = self.load(sid, diag)

        if self.transform is not None:
            image = self.transform(image)

        if self.get_sid:
            sample = (image, target, sid)
        else:
            sample = (image, target)

        return sample


class ADNIRandomSampleClassDataset(ADNIClassDataset):
    def __init__(self, root, data, labels, class_idx, num_samples, transform=None, get_sid=False, pstfx='.nii'):
        super(ADNIRandomSampleClassDataset, self).__init__(
            root, data, labels, class_idx, transform, get_sid, pstfx)
        self.num_samples = num_samples
        self.sampled = []

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.num_samples <= len(self.sampled):
            self.sampled = []
        sid, diag = random.choice(self.abstract_data)
        while sid in self.sampled:
            sid, diag = random.choice(self.abstract_data)
        self.sampled.append(sid)

        image, target = self.load(sid, diag)

        if self.transform is not None:
            image = self.transform(image)

        if self.get_sid:
            sample = (image, target, sid)
        else:
            sample = (image, target)

        return sample


class ADNIDataset(ConcatDataset):
    def __init__(self, root, data, labels, transform=None, get_sid=False, pstfx='.nii', balancing=False):
        class_datasets = [ADNIClassDataset(root, data, labels, class_idx, transform, get_sid, pstfx)
                          for class_idx in range(len(labels))]

        balancing_datasets = []
        if balancing:
            num_classes = [len(dset) for dset in class_datasets]
            lack_num_samples = [max(num_classes)-ncls for ncls in num_classes]

            balancing_datasets = [ADNIRandomSampleClassDataset(
                root, data, labels, lack_class_idx, num_samples, transform, get_sid, pstfx)
                for lack_class_idx, num_samples in enumerate(lack_num_samples) if num_samples != 0]

        super(ADNIDataset, self).__init__(class_datasets+balancing_datasets)


def transform_presets(mode):
    if mode == 'mri_random_crop':
        t = Compose([
            RandomCrop(112, 144, 112),
            ToTensor(),
            Normalize(0.5, 0.5)])
    elif mode == 'mri_nine_crop':
        t = Compose([
            NineCrop((157, 189, 156), (112, 144, 112)),
            Lambda(lambda patches: torch.stack([
                ToTensor()(patch) for patch in patches])),
            Normalize(0.5, 0.5)])

    return t


def get_dataloader(k=5, cur_fold=0, root='adni_dataset', modality='MRI', data_pstfx='.nii', load2mem=False,
                   labels=('CN', 'AD'), batch_size=40, seed=1234, balancing=False,
                   get_sid=False, verbose=True, num_workers=5):
    train, test = kfold_split(k=k, cur_fold=cur_fold,
                              modality=modality, labels=labels,
                              seed=seed, verbose=verbose)

    root = os.path.join(root, modality)
    trainset = ADNIDataset(root, train, labels,
                           transform_presets('mri_random_crop'),
                           get_sid, data_pstfx, balancing)
    testset = ADNIDataset(root, test, labels,
                          transform_presets('mri_nine_crop'),
                          get_sid, data_pstfx)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=True,
                             worker_init_fn=lambda _: torch.initial_seed())
    testloader = DataLoader(testset, num_workers=4, pin_memory=True)

    return trainloader, testloader


def process_ninecrop_batch(batch, device=None, non_blocking=False, metric_mode=False):
    x, y = batch

    if len(x.shape) == 6:
        _, npatches, c, i, j, k = x.shape
        x = x.view(-1, c, i, j, k)

    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


def mean_over_ninecrop(args):
    output, target = args
    if output.size(0) != target.size(0):
        n = target.size(0)
        npatches = output.size(0) // n
        output = output.view(n, npatches, *output.shape[1:])
        output = torch.mean(output, dim=1)
    return output, target
