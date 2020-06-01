import random
import numpy as np
import torch


class Crop(object):
    def __init__(self, *cropping):
        c = cropping
        self.cropping = (slice(c[2], -c[3] if c[3] != 0 else None),
                         slice(c[4], -c[5] if c[5] != 0 else None),
                         slice(c[0], -c[1] if c[1] != 0 else None))

    def __call__(self, data):
        data = data[self.cropping[0], self.cropping[1], self.cropping[2]]
        return data


class RandomCrop(object):
    def __init__(self, *cropping):
        self.cropping = cropping

    def __call__(self, data):
        i = [random.choice(list(range(0, data.shape[i+1]-self.cropping[i]-1)))
             for i in range(len(self.cropping))]

        s = [slice(None)] + [slice(i[r], i[r]+self.cropping[r])
                             for r in range(len(i))]
        return data[s]


class MultiModalRandomCrop(object):
    def __init__(self, mri_cropping, pet_cropping):
        self.mri_cropping = mri_cropping
        self.pet_cropping = pet_cropping

    def _crop(self, data, slices):
        return data[:, slices[0], slices[1], slices[2]]

    def __call__(self, mri, pet):
        mri_idx = [random.choice(list(range(0, mri.shape[i+1]-self.mri_cropping[i]-1)))
                   for i in range(len(self.mri_cropping))]
        pet_idx = np.array(mri_idx)//2

        mri_slices = [slice(mri_idx[i], mri_idx[i]+self.mri_cropping[i])
                      for i in range(len(self.mri_cropping))]
        pet_slices = [slice(pet_idx[i], pet_idx[i]+self.pet_cropping[i])
                      for i in range(len(self.pet_cropping))]

        return self._crop(mri, mri_slices), self._crop(pet, pet_slices)


class PartialFiveCrop(object):
    def __init__(self, inshape, outshape, index):
        self.inshape = np.array(inshape)
        self.outshape = np.array(outshape)

        ins = self.inshape
        ots = self.outshape
        rem = ins - ots
        m = rem // 2
        mr = ots + m

        self.index = index

        self.indices = [
            (slice(None), slice(0, ots[0]), slice(0, ots[1])),
            (slice(None), slice(0, ots[0]), slice(rem[1], None)),

            (slice(None), slice(rem[0], None), slice(0, ots[1])),
            (slice(None), slice(rem[0], None), slice(rem[1], None)),

            (slice(None), slice(m[0], mr[0]), slice(m[1], mr[1]))
        ]

    def __call__(self, image):
        t = image[self.indices[self.index]]
        return t


class FiveCrop(PartialFiveCrop):
    def __init__(self, inshape, outshape):
        super(FiveCrop, self).__init__(inshape, outshape, None)

    def __call__(self, image):
        patches = []
        for i in range(5):
            self.index = i
            patches += [super(FiveCrop, self).__call__(image)]
        return patches


class Pad(object):
    def __init__(self, *padding):
        self.padding = padding

    def __call__(self, data):
        return torch.nn.functional.pad(data, self.padding)


class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        data = data.float()
        data /= 255
        return data


class Normalize(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data = (data-self.mean)/self.std
        return data
