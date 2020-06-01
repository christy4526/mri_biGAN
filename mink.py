from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import DataLoader
import random
import numpy as np


class NormalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ADDataset(Dataset):
    def __init__(self, data):
        self.data = [d for d in data if 'AD' in d]
        ncn = len([d for d in data if 'CN' in d])
        nad = len(self.data)
        self.lack = ncn - nad
        self.sampled = []

    def __len__(self):
        return self.lack

    def __getitem__(self, idx):
        sample = random.choice(self.data)
        while sample in self.sampled:
            sample = random.choice(self.data)

        self.sampled.append(sample)

        if len(self.sampled) >= self.lack:
            self.sampled = []

        return sample


if __name__ == '__main__':
    data = ['AD']*188+['CN']*228

    data = [d+str(idx) for idx, d in enumerate(data, 1)]

    normal_set = NormalDataset(data)
    ad_set = ADDataset(data)

    dataset = ConcatDataset([normal_set, ad_set])

    print('length of total dataset 228*2', len(dataset))

    loader = DataLoader(dataset)

    data = dict(ad=[], cn=[])

    for x in loader:
        x = x[0]

        if 'AD' in x:
            data['ad'].append(x)
        else:
            data['cn'].append(x)

    print('#ad', len(data['ad']), '#cn', len(data['cn']), 'balanced')
    unique = np.unique(data['ad'])

    for d in unique:
        data['ad'].pop(data['ad'].index(d))

    key = lambda x: int(x[2:])
    print('re-sampled ad data', sorted(data['ad'], key=key))
