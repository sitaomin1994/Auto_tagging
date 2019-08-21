import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CodeChangeDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, y, transform=None):
        """
        Args:
            data: dataframe contains features and labels
            target_col : target columns name
        """
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sentence = self.X[idx]
        label = np.asarray(self.y[idx])
        length = np.where(sentence == 0)[0][0]
        sample = {'feature': sentence, 'label': label, 'length': length}

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sentence, label, length = sample['sentence'], sample['label'], sample['length']

        return {'feature': torch.from_numpy(sentence),
                'label': torch.from_numpy(label),
                'length': torch.from_numpy(length)}