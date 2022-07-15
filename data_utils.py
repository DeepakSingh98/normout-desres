# Credit: https://github.com/GMvandeVen/continual-learning/blob/master/data.py

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
import torch

#----------------------------------------------------------------------------------------------------------#


class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].
    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indices = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indices.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indices)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indices[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample