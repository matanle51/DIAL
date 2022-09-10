import torch
from torch.utils.data import Dataset
from PIL import Image
from os.path import join
import numpy as np
from torchvision import transforms


class CIFAR10C(Dataset):
    def __init__(self, root, transform=None, attack_type=''):
        dataPath = join(root, '{}.npy'.format(attack_type))
        labelPath = join(root, 'labels.npy')

        self.data = np.load(dataPath)
        self.label = np.load(labelPath).astype(np.long)
        self.transform = transform


    def __getitem__(self, idx):

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = self.label[idx]

        return img, label

    def __len__(self):
        return self.data.shape[0]