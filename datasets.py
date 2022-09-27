import torch
import os
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import Dataset

import PIL
import numpy as np


class Dataset_train(Dataset):
    def __init__(self, dataset_size, path_pos, path_neg, path_blk, device):
        super(Dataset_train, self).__init__()
        self.path_pos = path_pos
        self.path_neg = path_neg
        self.path_blk = path_blk
        self.list_pos = os.listdir(self.path_pos)
        self.list_neg = os.listdir(self.path_neg)
        self.list_blk = os.listdir(self.path_blk)
        self.list_pos.sort()
        self.list_neg.sort()
        self.list_blk.sort()
        self.num_pos = len(self.list_pos)
        self.num_neg = len(self.list_neg)
        self.num_blk = len(self.list_blk)
        self.device = device
        self.size = dataset_size
        self.transforms = transforms.Compose([
            transforms.Resize(self.size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=[164.7261, 129.4018, 176.4253], std=[43.0450, 49.9314, 32.2143])
                       ])

    def __getitem__(self, index):

        if index < self.num_pos:
            image = self.read(self.path_pos, self.list_pos[index])
            label = torch.ones(1)
        elif index < self.num_pos + self.num_neg:
            image = self.read(self.path_neg, self.list_neg[index - self.num_pos])
            label = torch.zeros(1)
        else:
            image = self.read(self.path_blk, self.list_blk[index - self.num_pos - self.num_neg])
            label = torch.zeros(1)

        return image, label

    def __len__(self):
        return self.num_pos + self.num_neg + self.num_blk

    def read(self, path, name):
        img = io.imread(os.path.join(path, name))
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        img = self.transforms(img)

        return img


class Dataset_valid(Dataset):
    def __init__(self, dataset_size, path_pos, path_neg, path_gdt, device):
        super(Dataset_valid, self).__init__()
        self.path_pos = path_pos
        self.path_neg = path_neg
        self.path_gdt = path_gdt
        self.list_pos = os.listdir(self.path_pos)
        self.list_neg = os.listdir(self.path_neg)
        self.list_gdt = os.listdir(self.path_gdt)
        self.list_pos.sort()
        self.list_neg.sort()
        self.list_gdt.sort()
        self.num_pos = len(self.list_pos)
        self.num_neg = len(self.list_neg)
        self.device = device
        self.size = dataset_size
        self.transforms_test = transforms.Compose([
            transforms.Resize(self.size),
            transforms.Normalize(mean=[164.7261, 129.4018, 176.4253], std=[43.0450, 49.9314, 32.2143])
                        ])
        self.transforms_grdth = transforms.Compose([
            transforms.Resize(self.size)
                        ])

    def __getitem__(self, index):

        if index < self.num_pos:
            image = self.read(self.path_pos, self.list_pos[index], 'test')
            grdth = self.read(self.path_gdt, self.list_gdt[index], 'grdth')
        else:
            image = self.read(self.path_neg, self.list_neg[index-self.num_pos], 'test')
            grdth = torch.zeros(1, self.size[0], self.size[1])

        return image, grdth

    def __len__(self):
        return self.num_pos + self.num_neg

    def read(self, path, name, norm=None):
        img = io.imread(os.path.join(path, name))

        if norm == 'test':
            img = torch.from_numpy(img).float().permute(2, 0, 1)
            img = self.transforms_test(img)

        elif norm == 'grdth':
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = torch.from_numpy(img).float().unsqueeze(0)
            img = self.transforms_grdth(img)
            img = (img > 0) + 0

        return img


class Dataset_test(Dataset):
    def __init__(self, dataset_size, path_pos, path_neg, path_gdt, device):
        super(Dataset_test, self).__init__()
        self.path_pos = path_pos
        self.path_neg = path_neg
        self.path_gdt = path_gdt
        self.list_pos = os.listdir(self.path_pos)
        self.list_neg = os.listdir(self.path_neg)
        self.list_gdt = os.listdir(self.path_gdt)
        self.list_pos.sort()
        self.list_neg.sort()
        self.list_gdt.sort()
        self.num_pos = len(self.list_pos)
        self.num_neg = len(self.list_neg)
        self.device = device
        self.size = dataset_size
        self.transforms_test = transforms.Compose([
            transforms.Resize(self.size),
            transforms.Normalize(mean=[164.7261, 129.4018, 176.4253], std=[43.0450, 49.9314, 32.2143])
        ])
        self.transforms_grdth = transforms.Compose([
            transforms.Resize(self.size)
        ])

    def __getitem__(self, index):

        if index < self.num_pos:
            image = self.read(self.path_pos, self.list_pos[index], 'test')
            label = self.read(self.path_gdt, self.list_gdt[index], 'grdth')
            image_show = self.read(self.path_pos, self.list_pos[index])
        else:
            image = self.read(self.path_neg, self.list_neg[index-self.num_pos], 'test')
            label = torch.zeros(self.size)
            image_show = self.read(self.path_neg, self.list_neg[index-self.num_pos])

        return image, label, image_show

    def __len__(self):
        return self.num_pos + self.num_neg

    def read(self, path, name, norm=None):
        img = io.imread(os.path.join(path, name))

        if norm == 'test':
            img = torch.from_numpy(img).float().permute(2, 0, 1)
            img = self.transforms_test(img)

        elif norm == 'grdth':
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = torch.from_numpy(img).float().unsqueeze(0)
            img = self.transforms_grdth(img)
            img = (img > 0) + 0

        return img
    

