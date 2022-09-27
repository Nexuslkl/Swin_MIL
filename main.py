import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
import os

from models import *
from functions import *
from datasets import *
from utils import *


def main():
    print('Loading......')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset_name = 'colon'
    path_work = 'work/test/'
    if os.exist(path_work) is False:
        os.mkdir(path_work)

    path_train_pos = '../data/' + dataset_name + '/train/pos'
    path_train_neg = '../data/' + dataset_name + '/train/neg'
    path_train_gdt = '../data/' + dataset_name + '/train/gt'

    if os.path.exists('../data/' + dataset_name + '/valid'):
        path_valid_pos = '../data/' + dataset_name + '/valid/pos'
        path_valid_neg = '../data/' + dataset_name + '/valid/neg'
        path_valid_gdt = '../data/' + dataset_name + '/valid/gt'
    else:
        path_valid_pos = '../data/' + dataset_name + '/test/pos'
        path_valid_neg = '../data/' + dataset_name + '/test/neg'
        path_valid_gdt = '../data/' + dataset_name + '/test/gt'
    path_test_pos = '../data/' + dataset_name + '/test/pos'
    path_test_neg = '../data/' + dataset_name + '/test/neg'
    path_test_gdt = '../data/' + dataset_name + '/test/gt'

    dataset_size = [256, 256]
    dataset_train = Dataset_train(dataset_size, path_train_pos, path_train_neg, path_blank, device)
    dataset_valid = Dataset_valid(dataset_size, path_valid_pos, path_valid_neg, path_valid_gdt, device)
    dataset_test = Dataset_test(dataset_size, path_test_pos, path_test_neg, path_test_gdt, device)

    batch_size = 4
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=0)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    test_num_pos = 80

    model = SA_MIL(w=[0.2, 0.35, 0.45]).to(device)

    hyperparameters = {
        'r' : 4,
        'lr' : 1e-5,
        'wd' : 0.0005,
        'epoch' : 1000,
        'pretrain' : True,
        'optimizer' : 'side'  # side
    }

    print('Dataset: ' + dataset_name)
    print('Data Volume: ', len(dataloader_train.dataset))
    print('Model: ', type(model))
    print('Batch Size: ', batch_size)
    train(path_work, model, dataloader_train, device, hyperparameters, valid, dataloader_valid, test_num_pos)
    # test(path_work, model, dataloader_test, device)


if __name__ == '__main__':
    main()





