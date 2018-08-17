from __future__ import division

import os
import pickle
import random

import numpy as np
import torch
import torch.utils.data as torch_data

from process_data import DataScaler
from process_data_dev import DATA_DIR_PATH


class MyDataset(torch_data.Dataset):
    def __init__(self, data_set):
        self.data_set = data_set
        self.scaler = DataScaler(DATA_DIR_PATH)

    def __len__(self):
        return len(self.data_set)
        pass

    def __getitem__(self, item):
        item = self.data_set[item]
        data_mat = self.scaler.normalize(item[0], 'cnn')
        data_mat = torch.from_numpy(data_mat.T).float()
        label = torch.from_numpy(np.array(item[1], dtype=int))
        return data_mat, label

def generate_data_set(split_ratio):
    """
    generate the train/test data set
    split the data set according to split ratio
    and load up to DataSet obj
    :param split_ratio: how much data save as test data, or train data
    :return: dict {
        'train': train DataSet,
        'test': test DataSet
    }
    """
    data_path = os.path.join(DATA_DIR_PATH, 'new_train_data')
    with open(data_path, 'r+b') as f:
        data_set = pickle.load(f)
    print('data set size %d' % len(data_set))
    random.shuffle(data_set)
    train_data = data_set[int(len(data_set) * split_ratio):]
    test_data = data_set[: int(len(data_set) * split_ratio)]
    return {
        'train': MyDataset(train_data),
        'test': MyDataset(test_data)
    }
