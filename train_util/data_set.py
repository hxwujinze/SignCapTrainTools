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


class SiameseNetworkTrainDataSet:
    """
    生成随机的相同或者不同的数据对进行训练
    """

    def __init__(self, data):
        """
        spilt data set into set of each categories as dict
        :param data:
        """
        self.data_len = len(data)
        self.data = data
        self.data_dict = {}
        for (each_label, each_data) in data:
            if self.data_dict.get(each_label) is None:
                self.data_dict[each_label] = [each_data]
            else:
                self.data_dict[each_label].append(each_data)
        self.class_cnt = len(self.data_dict.keys())

    # 要保证network尽可能见过最多class 并且能对他们进行正误的分辨
    def __getitem__(self, item):
        """
        50% probability, get same data or different data
        :param item:
        :return:
        """
        x1_ = random.choice(self.data)
        x1_label = x1_[0]
        x1_ = x1_[1]
        get_same = bool(random.randint(0, 1))
        if get_same:
            x2_ = random.choice(self.data_dict[x1_label])
        else:
            x2_label = x1_label
            while x2_label == x1_label:
                x2_label = random.randint(1, self.class_cnt)
            x2_ = random.choice(self.data_dict[x2_label])
        return (x1_, x2_), \
               np.array([0 if get_same else 1], dtype=np.float32)

    def look_input_data(self):
        """
        for test
        输出几个训练数据看看
        :return:
        """
        import matplotlib.pyplot as plt
        for i in range(20):
            each_sample = self[i]
            fig = plt.figure()
            fig.add_subplot(111, title=str(each_sample[2]))
            plt.plot(range(len(each_sample[0][0])), each_sample[0][0])
            plt.plot(range(len(each_sample[1][0])), each_sample[1][0])
        plt.show()

    def __len__(self):
        return self.data_len


def generate_data_set(split_ratio, data_set_type):
    """
    generate the train/test data set
    split the data set according to split ratio
    and load up to DataSet obj
    :param split_ratio: how much data save as test data, or train data
    :param data_set_type: which train data set type you load up in
                          pass the class into that
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
        'train': data_set_type(train_data),
        'test': data_set_type(test_data)
    }

