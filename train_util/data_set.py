from __future__ import division

import torch.utils.data as data

class MyDataset(data.Dataset):
    def __init__(self, data):
        # no need inhert __init__, just a interface
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        data = None
        label = None
        pass
        return data, label

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
    pass
