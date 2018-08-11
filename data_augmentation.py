import os

import matplotlib.pyplot as plt
import numpy as np

import process_data_dev

"""
目前idea 
1 时间方向上进行平移
    对于有效段的截取 左右空闲区域的大小可以出现一定的变动
2 数值上的scaling  
    比如某些时间点值的大小可以以某一值为锚点，在正态分布的情况下进行波动
3 时间上的scaling
    加大或者降低某些值的变化速度 
"""

class DataAugment:
    def __init__(self):
        self.process_list = []

    def __call__(self, *args, **kwargs):
        data = args[0]
        for each_act in self.process_list:
            data = each_act(data)
        return data

    @staticmethod
    def scaling_in_amplitude(data):

        return data

    @staticmethod
    def scaling_in_time(data):
        domain_data_type = ['acc', 'gyr', 'emg']
        augmented_data = {}
        for each_cap in range(len(data['acc'])):
            for each_type in domain_data_type:
                pass

        return augmented_data

"""
data format:
    
"""

def load_train_data(sign_id, date, batch_range):
    data_path = os.path.join(DATA_DIR_PATH, 'resort_data', date)
    overall_data = None
    for batch in batch_range:
        curr_batch_data = process_data_dev.load_train_data(sign_id, batch, data_path)
        if len(curr_batch_data['acc']) == 0:
            continue
        if overall_data is None:
            overall_data = curr_batch_data
        else:
            for each_type in curr_batch_data.keys():
                overall_data[each_type] = np.vstack((overall_data[each_type], curr_batch_data[each_type]))
    return overall_data

def draw_box_plt():
    """
    todo 箱型图 用于估计每个由于动作的数据的分布情况
    :return:
    """

    # 存在离群点密集 batch 20    25

    data = load_train_data(sign_id=13,
                           date='0810-2',
                           batch_range=range(1, 16))
    for each_type in ['acc', 'gyr']:
        overall_data = [None, None, None]
        for each_cap in data[each_type]:
            each_cap = each_cap.T
            for each_dim in range(3):
                if overall_data[each_dim] is None:
                    overall_data[each_dim] = each_cap[each_dim]
                else:
                    overall_data[each_dim] = np.vstack((overall_data[each_dim], each_cap[each_dim]))
        for each_dim in range(3):
            # todo 使用箱型图进行数据过滤
            plt.figure(each_type + " %d" % each_dim)
            mean = np.mean(overall_data[each_dim], axis=0)
            st_med = np.percentile(overall_data[each_dim], q=15, axis=0)
            # 求百分位数 ，箱型图分别用的是75 和25
            rd_med = np.percentile(overall_data[each_dim], q=85, axis=0)
            lower_bound = st_med - (rd_med - st_med) * 0.5
            upper_bound = rd_med + (rd_med - st_med) * 0.5

            plt.boxplot(x=overall_data[each_dim], sym='x', usermedians=mean)
            plt.plot(mean)
            plt.plot(st_med)
            plt.plot(rd_med)
            plt.plot(lower_bound)
            plt.plot(upper_bound)
    plt.show()

DATA_DIR_PATH = os.path.join(os.getcwd(), 'data')

def read_gesture_table():
    f = open(os.path.join(DATA_DIR_PATH, 'gesture_table'), 'r', encoding='utf-8')
    lines = f.readlines()

    for each in lines:
        each = each.strip('\n')
        print("\'%s\'," % each, end='')

    f.close()
    pass

def main():
    # read_gesture_table()
    draw_box_plt()

if __name__ == '__main__':
    main()
