import os
import pickle
from multiprocessing import Pool

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
    """
    load dedicate date captured data
    format as {
        'acc': nparray(
            each capture data nparray
        ),
        'gyr': same as upper,
        'emg': same as upper

        # data contains 3 or 8 dim, represent that each channel data in each time step

    }
    :param sign_id: which sigh gonna load
    :param date: capture date
    :param batch_range: the range of batch of this data
    :return: if doesn't find the target sign data, will return None
    """
    overall_data = None
    for each_date in date:
        data_path = os.path.join('cleaned_data', each_date)
        for batch in batch_range:
            batch = int(batch)
            curr_batch_data = process_data_dev.load_train_data(sign_id, batch, data_path, verbose=False)
            if len(curr_batch_data['acc']) == 0:
                continue
            if overall_data is None:
                overall_data = curr_batch_data
            else:
                for each_type in curr_batch_data.keys():
                    overall_data[each_type] = np.vstack((overall_data[each_type], curr_batch_data[each_type]))
    return overall_data

def draw_box_plt(data):
    """
    print the boxplot according to the input data batch
    data batch format same as function return value in load_data
    :param data input data batch
    :return:
    """
    print('boxplot')
    for each_type in ['acc', 'gyr', ]:
        overall_data = [None, None, None]
        dim_range = 3
        if each_type == 'emg':
            dim_range = 8
            overall_data = [None for i in range(dim_range)]
        for each_cap in data[each_type]:
            each_cap = each_cap.T
            for each_dim in range(dim_range):
                if overall_data[each_dim] is None:
                    overall_data[each_dim] = each_cap[each_dim]
                else:
                    overall_data[each_dim] = np.vstack((overall_data[each_dim], each_cap[each_dim]))
            if each_type == 'emg':
                overall_data = [np.abs(each_dim_data) for each_dim_data in overall_data]
        for each_dim in range(dim_range):
            plt.figure(each_type + " %d" % each_dim)
            median = np.median(overall_data[each_dim], axis=0)
            mean = np.mean(overall_data[each_dim], axis=0)
            st_med = np.percentile(overall_data[each_dim], q=20, axis=0)
            # 求百分位数 ，箱型图分别用的是75 和25
            rd_med = np.percentile(overall_data[each_dim], q=80, axis=0)
            mid_point = mean
            if each_type != 'emg':
                bound_rate = 1.5
                lower_bound = median - (rd_med - st_med) * bound_rate
                upper_bound = median + (rd_med - st_med) * bound_rate
            else:
                bound_rate = 3.5
                lower_bound = mid_point - (rd_med - st_med) * bound_rate
                upper_bound = mid_point + (rd_med - st_med) * bound_rate

            plt.boxplot(x=overall_data[each_dim], sym='x', usermedians=mid_point)
            plt.plot(mean)
            plt.plot(st_med)
            plt.plot(rd_med)
            plt.plot(lower_bound)
            plt.plot(upper_bound)
    plt.show()

def draw_plot(data):
    for each_type in ['acc', 'gyr']:
        for dim in range(3):
            plt.figure('%s dim%d' % (each_type, dim + 1))
            for each_cap in data[each_type][:100]:
                each_cap = each_cap.T[dim, :]
                plt.plot(each_cap)
    plt.show()


DATA_DIR_PATH = os.path.join(os.getcwd(), 'data')

from process_data_dev import GESTURES_TABLE


def data_distribution_statistics():
    """
    calculate data distribution by box plot method
    the result will save as following format
    {
        sign_id :{
            'acc':[
                each_channel of {
                    (ndarray)
                    'mean': mean,
                    'st_med': st_med,
                    'rd_med': rd_med,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                }
            ],
            'gyr':[
                same as upper
            ],
            'emg':[
                same as upper
            ]
        },....
    }

    :return:
    """

    task_list = []
    target_dir = os.path.join(DATA_DIR_PATH, 'cleaned_data')
    for each_sign in range(0, 2):
        print("processing sign %d %s" % (each_sign + 1, GESTURES_TABLE[each_sign]))
        each_sign += 1

        date_list = os.listdir(target_dir)
        for each_date in date_list:
            batch_list = os.listdir(os.path.join(target_dir, each_date))
            batch_list = map(lambda x: int(x), batch_list)

            batch_list = sorted(batch_list)
            for each_batch in batch_list:
                task_list.append((each_batch, each_date, each_sign))

    p = Pool(5)
    distributions = p.map(get_distribution_single, task_list)
    tmp = {}
    for each in distributions:
        if each is None:
            continue
        if tmp.get(each[2]) is None:
            tmp[each[2]] = [each]
        else:
            tmp[each[2]].append(each)
    distributions = tmp
    return distributions


def get_distribution_single(task_args):
    batch_num = task_args[0]
    date = task_args[1]
    sign_id = task_args[2]

    data_path = os.path.join('resort_data', date)
    data = process_data_dev.load_train_data(sign_id=sign_id,
                                            batch_num=batch_num,
                                            data_path=data_path)
    distributions = cal_distribution(data)
    return date, batch_num, sign_id, distributions


def cal_distribution(data):
    overall_data = {
        'acc': [None, None, None],
        'gyr': [None, None, None],
    }
    default_threshold = {
        'acc': 0.37,
        'gyr': 75
    }
    if len(data['acc']) == 0:
        return None

    for each_type in ['acc', 'gyr']:
        dim_range = 3
        for each_cap in data[each_type]:
            each_cap = each_cap.T
            for each_dim in range(dim_range):
                if overall_data[each_type][each_dim] is None:
                    overall_data[each_type][each_dim] = each_cap[each_dim]
                else:
                    overall_data[each_type][each_dim] = \
                        np.vstack((overall_data[each_type][each_dim], each_cap[each_dim]))
    distributions = {}
    total_failed_cnt = 0
    for each_type in ['acc', 'gyr']:
        distribution_data = []
        for each_dim in range(3):
            median = np.mean(overall_data[each_type][each_dim], axis=0)

            max_val = np.percentile(overall_data[each_type][each_dim], q=90, axis=0)
            min_val = np.percentile(overall_data[each_type][each_dim], q=15, axis=0)
            threshold_val = np.concatenate((median - min_val, max_val - median))
            threshold_val = np.mean(threshold_val)
            threshold_val = default_threshold[each_type] if threshold_val > default_threshold[
                each_type] else threshold_val
            print(threshold_val)

            st_pos = np.percentile(overall_data[each_type][each_dim], q=25, axis=0)
            rd_pos = np.percentile(overall_data[each_type][each_dim], q=75, axis=0)
            st_med_d = median - st_pos
            rd_med_d = rd_pos - median
            failed_cnt = 0
            for each_judge in (st_med_d, rd_med_d):
                book = np.where(each_judge > threshold_val, 1, 0)
                try:
                    outlier_cnt = np.count_nonzero(book[16:144])
                except IndexError:
                    distributions['judge_res'] = False
                    return distributions
                if outlier_cnt > 24:
                    failed_cnt += 1
            if failed_cnt > 0:
                is_failed = True
            else:
                is_failed = False

            distribution_data.append({
                'st_med_d': median - st_pos,
                'rd_med_d': rd_pos - median,
                'failed': is_failed
            })

            if is_failed:
                total_failed_cnt += 1

        distributions[each_type] = distribution_data

    if total_failed_cnt >= 3:
        distributions['judge_res'] = False
    else:
        distributions['judge_res'] = True
    return distributions


def data_clean(sign_id, data_batch):
    """
    check each capture in data batch, if one of the type data,contains more than
    40% outliers, remove that capture data
    :param sign_id:
    :param data_batch:
    :return:
    """
    if not os.path.exists(os.path.join(DATA_DIR_PATH, 'data_distributions.dat')):
        data_distribution_statistics()
    with open(os.path.join(DATA_DIR_PATH, 'data_distributions.dat'), 'r+b') as f:
        data_distribution = pickle.load(f)

    new_data_batch = {
        'acc': [],
        'gyr': [],
        'emg': []
    }

    for each_cap_iter in range(len(data_batch['acc'])):
        need_remove = False
        if each_cap_iter % 100 == 0:
            print('clean process %d/ %d' % (each_cap_iter, len(data_batch['acc'])))

        for each_type in ['acc', 'gyr']:
            each_cap_data = data_batch[each_type][each_cap_iter]
            check_rules = data_distribution[sign_id][each_type]
            outlier_dim_cnt = 0
            for each_dim in range(len(check_rules)):
                check_in_dim = check_rules[each_dim]
                book = np.zeros(160)
                cap_dim_data = each_cap_data.T[each_dim, :].T
                book = np.where(np.logical_and(cap_dim_data < check_in_dim['upper_bound'],
                                               cap_dim_data > check_in_dim['lower_bound']),
                                book,
                                cap_dim_data)
                outlier_cnt = np.count_nonzero(book[16:144])
                if outlier_cnt > 40:
                    outlier_dim_cnt += 1
            if outlier_dim_cnt >= 1:
                need_remove = True
                break

        if not need_remove:
            for each_type in ['acc', 'gyr', 'emg']:
                new_data_batch[each_type].append(data_batch[each_type][each_cap_iter])
    for each_type in ['acc', 'gyr', 'emg']:
        new_data_batch[each_type] = np.array(new_data_batch[each_type])
    print('after cleaned data set size %d' % len(new_data_batch['acc']))
    return new_data_batch


def read_gesture_table():
    f = open(os.path.join(DATA_DIR_PATH, 'gesture_table'), 'r', encoding='utf-8')
    lines = f.readlines()

    for each in lines:
        each = each.strip('\n')
        print("\'%s\'," % each, end='')

    f.close()
    pass

def clean_all_data(date_list=None):
    if date_list is None:
        date_list = os.listdir(os.path.join(DATA_DIR_PATH, 'resort_data'))
    cleaned_data = []
    sign_cnt = []
    sign_cnt_cleaned = []
    for each_sign in range(len(GESTURES_TABLE)):
        each_sign += 1
        data = load_train_data(sign_id=each_sign,
                               date=date_list,
                               batch_range=range(1, 99))
        if data is None:
            sign_cnt.append(0)
            sign_cnt_cleaned.append(0)
            cleaned_data.append({
                'acc': [],
                'gyr': [],
                'emg': []
            })
            continue
        sign_cnt.append(len(data['acc']))
        data = data_clean(each_sign, data)
        cleaned_data.append(data)
        sign_cnt_cleaned.append(len(data['acc']))
    with open(os.path.join(DATA_DIR_PATH, 'cleaned_data.dat'), 'w+b') as f:
        pickle.dump(cleaned_data, f)
    print('after clean')
    all_cnt = 0
    all_cnt_clean = 0
    for each_sign in range(len(GESTURES_TABLE)):
        all_cnt += sign_cnt[each_sign]
        all_cnt_clean += sign_cnt_cleaned[each_sign]
        print('sign %d %s cnt from %d to %d' % (each_sign + 1,
                                                GESTURES_TABLE[each_sign],
                                                sign_cnt[each_sign],
                                                sign_cnt_cleaned[each_sign]))
    print('all cnt from %d to %d' % (all_cnt, all_cnt_clean))


def load_data(sign_id, date_list=None):
    if date_list is None:
        date_list = os.listdir(os.path.join(DATA_DIR_PATH, 'cleaned_data'))
    # date_list = ['0811-2']
    # date_list.remove('0812-1')
    # print(date_list)
    data = load_train_data(sign_id=sign_id, date=date_list, batch_range=range(1, 99))
    return data


def show_data_distribution(distribution):
    for each_sign in distribution.keys():
        for each_type in ['acc', 'gyr']:
            for each_dim in range(3):
                for each_val in ['st_med_d', 'rd_med_d', ]:
                    plt.figure()
                    for each_batch in distribution[each_sign]:
                        data = each_batch[-1][each_type][each_dim][each_val]
                        plt.title('sign %d %s %d %s' % (each_sign, each_type, each_dim, each_val))
                        plt.plot(range(len(data)), data, )
        plt.show()




def clean_data_test(sign_id):
    data = load_data(sign_id)
    if data is None:
        print('data count is 0')
        return
    # data_distribution_statistics(True)
    draw_box_plt(data)
    data = data_clean(sign_id=sign_id, data_batch=data)
    draw_plot(data)

def main():
    # read_gesture_table()
    # 存在离群点密集 batch 20    25
    distribution = data_distribution_statistics()
    # clean_all_data()
    # clean_data_test(31)
    show_data_distribution(distribution)
    # clean_data_test(58)
    # data = load_train_data(sign_id=1,
    #                        batch_range=range(99),
    #                        date=['0810-2'])
    # draw_box_plt(data)
    # draw_plot(data)
    pass
if __name__ == '__main__':
    main()
