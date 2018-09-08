# coding:utf-8
# py3
import os
import pickle
import random
import shutil
import time
from multiprocessing import Pool

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.legend_handler import HandlerLine2D
from torch.autograd import Variable

import process_data
# from models.CNN_model import CNN, get_max_index
from models.verify_model import SiameseNetwork
from process_data import feature_extract_single, feature_extract, TYPE_LEN, \
    append_single_data_feature, append_feature_vector, wavelet_trans

Width_EMG = 9
Width_ACC = 3
Width_GYR = 3
LENGTH = 160

WINDOW_STEP = 16
torch.nn.RReLU
EMG_WINDOW_SIZE = 3
FEATURE_LENGTH = 44

DATA_DIR_PATH = os.path.join(os.getcwd(), 'data')
print(DATA_DIR_PATH)
myfont = font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
mpl.rcParams['axes.unicode_minus'] = False

CAP_TYPE_LIST = ['acc', 'gyr', 'emg']

OLD_GESTURES_TABLE = ['肉 ', '鸡蛋 ', '喜欢 ', '您好 ', '你 ', '什么 ', '想 ', '我 ', '很 ', '吃 ',  # 0-9
                  '老师 ', '发烧 ', '谢谢 ', '', '大家 ', '支持 ', '我们 ', '创新 ', '医生 ', '交流 ',  # 10 - 19
                  '团队 ', '帮助 ', '聋哑人 ', '请 ']  # 20 - 23

GESTURES_TABLE = ['朋友', '下午', '天', '早上', '上午', '中午', '谢谢', '对不起', '没关系', '昨天', '今天',
                  '明天', '家', '回', '去', '迟到', '交流', '联系', '你', '什么', '想', '我', '机场', '晚上',
                  '卫生间', '退', '机票', '着急', '怎么', '办', '行李', '可以', '托运', '起飞', '时间', '错过',
                  '改签', '航班', '延期', '请问', '怎么走', '在哪里', '找', '不到', '没收', '为什么', '航站楼',
                  '取票口', '检票口', '身份证', '手表', '钥匙', '香烟', '刀', '打火机', '沈阳', '大家',
                  '支持', '我们', '医生', '帮助', '聋哑人', '', '充电', '寄存', '中国', '辽宁', '北京',
                  '世界']

SIGN_COUNT = len(GESTURES_TABLE)


# process train data
def load_train_data(sign_id, batch_num, data_path='collected_data', verbose=True):
    """
    从采集文件夹中读取采集数据 并对数据进行裁剪
    读取数据 从文件中读取数据
    数据格式:
    {
        'acc': [ 每次手语的160个数据组成的nparrayy 一共20个 一次采集一个ndarray]
                ndarray中每个数据的形式以 [x, y, z] 排列
        'gyr': [ 同上 ]
        'emg': [ 形式同上 但是有8个维度]
    }

    py2 py3 pickle 不能通用
    :param sign_id: 需要取得的手语id
    :param batch_num: 需要取得的手语所在的batch数
    :return: 返回dict  包含这个手语的三种采集数据, 多次采集的数据矩阵的list
    """
    # Load and return data
    # initialization
    path = os.path.join(DATA_DIR_PATH, data_path)
    file_num = sign_id
    file_emg = os.path.join(path, str(batch_num), 'Emg', str(file_num) + '.txt')
    data_emg = file2matrix(file_emg, Width_EMG)
    file_acc = os.path.join(path, str(batch_num), 'Acceleration', str(file_num) + '.txt')
    data_acc = file2matrix(file_acc, Width_ACC)
    file_gyr = os.path.join(path, str(batch_num), 'Gyroscope', str(file_num) + '.txt')
    data_gyr = file2matrix(file_gyr, Width_GYR)

    processed_data_emg = []
    processed_data_acc = []
    processed_data_gyr = []
    if len(data_emg) != 0:
        capture_tag_list = list(data_emg[:, -1])
        capture_length_book = {}
        for i in capture_tag_list:
            capture_length_book[i] = capture_length_book.get(i, 0) + 1
        index = 0
        capture_times = len(capture_length_book.keys())
        capture_times = capture_times if capture_times < 20 else 21
        start_at = 1
        if batch_num >= 20:
            capture_times = capture_times if capture_times < 20 else 21
            start_at = 0

        for i in range(start_at, capture_times):
            resize_data_emg = length_adjust(data_emg[index:index + capture_length_book[i], 0:8])
            if resize_data_emg is None:
                continue
            processed_data_emg.append(resize_data_emg)  # init
            resize_data_acc = length_adjust(data_acc[index:index + capture_length_book[i], :])
            processed_data_acc.append(resize_data_acc)
            resize_data_gyr = length_adjust(data_gyr[index:index + capture_length_book[i], :])
            processed_data_gyr.append(resize_data_gyr)
            index += capture_length_book[i]
        if verbose:
            print('Load done , batch num: %d, sign id: %d, ' % (batch_num, sign_id,))

    return {
        'emg': processed_data_emg,  # 包含这个手语多次采集的数据矩阵的list
        'acc': processed_data_acc,
        'gyr': processed_data_gyr,
    }


def file2matrix(filename, data_col_num):
    del_sign = '()[]'
    separator = ','
    try:
        fr = open(filename, 'r', encoding='utf8')
    except IOError:
        lines_num = 0
        return np.zeros((lines_num, data_col_num), dtype=float)
    all_array_lines = fr.readlines()
    fr.close()
    lines_num = len(all_array_lines)
    return_matrix = np.zeros((lines_num, data_col_num), dtype=float)
    index = 0
    for line in all_array_lines:
        line = line.strip()
        line = line.strip(del_sign)
        list_from_line = line.split(separator)
        return_matrix[index, :] = list_from_line[0:data_col_num]
        index += 1
    return return_matrix

def length_adjust(data):
    """
    主要是对老数据进行兼容 ，
    对于新采集程序 该功能无效
    :param data:
    :return:
    """
    tail_len = len(data) - LENGTH
    if tail_len < 0:
        print('Length Error')
        adjusted_data = None
    else:
        # 前后各去掉多出来长度的一半
        end = len(data) - tail_len / 2
        begin = tail_len / 2
        adjusted_data = data[int(begin):int(end), :]
    return adjusted_data


def trans_data_to_time_seqs(data_set):
    return data_set.T

def expand_emg_data(data):
    expnded = []
    for each_data in data:
        each_data_expand = expand_emg_data_single(each_data)
        expnded.append(np.array(each_data_expand))
    return expnded

def expand_emg_data_single(data):
    each_data_expand = []
    for each_dot in range(len(data)):
        for time in range(16):
            each_data_expand.append(data[each_dot][:])
    return each_data_expand

def cut_out_data(data):
    for each_cap_type in CAP_TYPE_LIST:
        for each_data in range(len(data[each_cap_type])):
            data[each_cap_type][each_data] = data[each_cap_type][each_data][16:144, :]
    return data


def pickle_train_data_new():
    with open(os.path.join(DATA_DIR_PATH, 'cleaned_data.dat'), 'r+b') as f:
        data_set = pickle.load(f)

    train_data_set = []
    for each_sign in range(len(GESTURES_TABLE)):
        print("process sign %d" % each_sign)
        raw_data_set = data_set[each_sign]
        extracted_data_set = []
        for each_cap_type in CAP_TYPE_LIST:
            print("extracting %s" % each_cap_type)
            if each_cap_type == 'emg':
                extracted_data_set.append(process_data.emg_feature_extract(raw_data_set, True)['trans'])
            else:
                extracted_data_blocks = feature_extract(raw_data_set, each_cap_type)
                extracted_data_set.append(extracted_data_blocks['poly_fit'])
        extracted_data_set = append_feature_vector(extracted_data_set)
        # stack up for normalization
        overall_data_mat = None
        for each_mat in extracted_data_set:
            if overall_data_mat is None:
                overall_data_mat = each_mat
            else:
                overall_data_mat = np.vstack((overall_data_mat, each_mat))
            # save into train data
            train_data_set.append((each_mat, each_sign))

    scaler = process_data.DataScaler(DATA_DIR_PATH)

    scaler.generate_scale_data(overall_data_mat, 'cnn')
    vectors_name = ['cnn_acc', 'cnn_gyr', 'cnn_emg']
    vectors_range = ((0, 3), (3, 6), (6, 14))
    scaler.split_scale_vector('cnn', vectors_name, vectors_range)

    # scaler.expand_scale_data()
    scaler.store_scale_data()

    with open(os.path.join(DATA_DIR_PATH, 'new_train_data'), 'w+b') as f:
        pickle.dump(train_data_set, f)




def pickle_train_data(batch_num, feedback_data=None):
    """
    从采集生成的文件夹中读取数据 存为python对象
    同时生成RNN CNN的两种数据

    采用追加的方式 当采集文件夹数大于当前数据对象batch最大值时 进行数据的追加
    :param batch_num: 当前需要提取的采集数据文件夹数
    :param feedback_data 是否将之前feedback数据纳入训练集
    """
    model_names = ['cnn']

    train_data_set = {
        'rnn': [],
        'cnn': []
    }

    overall_data = {
        'rnn': None,
        'cnn': None
    }
    extract_time = time.clock()
    for each_batch in range(1, batch_num + 1):
        for each_sign in range(1, len(GESTURES_TABLE) + 1):
            # 一个手势一个手势的读入数据
            raw_data_set = load_train_data(batch_num=each_batch, sign_id=each_sign)
            extracted_data_set = {
                'rnn': [],
                'cnn': []
            }

            # 根据数据采集种类 提取特征
            for each_cap_type in CAP_TYPE_LIST:
                if each_cap_type == 'emg':
                    extracted_data_set['rnn'].append(process_data.emg_feature_extract(raw_data_set, False)['trans'])
                    extracted_data_set['cnn'].append(process_data.emg_feature_extract(raw_data_set, True)['trans'])
                else:
                    extracted_data_blocks = feature_extract(raw_data_set, each_cap_type)
                    extracted_data_set['rnn'].append(extracted_data_blocks['append_all'])
                    extracted_data_set['cnn'].append(extracted_data_blocks['poly_fit'])

            for each_name in model_names:
                extracted_data_set[each_name] = append_feature_vector(extracted_data_set[each_name])
                # print('append %s took %f' % (each_name, time.clock() - append_time))

                for each_data_mat in extracted_data_set[each_name]:
                    if overall_data[each_name] is None:
                        overall_data[each_name] = each_data_mat
                    else:
                        overall_data[each_name] = np.vstack((overall_data[each_name], each_data_mat))
                    train_data_set[each_name].append((each_sign, each_data_mat))

    scaler = process_data.DataScaler(DATA_DIR_PATH)

    for model_type in model_names:
        scaler.generate_scale_data(overall_data[model_type], model_type)
        if model_type == 'rnn':
            vectors_name = ['rnn_acc', 'rnn_gyr', 'rnn_emg']
            vectors_range = [(0, 11), (11, 22), (22, 30)]
        else:
            vectors_name = ['cnn_acc', 'cnn_gyr', 'cnn_emg']
            vectors_range = ((0, 3), (3, 6), (6, 14))
        scaler.split_scale_vector(model_type, vectors_name, vectors_range)

    scaler.expand_scale_data()
    scaler.store_scale_data()


    for each_model_type in model_names:
        data_set = train_data_set[each_model_type]
        for each in range(len(data_set)):
            data_set[each] = (data_set[each][0],
                              scaler.normalize(data_set[each][1], each_model_type))
        file = open(DATA_DIR_PATH + '\\data_set_' + each_model_type, 'w+b')
        pickle.dump((batch_num, train_data_set[each_model_type]), file)
        file.close()

    print('extract take %f' % (time.clock() - extract_time))

"""
raw capture data: {
    'acc': 采集时acc的buffer 连续的2维数组 
           第一维是采集时刻 第二维是各个通道的数据
    'gyr: 同上
    'emg':同上
}

processed data:[
    {
        'data': 数据块 2维数组
            第一维是一段手语采集每个window提取的feature vector
            第二维是feature vector的各个数据
        'index': 识别出来的手语id
        'time': 数据被传入处理时的进程时间点
    } ....
]

"""


# load data from online..
def load_feed_back_data():
    """
    从feedback 数据对象中获得数据
    并转换为符合绘图and训练数据的形式
    :return:[  [ dict(三种采集类型数据)该种手语的每次采集数据 ,... ] 每种手语 ,...]
    """
    file_name = \
        r'C:\Users\Scarecrow\PycharmProjects\SignProjectServerPy2\utilities_access\models_data\feedback_data_'
    file_ = open(file_name, 'r+b')
    # file = open('data_aaa', 'r+b')
    feedback_data_set = pickle.load(file_, encoding='iso-8859-1')
    # [ (sign_id, data), .....  ]
    file_.close()
    data_set = list(range(SIGN_COUNT))
    for each_cap in feedback_data_set:
        data_set[each_cap[0]] = each_cap[1]
    return data_set


def load_online_processed_data():
    """
    加载所有的processed data history 每个文件分开存放在一个list里
    list中每个数据是个dict  包含以下内容
    :return: [
        {
            'data' : 进行特征提取后 可以直接输入nnet的矩阵
                    时序x三种采集方式的特征提取后的拼接向量
            'index' : 该数据的识别结果
            'time' :
        }....
    ]
    """
    data_list = []
    print('got online processed data file list :')
    file_cnt = 1
    history_data_path = os.path.join(DATA_DIR_PATH, 'history_data')
    for root, dirs, files in os.walk(history_data_path):
        for file_ in files:
            if file_.startswith('history_recognized_data'):
                print(str(file_cnt) + '. ' + file_)
                file_cnt += 1
                file_ = history_data_path + '\\' + file_
                file_ = open(file_, 'rb')
                data = pickle.load(file_)
                data_list.append(data)
                file_.close()
    print('select online processed data:')
    index_ = int(input()) - 1
    history_data = {
        'data': data_list[index_],
        'for_cnn': 'True'
    }
    return history_data


def load_raw_capture_data():
    """
    读入raw capture data
    交互式的输入要加载的 raw capture data文件
    :return: dict{
        'acc': ndarray  时序 x 通道
        'gyr': ndarray
        'emg'；ndarray
    }
    """
    data_list = []
    file_id = 1
    print('file list: ')
    history_data_path = os.path.join(DATA_DIR_PATH, 'history_data')
    for root, dirs, files in os.walk(history_data_path):
        for file_ in files:
            if file_.startswith('raw_data_history'):
                print(str(file_id) + '. ' + file_)
                file_ = history_data_path + '\\' + file_
                file_ = open(file_, 'rb')
                data = pickle.load(file_, encoding='iso-8859-1')
                data_list.append(data)
                file_.close()
                file_id += 1
    print('get %d history data\ninput selected data num:' % len(data_list))
    num = input()
    num = int(num) - 1

    selected_data = data_list[num]

    selected_data = {
        'acc': np.array(selected_data['acc']),
        'gyr': np.array(selected_data['gyr']),
        'emg': np.array(selected_data['emg'])
    }
    return selected_data


# process data from online
def split_online_processed_data(online_data):
    """
    将 recognize history data种 直接输入算法的输入mat进行拆分
    将其转换为 各个采集类型以及各种特征提取方式分开的 格式
    同时将每个数据段进行拼接 生成一个连续的数据
    :param online_data: history recognized 文件直接pickle.load后的dict
    :return: tuple(拆分后的数据块, 连续的全局数据)
    """
    splited_data_list = []
    overall_data_list = {
        'acc': None,
        'gyr': None,
        'emg': None
    }

    data_part = online_data['data']
    is_for_cnn = online_data['for_cnn']
    for each_data in data_part:
        # 先对输入数据进行拆分
        if is_for_cnn == 'False':
            # 之前的数据提取方式会对数据进行多种数据提取方式
            # 扩大了输入矩阵的特征向量宽度
            acc_data = each_data['data'][:, 0:15]
            gyr_data = each_data['data'][:, 15:30]
            emg_data = each_data['data'][:, 30:]
        else:
            # 目前cnn 的输入不进行过多的特征提取操作
            acc_data = each_data['data'][:, 0:3]
            gyr_data = each_data['data'][:, 3:6]
            emg_data = each_data['data'][:, 6:]


        overall_data_list['acc'] = \
            append_overall_data(overall_data_list['acc'], acc_data, for_cnn=is_for_cnn)
        acc_data = split_features(acc_data)

        overall_data_list['gyr'] = \
            append_overall_data(overall_data_list['gyr'], gyr_data, for_cnn=is_for_cnn)
        gyr_data = split_features(gyr_data)

        overall_data_list['emg'] = \
            append_overall_data(overall_data_list['emg'], emg_data, for_cnn=is_for_cnn)
        emg_data = {
            'trans': [emg_data]
        }

        splited_data_list.append({
            'acc': acc_data,
            'gyr': gyr_data,
            'emg': emg_data
        })

    overall_data_list['acc'] = \
        split_features(overall_data_list['acc'])
    overall_data_list['gyr'] = \
        split_features(overall_data_list['gyr'])
    overall_data_list['emg'] = {
        'trans': [overall_data_list['emg']]
    }
    return splited_data_list, overall_data_list

def append_overall_data(curr_data, next_data, for_cnn):
    """
    拼接完整的采集数据
    :param curr_data: 当前已经完成拼接的数据
    :param next_data: 下一个读入的数据
    :param for_cnn 设置是否拼接操作是否是为CNN的输出拼接 如果是 需要进行不同的操作
    :return: 拼接完成的数据
    """
    if curr_data is None:
        curr_data = next_data
    else:
        # 只取最后一个数据点追加在后面
        if for_cnn == "False":
            curr_data = np.vstack((curr_data, next_data[-1, :]))
        else:
            curr_data = np.vstack((curr_data, next_data[-8:, :]))

    return curr_data


def split_features(data):
    # 只有raw的情况
    if len(data[0]) == 3:
        return {
            'rms': [],
            'zc': [],
            'arc': [],
            'cnn_raw': [data]
        }

    # 正常有其他几种特征的情况
    rms_feat = data[:, :3]
    zc_feat = data[:, 3:6]
    arc_feat = data[:, 6:18]
    return {
        'rms': [rms_feat],
        'zc': [zc_feat],
        'arc': [arc_feat]
    }

def process_raw_capture_data(raw_data, for_cnn=False):
    """
    对raw capture data进行特征提取等处理 就像在进行识别前对数据进行处理一样
    将raw capture data直接转换成直接输入算法识别进程的data block
    用于对识别时对输入数据处理情况的还原和模拟 便与调参
    加入了拓展归一化的功能  对288 窗口的数据进行归一化
    然后再以128的窗口特征提取
    :param raw_data: 选择的raw capture data ，load_raw_capture_data()的直接输出
    :param for_cnn
    :return: 返回格式与recognized history data 相同格式的数据
    """

    normalized_ptr_start = 0
    normalized_ptr_end = 160  # (288 - 128 )  = 160
    feat_extract_ptr_start = 0
    feat_extract_ptr_end = 128
    normalized_data = {
        'acc': None,
        'gyr': None,
        'emg': None,
    }
    data_scaler = process_data.DataScaler(DATA_DIR_PATH)
    start_ptr = 0
    end_ptr = 160
    processed_data = {
        'data': [],
        'for_cnn': str(for_cnn)
    }
    while end_ptr < len(raw_data['acc']):
        # time.sleep(0.16)
        # print("input sector: start ptr %d, end_ptr %d" % (start_ptr, end_ptr))
        if not for_cnn:
            acc_feat = feature_extract_single(raw_data['acc'][start_ptr:end_ptr, :], 'acc')
            gyr_feat = feature_extract_single(raw_data['gyr'][start_ptr:end_ptr, :], 'gyr')
            emg_feat = wavelet_trans(raw_data['emg'][start_ptr:end_ptr, :])
            all_feat = append_single_data_feature(acc_feat[3], gyr_feat[3], emg_feat)
        else:
            if end_ptr >= normalized_ptr_end:
                # print("normalized sector: start ptr %d, end_ptr %d" % (normalized_ptr_start, normalized_ptr_end))
                type_eumn = ['acc', 'gyr']
                for each_type in type_eumn:
                    data_seg = raw_data[each_type][normalized_ptr_start:normalized_ptr_end, :]
                    tmp = data_seg
                    tmp = data_scaler.normalize(tmp, 'cnn_' + each_type)
                    if normalized_data[each_type] is None:
                        normalized_data[each_type] = tmp
                    else:
                        normalized_data[each_type] = np.vstack(
                            (normalized_data[each_type], tmp[-16:, :]))
                normalized_ptr_start += 16
                normalized_ptr_end += 16
            if normalized_ptr_end >= feat_extract_ptr_end:
                print(
                    "feature extract sector: start ptr %d, end_ptr %d" % (feat_extract_ptr_start, feat_extract_ptr_end))
                acc_feat = normalized_data['acc'][feat_extract_ptr_start:feat_extract_ptr_end, :]
                gyr_feat = normalized_data['gyr'][feat_extract_ptr_start:feat_extract_ptr_end, :]

                acc_feat = process_data.feature_extract_single_polyfit(acc_feat, 2)
                gyr_feat = process_data.feature_extract_single_polyfit(gyr_feat, 2)
                emg_feat = wavelet_trans(raw_data['emg'][feat_extract_ptr_start:feat_extract_ptr_end, :])
                # 滤波后伸展
                emg_feat = process_data.expand_emg_data_single(emg_feat)
                all_feat = append_single_data_feature(acc_feat, gyr_feat, emg_feat)
                extract_step = random.randint(8, 24)
                feat_extract_ptr_end += extract_step
                feat_extract_ptr_start += extract_step
                processed_data['data'].append({'data': all_feat})
        start_ptr += WINDOW_STEP
        end_ptr += WINDOW_STEP
    return processed_data

# plot output
def generate_plot(data_set, data_cap_type, data_feat_type):
    """
    根据参数设置生成plot 但是不显示
    是个应该被其他print plot调用的子函数
    直接调用不会输出折线图
    :param data_set:
    :param data_cap_type:
    :param data_feat_type:
    :return:
    """
    if data_feat_type != 'arc':
        dim_size = TYPE_LEN[data_cap_type]
    else:
        dim_size = len(data_set[data_feat_type][0][0, :])  # 三个维度的三次多项式拟合的四个系数
    for dimension in range(dim_size):
        fig_ = plt.figure()
        if data_feat_type != 'arc':
            plt_title = '%s %s dim%s' % (data_feat_type, data_cap_type, str(dimension + 1))
        else:
            plt_title = 'arc dim %d param %d' % (dimension / 4 + 1, dimension % 4 + 1)

        fig_.add_subplot(111, title=plt_title)
        capture_times = len(data_set[data_feat_type])
        capture_times = capture_times if capture_times < 20 else 20
        # capture_times = 1

        # 最多只绘制20次采集的数据 （太多了会看不清）
        handle_lines_map = {}
        for capture_num in range(0, capture_times):
            single_capture_data = trans_data_to_time_seqs(data_set[data_feat_type][capture_num])
            data = single_capture_data[dimension]
            plot = plt.plot(range(len(data)), data, '.-', label='cap %d' % capture_num, )
            handle_lines_map[plot[0]] = HandlerLine2D(numpoints=1)
            plt.pause(0.008)
        plt.legend(handler_map=handle_lines_map)


scaler = process_data.DataScaler(DATA_DIR_PATH)


def print_train_data(sign_id,
                     batch_num,
                     data_cap_type,
                     data_feat_type,
                     capture_date=None,
                     data_path='collected_data',
                     for_cnn=False):
    """
    从采集文件中将 训练用采集数据 绘制折线图
    :param sign_id:
    :param batch_num:
    :param data_cap_type:
    :param data_feat_type:
    :param capture_date:
    :param data_path: collect data path
    :param for_cnn
    """
    if capture_date is not None:
        # data_path = os.path.join('resort_data', capture_date)
        data_path = os.path.join(data_path, capture_date)

    data_set = load_train_data(sign_id=sign_id,
                               batch_num=batch_num,
                               data_path=data_path)  # 从采集文件获取数据
    if data_cap_type == 'emg':
        data_set = process_data.emg_feature_extract(data_set, for_cnn)
    else:
        data_set = feature_extract(data_set, data_cap_type)

    global scaler
    to_scale_data = data_set[data_feat_type]
    if for_cnn:
        scale_type_name = 'cnn_%s' % data_cap_type
    else:
        scale_type_name = 'rnn_%s_%s' % (data_cap_type, data_feat_type)

    # if data_feat_type != 'raw' and data_feat_type != 'trans':
    #     for each in range(len(to_scale_data)):
    #         to_scale_data[each] = scaler.normalize(to_scale_data[each], scale_type_name)

    generate_plot(data_set, data_cap_type, data_feat_type)
    plt.show()

def print_raw_capture_data():
    """
    显示raw capture data的时序信号折线图
    """
    selected_data = load_raw_capture_data()
    print('input selected raw capture data type: ')
    selected_type = input()
    selected_data = selected_data[selected_type]
    selected_data = selected_data.T
    for each_dim in selected_data:
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(range(len(each_dim)), each_dim)
    plt.show()

def print_processed_online_data(data, cap_type, feat_type, block_cnt=0, overall=True, ):
    """
    输出处理后的数据 就是在识别时可以直接输入算法的数据
    :param data:
    :param cap_type:
    :param feat_type:
    :param block_cnt:
    :param overall:
    :return:
    """
    data_single = data[0]
    data_overall = data[1]
    if not overall:
        for each_cap in data_single:
            if block_cnt == 0:
                break
            block_cnt -= 1
            try:
                print(each_cap['index'])
            except KeyError:
                print('index unknown')
            generate_plot(each_cap[cap_type], cap_type, feat_type)
    else:
        generate_plot(data_overall[cap_type], cap_type, feat_type)
    plt.show()


def cnn_recognize_test(online_data):
    # verifier = SiameseNetwork(train=False)
    # load_model_param(verifier, 'verify_model')
    # verifier.double()
    # verifier.eval()

    cnn = CNN()
    cnn.double()
    cnn.eval()
    cnn.cpu()
    load_model_param(cnn, 'cnn_model')

    file_ = open(DATA_DIR_PATH + '\\reference_verify_vector_cnn', 'rb')
    verify_vectors = pickle.load(file_)
    file_.close()
    online_data = online_data['data']
    for each in online_data:
        start_time = time.clock()
        x = np.array([each['data'].T])
        x = torch.from_numpy(x).double()
        x = Variable(x)
        y = cnn(x)
        predict_index = get_max_index(y)[0]
        cnn_cost_time = time.clock() - start_time
        start_time = time.clock()
        print('\nindex from cnn %d' % predict_index)
        print('sign: %s' % GESTURES_TABLE[predict_index])
        # verify_vec = verifier(x)
        # reference_vec = np.array([verify_vectors[predict_index + 1]])
        # reference_vec = Variable(torch.from_numpy(reference_vec).double())
        # diff = F.pairwise_distance(verify_vec, reference_vec)
        # diff = torch.squeeze(diff).data[0]
        # print('diff %f' % diff)
        verifier_cost_time = time.clock() - start_time
        print('time cost : cnn %f, verify %f' % (cnn_cost_time, verifier_cost_time))

def generate_verify_vector(model_type):
    """
    根据所有训练数据生成reference vector 并保存至文件
    :return:
    """
    print('generating verify vector (%s)...' % model_type)
    # load data 从训练数据中获取
    f = open(os.path.join(DATA_DIR_PATH, 'data_set_%s' % model_type), 'r+b')
    raw_data = pickle.load(f)
    f.close()
    try:
        raw_data = raw_data[1].extend(raw_data[2])
    except IndexError:
        raw_data = raw_data[1]
    # train_data => (batch_amount, data_set_emg)

    data_orderby_class = {}
    for (each_label, each_data) in raw_data:
        if model_type == 'cnn':
            each_data = each_data.T
        if data_orderby_class.get(each_label) is None:
            # 需要调整长度以及转置成时序
            data_orderby_class[each_label] = [each_data]
        else:
            data_orderby_class[each_label].append(each_data)

    verifier = SiameseNetwork(train=False, model_type=model_type)
    load_model_param(verifier, 'verify_model_' + model_type)
    verifier.double()
    verify_vectors = {}
    #
    for each_sign in data_orderby_class.keys():
        verify_vectors[each_sign] = []

        for each_cap in data_orderby_class[each_sign]:
            each_cap = torch.from_numpy(np.array([each_cap])).double()
            each_cap = Variable(each_cap)
            vector = verifier(each_cap)
            vector = vector.data.float().numpy()[0]
            verify_vectors[each_sign].append(vector)
            # print('verify cost time %f' % (time.clock() - start))

    print('show image? y/n')
    is_show = input()
    if is_show == 'y':
        fig = plt.figure()
        fig.add_subplot(111, title='%s verify vectors' % model_type)

    for each_sign in verify_vectors.keys():
        verify_vector_mean = np.mean(np.array(verify_vectors[each_sign]), axis=0)
        verify_vectors[each_sign] = verify_vector_mean
        if is_show == 'y':
            plt.scatter(range(len(verify_vector_mean)), verify_vector_mean, marker='.')
            print("sign: " + str(each_sign))
            plt.pause(0.3)
    if is_show == 'y':
        plt.show()

    file_ = open(DATA_DIR_PATH + '\\reference_verify_vector_' + model_type, 'wb')
    pickle.dump(verify_vectors, file_)
    file_.close()

def load_model_param(model, model_type_name):
    for root, dirs, files in os.walk(DATA_DIR_PATH):
        for file_ in files:
            file_name_split = os.path.splitext(file_)
            if file_name_split[1] == '.pkl' and file_name_split[0].startswith(model_type_name):
                file_ = DATA_DIR_PATH + '\\' + file_
                model.load_state_dict(torch.load(file_))
                model.eval()
                return model

# GESTURES_TABLE = ['朋友', '家', '回', '去', '迟到', '交流', '联系', '客气', '再见', '劳驾', '谢谢',
#                   '对不起', '没关系', '起来', '帮助', '中国', '时间', '时差', '天', '延期', '早上', '上午',
#                   '中午', '下午', '晚上', '分钟', '小时', '昨天', '今天', '明天', '后天', '你', '什么', '想',
#                   '我', '先生', '女士', '香水', '发胶', '浴液', '手表', '钥匙', '废物', '香烟', '刀', '打火机',
#                   '乡', '吵架', '分开', '社会', '失联', '导游', '参观', '支持', '北京', '辽宁', '沈阳', '世界',
#                   '方向', '位置', '东', '西', '南', '北', '上', '下', '前', '后', '左', '右', '对面', '旁边', '中间',
#                   '这里', '那里', '很', '大家 ', '我们', '同志', '姑娘', '老', '打架', '请问', '为什么', '找',
#                   '不到', '在哪', '怎么走']
NEW_GESTURE_TABLE = []


def statistics_data(data_dir_name, show_needing_info=True):
    data_path = os.path.join(DATA_DIR_PATH, data_dir_name)
    date_list = os.listdir(data_path)
    data_stat_book = {}
    for each_sign in range(1, len(GESTURES_TABLE) + 1):
        data_stat_book[each_sign] = {
            'occ_time': 0,
            'occ_pos': [],
        }
    print('date %s' % str(date_list))

    task_list = []
    for each_date in date_list:
        path = os.path.join(data_path, each_date)
        batch_list = os.listdir(path)
        for each_batch in batch_list:
            task_list.append((path, each_batch, each_date, data_path))
    p = Pool(7)
    res = p.map(scan_data, task_list)

    for each_res in res:
        for each_sign in each_res.keys():
            data_stat_book[each_sign]['occ_time'] += each_res[each_sign]['occ_time']
            data_stat_book[each_sign]['occ_pos'] += each_res[each_sign]['occ_pos']



    sum_up = 0
    for each in sorted(data_stat_book.keys()):
        print("sign %d %s, cnt %d, occ pos %s" % (each, GESTURES_TABLE[each - 1],
                                                  data_stat_book[each]['occ_time'],
                                                  data_stat_book[each]['occ_pos']))
        sum_up += data_stat_book[each]['occ_time']
    print('sum %d' % sum_up)
    if show_needing_info:
        add_600_list = []
        add_400_list = []
        for each in data_stat_book.keys():
            if data_stat_book[each]['occ_time'] != 0:
                if data_stat_book[each]['occ_time'] < 200:
                    add_600_list.append(each)
                elif data_stat_book[each]['occ_time'] < 400:
                    add_400_list.append(each)

        print('need 600')
        for each in add_600_list:
            print(GESTURES_TABLE[each - 1], end=', ')
        print('\nneed 400')
        for each in add_400_list:
            print(GESTURES_TABLE[each - 1], end=', ')
    return data_stat_book


def scan_data(args):
    path = args[0]
    each_batch = args[1]
    each_date = args[2]
    data_path = args[3]

    data_stat_book = {}
    data_files = os.path.join(path, each_batch, 'Emg')
    data_files = os.listdir(data_files)
    for each_sign in data_files:

        each_sign = int(each_sign.split('.')[0])
        data = load_train_data(batch_num=int(each_batch),
                               sign_id=int(each_sign),
                               data_path=os.path.join(data_path, each_date))
        try:
            a = data['acc'][0][0]  # check dim
            if data_stat_book.get(each_sign) is None:
                data_stat_book[each_sign] = {
                    'occ_time': len(data['acc']),
                    'occ_pos': ["%s %s" % (each_date, each_batch)]
                }
            else:
                data_stat_book[each_sign]['occ_time'] += len(data['acc'])
                data_stat_book[each_sign]['occ_pos'].append("%s %s" % (each_date, each_batch))
        except (KeyError, IndexError):
            print(each_date, each_batch, each_sign)
    return data_stat_book



def get_gesture_label_trans_table():
    global NEW_GESTURE_TABLE
    if len(NEW_GESTURE_TABLE) == 0:
        NEW_GESTURE_TABLE = GESTURES_TABLE
    # get mapping from old gesture_table to new gesture table
    map_table = {}
    for each in range(len(NEW_GESTURE_TABLE)):
        try:
            map_table[GESTURES_TABLE.index(NEW_GESTURE_TABLE[each])] = each
        except ValueError:
            print("new add label %s" % NEW_GESTURE_TABLE[each])

    for each in range(len(GESTURES_TABLE)):
        try:
            NEW_GESTURE_TABLE.index(GESTURES_TABLE[each])
        except ValueError:
            print('removed label %s ' % GESTURES_TABLE[each])
    return map_table

def resort_data(date_list=None):
    map_table = {
        '66': '31',
        '67': '66',
        '68': '67',
        '69': '68',
        '70': '69',
    }
    data_path = os.path.join(DATA_DIR_PATH, 'collect_data_new')
    resort_path = os.path.join(DATA_DIR_PATH, 'resort_data')
    if date_list is None:
        print("resort all data?")
        res = input()
        if res == 'y':
            date_list = os.listdir(data_path)
        else:
            return
    tmp_date_list = []
    overall_date_list = os.listdir(data_path)
    for each_candidate_date in date_list:
        if each_candidate_date.endswith("*"):
            each_candidate_date = each_candidate_date.strip('*')
            for each_overall_date in overall_date_list:
                if each_overall_date.startswith(each_candidate_date):
                    tmp_date_list.append(each_overall_date)
        else:
            tmp_date_list.append(each_candidate_date)

    date_list = tmp_date_list

    for each_date in date_list:
        print("resorting date %s" % each_date)
        path = os.path.join(data_path, each_date)
        batch_list = os.listdir(path)
        for each_batch_num in range(len(batch_list)):
            data_files_path = os.path.join(path, batch_list[each_batch_num])
            data_files = os.listdir(os.path.join(data_files_path, 'Emg'))
            for each_data in data_files:
                for each_type in ['Acceleration', 'Emg', 'Gyroscope']:
                    old_path = os.path.join(data_files_path, each_type, each_data)
                    new_label = each_data
                    if map_table.get(each_data.strip('.txt')) is not None:
                        new_label = '%s.txt' % map_table[each_data.strip('.txt')]
                    target_path = os.path.join(resort_path, each_date, str(each_batch_num + 1), each_type)
                    if not os.path.exists(target_path):
                        os.makedirs(target_path)
                    new_path = os.path.join(target_path, new_label)
                    shutil.copyfile(old_path, new_path)

def merge_old_data():
    global OLD_GESTURES_TABLE
    OLD_GESTURES_TABLE = [each.strip(' ') for each in OLD_GESTURES_TABLE]
    trans_table = {}
    for each_sign in range(len(GESTURES_TABLE)):
        try:
            index = OLD_GESTURES_TABLE.index(GESTURES_TABLE[each_sign])
            trans_table[index] = each_sign
        except ValueError:
            continue

    source_dir = os.path.join(DATA_DIR_PATH, 'collected_data')
    target_batch_dir_list = []
    target_dir_path = os.path.join(DATA_DIR_PATH, 'resort_data')
    for each_date_dir in sorted(os.listdir(target_dir_path), reverse=True):
        for each_batch in sorted(os.listdir(os.path.join(target_dir_path, each_date_dir))):
            target_batch_dir_list.append((each_date_dir, each_batch))

    batch_list = os.listdir(source_dir)
    for each_batch in range(len(batch_list)):
        data_files_path = os.path.join(source_dir, batch_list[each_batch])
        data_files = os.listdir(os.path.join(data_files_path, 'Emg'))

        for each_data_cap in data_files:
            each_data_cap_label = int(each_data_cap.strip('.txt')) - 1
            if trans_table.get(each_data_cap_label) is None:
                continue

            for each_type in ['Acceleration', 'Emg', 'Gyroscope']:
                old_path = os.path.join(data_files_path, each_type, each_data_cap)
                trans_label = trans_table[each_data_cap_label]
                target_path = os.path.join(target_dir_path,
                                           target_batch_dir_list[each_batch][0],
                                           target_batch_dir_list[each_batch][1],
                                           each_type,
                                           '%d.txt' % (trans_label + 1))
                shutil.copyfile(old_path, target_path)


def main():
    # merge_old_data()

    # 从feedback文件获取数据
    # data_set = load_feed_back_data()[sign_id]

    # resort_data(['0817-*',])
    # res = statistics_data('cleaned_data')
    # sign 3 天, cnt 656, occ pos ['0810-2 12', '0810-2 13', '0810-2 14', '0810-2 21', '0810-2 23',
    # '0811-1 15',
    #  '0811-1 16', '0811-1 17', '0811-1 2', '0811-1 9', '0811-2 18', '0811-2 2', '0811-2 22',
    # '0811-2 23', '0811-2 25', '0811-2 26', '0811-2 27', '0811-2 3', '0811-2 4', '0811-2 5',
    # '0811-2 9', '0812-1 21', '0812-1 22', '0812-1 23', '0812-1 24', '0812-1 25', '0812-1 26',
    #  '0812-1 27', '0812-1 28', '0812-1 29', '0813-2 10', '0813-3 7']
    print_train_data(sign_id=3,
                     batch_num=25,
                     data_cap_type='acc',  # 数据采集类型 emg acc gyr
                     data_feat_type='poly_fit',  # 数据特征类型 zc rms arc trans(emg) poly_fit(cnn)
                     capture_date='0812-1',
                     data_path='cleaned_data',
                     for_cnn=True)  # cnn数据是128长度  db4 4层变换 普通的则是 160 db3 5

    # 输出上次处理过的数据的scale
    # print_scale('acc', 'all')
    # pickle_train_data_new()

    # 将采集数据转换为输入训练程序的数据格式
    # pickle_train_data(batch_num=87)
    # pickle_train_data_new()

    # 生成验证模型的参照系向量
    # generate_verify_vector('rnn')
    # generate_verify_vector('cnn')

    # 从recognized data history中取得数据
    # online_data = load_online_processed_data()

    # plot 原始采集的数据
    # print_raw_capture_data()

    # 从 raw data history中获得data 并处理成能够直接输入到cnn的形式
    # raw_capture_data = load_raw_capture_data()
    # online_data = process_raw_capture_data(load_raw_capture_data(), for_cnn=True)
    # plt.figure("111")
    # plt.plot(range(len(raw_capture_data['emg'])), raw_capture_data['emg'], '.-', )
    # plt.show()

    # 识别能力测试
    # cnn_recognize_test(online_data)

    # online data is a tuple(data_single, data_overall)
    # processed_data = split_online_processed_data(online_data)
    # print_processed_online_data(processed_data,
    #                             cap_type='emg',
    #                             feat_type='trans',  # arc zc rms trans  cnn_raw cnn的输入
    #                             overall=True,
    #                             block_cnt=6)


if __name__ == "__main__":
    main()
