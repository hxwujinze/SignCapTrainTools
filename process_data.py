# coding:utf-8

import numpy as np
import pywt
from sklearn import preprocessing


WINDOW_SIZE = 16
TYPE_LEN = {
    'acc': 3,
    'gyr': 3,
    'emg': 8
}

'''
提取一个手势的一个batch的某一信号种类的全部数据
数据形式保存不变 只改变数值和每次采集ndarray的长度
（特征提取会改变数据的数量）
'''

# data process func for online

def feature_extract(data_set, type_name):
    """
    特征提取 并进行必要的归一化

    acc gyr数据的三种特征量纲相差不大 且有某些维度全局的值都很相近的情况
    于是暂时去除归一化的操作 拟对只对数据变化较大，且变化范围较大于1的数据维度进行部分归一化

    emg数据照常进行各种处理

    :param data_set: 来自Load_From_File过程的返回值 一个dict
                     包含一个手语 三种采集数据类型的 多次采集过程的数据
    :param type_name: 数据采集的类型 决定nparray的长度
    :return: 一个dict 包含这个数据采集类型的原始数据,3种特征提取后的数据,特征拼接后的特征向量
            仍保持多次采集的数据放在一起
    """
    global normalize_scale_collect
    normalize_scale_collect = []
    global standardize_scale_collect
    standardize_scale_collect = []
    if type_name == 'emg':
        return __emg_feature_extract(data_set)
    data_set_rms_feat = []
    data_set_zc_feat = []
    data_set_arc_feat = []
    data_set_append_feat = []
    data_set = data_set[type_name]
    for data in data_set:
        seg_ARC_feat, seg_RMS_feat, seg_ZC_feat, seg_all_feat \
            = feature_extract_single(data, type_name)
        data_set_arc_feat.append(seg_ARC_feat)
        data_set_rms_feat.append(seg_RMS_feat)
        data_set_zc_feat.append(seg_ZC_feat)
        data_set_append_feat.append(seg_all_feat)
    return {
        'type_name': type_name,
        'raw': data_set,
        'arc': data_set_arc_feat,
        'rms': data_set_rms_feat,
        'zc': data_set_zc_feat,
        'append_all': data_set_append_feat
    }

def feature_extract_single(data, type_name):
    window_amount = len(data) / WINDOW_SIZE
    # windows_data = data.reshape(window_amount, WINDOW_SIZE, TYPE_LEN[type_name])
    windows_data = np.vsplit(data, window_amount)
    win_index = 0
    is_first = True
    seg_all_feat = []
    for Win_Data in windows_data:
        # 依次处理每个window的数据
        win_RMS_feat = np.sqrt(np.mean(np.square(Win_Data), axis=0))
        Win_Data1 = np.vstack((Win_Data[1:, :], np.zeros((1, TYPE_LEN[type_name]))))
        win_ZC_feat = np.sum(np.sign(-np.sign(Win_Data) * np.sign(Win_Data1) + 1), axis=0) - 1
        win_ARC_feat = np.apply_along_axis(ARC, 0, Win_Data)
        # 将每个window特征提取的数据用vstack叠起来
        win_index += 1
        # 将三种特征拼接成一个长向量
        # 层叠 遍历展开
        Seg_Feat = np.vstack((win_RMS_feat, win_ZC_feat, win_ARC_feat))
        All_Seg_Feat = Seg_Feat.ravel()
        # (x_rms, y_rms, z_rms, x_zc, y_zc, z_zc, x_a, y_a, z_a, x_b, y_b, y_c, z_a, z_b, z_c)
        if is_first:
            is_first = False
            seg_all_feat = All_Seg_Feat
        else:
            seg_all_feat = np.vstack((seg_all_feat, All_Seg_Feat))

    seg_all_feat = normalize(seg_all_feat)
    # seg_all_feat = np.abs(seg_all_feat)
    seg_RMS_feat = seg_all_feat[:, 0:3]
    seg_ZC_feat = seg_all_feat[:, 3:6]
    seg_ARC_feat = seg_all_feat[:, 6:]
    try:
        seg_ARC_feat = np.hsplit(seg_ARC_feat, 3)
    except ValueError:
        print(seg_ARC_feat)
    seg_ARC_feat = np.vstack(tuple(seg_ARC_feat))
    return seg_ARC_feat, seg_RMS_feat, seg_ZC_feat, seg_all_feat

def ARC(Win_Data):
    Len_Data = len(Win_Data)
    # AR_coefficient = []
    AR_coefficient = np.polyfit(range(Len_Data), Win_Data, 2)
    return AR_coefficient

def append_feature_vector(data_set):
    """
    拼接三种数据采集类型的特征数据成一个大向量
    :param data_set: 第一维存储三种采集类型数据集的list
                     第二维是这个类型数据三种特征拼接后 每次采集获得的数据矩阵
    :return:
    """

    batch_list = []
    # 每种采集类型下有多个数据
    for i in range(len(data_set[0])):
        # 取出每个采集类型的数据列中的每个数据进行拼接
        batch_mat = append_single_data_feature(acc_data=data_set[0][i],
                                               gyr_data=data_set[1][i],
                                               emg_data=data_set[2][i], )
        batch_list.append(batch_mat)
    return batch_list

def append_single_data_feature(acc_data, gyr_data, emg_data):
    batch_mat = np.zeros(len(acc_data))
    is_first = True
    for each_window in range(len(acc_data)):
        # 针对每个识别window
        # 把这一次采集的三种数据采集类型进行拼接
        line = np.append(acc_data[each_window], gyr_data[each_window])
        line = np.append(line, emg_data[each_window])
        if is_first:
            is_first = False
            batch_mat = line
        else:
            batch_mat = np.vstack((batch_mat, line))
    return batch_mat

# emg data_process
def emg_feature_extract(data_set):
    return __emg_feature_extract(data_set)['trans']

def __emg_feature_extract(data_set):
    """
    特征提取
    :param data_set: 来自Load_From_File过程的返回值 一个dict
                     包含一个手语 三种采集数据类型的 多次采集过程的数据
    :return: 一个dict 包含这个数据采集类型的原始数据,3种特征提取后的数据,特征拼接后的特征向量
            仍保持多次采集的数据放在一起
    """
    data_trans = emg_wave_trans(data_set['emg'])
    return {
        'type_name': 'emg',
        'raw': data_set['emg'],
        'trans': data_trans,
        'append_all': data_trans,
    }

def wavelet_trans(data):
    data = np.array(data).T  # 转换为 通道 - 时序
    data = pywt.threshold(data, 30, mode='hard')  # 阈值滤波
    try:
        data = pywt.wavedec(data, wavelet='db3', level=5)  # 小波变换
    except ValueError:
        data = pywt.wavedec(data, wavelet='db3', level=4)
    data = np.vstack((data[0].T, np.zeros(8))).T
    # 转换为 时序-通道 追加一个零点在转换回 通道-时序
    data = pywt.threshold(data, 20, mode='hard')  # 再次阈值滤波
    data = data.T
    data = normalize(data)  # 转换为 时序-通道 以时序轴 对每个通道进行normalize
    data = eliminate_zero_shift(data)  # 消除零点漂移
    data = np.abs(data)  # 反转
    return data  # 转换为 时序-通道 便于rnn输入

def emg_wave_trans(data_set):
    res_list = []
    for each_cap in data_set:
        cap = wavelet_trans(each_cap)
        res_list.append(cap)
    return res_list

def eliminate_zero_shift(data):
    zero_point = []
    for each_chanel in range(len(data[0])):
        count_dic = {}
        for each_cap in range(len(data)):
            if count_dic.get(data[each_cap][each_chanel]) is None:
                count_dic[data[each_cap][each_chanel]] = 1
            else:
                count_dic[data[each_cap][each_chanel]] += 1
        max_occr = 0
        value = 0
        for each_value in count_dic.keys():
            if max_occr < count_dic[each_value]:
                max_occr = count_dic[each_value]
                value = each_value
        if max_occr > 1:
            zero_point.append(value)
        else:
            zero_point.append(0)
    zero_point = np.array(zero_point)
    data -= zero_point
    return data

# data scaling
normalize_scaler = preprocessing.MinMaxScaler()
normalize_scale_collect = []

def normalize(data):
    normalize_scaler.fit(data)
    scale_adjust()
    data = normalize_scaler.transform(data)
    # 记录每次的scale情况
    curr_scale = [each for each in normalize_scaler.scale_]
    normalize_scale_collect.append(curr_scale)
    return data

def scale_adjust():
    """
    根据scale的情况判断是否需要进行scale
    scale的大小是由这个数据的max - min的得出 如果相差不大 就不进行scale
    通过修改scale和min的值使其失去scale的作用

    note: scale 的大小是max - min 的倒数
    """
    curr_scale = normalize_scaler.scale_
    curr_min = normalize_scaler.min_

    for each_val in range(len(curr_scale)):
        if curr_scale[each_val] > 1:
            curr_scale[each_val] = 1
            curr_min[each_val] = 0
        # if abs(curr_min[each_val]) < 50:
        # curr_min[each_val] = 0

def get_feat_norm_scales():
    # 0 ARC 1 RMS 2 ZC 3 ALL
    feat_name = ['arc', 'rms', 'zc', 'all']
    scales = {
        'arc': [],
        'rms': [],
        'zc': [],
        'all': [],
    }
    for each in normalize_scale_collect:
        feat_no = normalize_scale_collect.index(each) % 4
        scales[feat_name[feat_no]].append(each)
    return scales


