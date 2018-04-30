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

def feature_extract(data_set, type_name, for_cnn):
    """
    特征提取 并进行必要的归一化

    acc gyr数据的三种特征量纲相差不大 且有某些维度全局的值都很相近的情况
    于是暂时去除归一化的操作 拟对只对数据变化较大，且变化范围较大于1的数据维度进行部分归一化

    emg数据照常进行各种处理

    :param data_set: 来自Load_From_File过程的返回值 一个dict
                     包含一个手语 三种采集数据类型的 多次采集过程的数据
    :param type_name: 数据采集的类型 决定nparray的长度
    :param for_cnn: 是否是为cnn模型进行特征提取 需要进行不一样的操作
    :return: 一个dict 包含这个数据采集类型的原始数据,3种特征提取后的数据,特征拼接后的特征向量
            仍保持多次采集的数据的np.array放在一个list中
            返回的数据的dict包含所有的数据 但是只有有效的字段有数据
    """
    global normalize_scale_collect
    normalize_scale_collect = []
    global standardize_scale_collect
    standardize_scale_collect = []
    if type_name == 'emg':
        return __emg_feature_extract(data_set, for_cnn)

    data_set_rms_feat = None
    data_set_zc_feat = None
    data_set_arc_feat = None
    data_set_polyfit_feat = []  # for cnn 使用多项式对间隔间的数据进行拟合 减少中间数据点
    data_set_appended_feat = []

    data_set = data_set[type_name]
    for raw_data in data_set:

        if not for_cnn:
            # 一般的特征提取过程
            # rms zc arc polyfit all
            seg_RMS_feat, seg_ZC_feat, seg_ARC_feat, seg_polyfit_data, seg_all_feat \
                = feature_extract_single(raw_data, type_name)
            if data_set_arc_feat is None:
                data_set_arc_feat = [seg_ARC_feat]
            else:
                data_set_arc_feat.append(seg_ARC_feat)

            if data_set_rms_feat is None:
                data_set_rms_feat = [seg_RMS_feat]
            else:
                data_set_rms_feat.append(seg_RMS_feat)

            if data_set_zc_feat is None:
                data_set_zc_feat = [seg_ZC_feat]
            else:
                data_set_zc_feat.append(seg_ZC_feat)
            data_set_polyfit_feat.append(seg_polyfit_data)

        else:
            # cnn的特征提取过程 只使用曲线拟合特征
            seg_polyfit_feat = feature_extract_single_polyfit(raw_data, 2)
            # cnn 数据的归一化
            if type_name == 'gyr':
                threshold = 25
                default_scale = 200
            else:
                threshold = 0.3
                default_scale = 5
            seg_polyfit_feat = normalize(seg_polyfit_feat, threshold, default_scale)

            # 多项式拟合后切割
            seg_polyfit_feat = seg_polyfit_feat[8:-8, :]
            # 给CNN喂128的片段短数据  拟合压缩前是

            data_set_polyfit_feat.append(seg_polyfit_feat)
            seg_all_feat = seg_polyfit_feat

        data_set_appended_feat.append(seg_all_feat)

    return {
        'type_name': type_name,
        'raw': data_set,
        'arc': data_set_arc_feat,
        'rms': data_set_rms_feat,
        'zc': data_set_zc_feat,
        'poly_fit': data_set_polyfit_feat,
        'append_all': data_set_appended_feat
    }

def feature_extract_single_polyfit(data, compress):
    seg_poly_fit = None
    start_ptr = 0
    end_ptr = 16
    while end_ptr <= len(data):
        window_data = data[start_ptr:end_ptr, :]
        window_extract_data = None
        x = np.arange(0, 16, 1)
        y = window_data
        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
        # 0   2   4   6   8   10    11    14
        poly_args = np.polyfit(x, y, 3)
        for each_channel in range(3):
            dots_in_channel = None
            window_poly = np.poly1d(poly_args[:, each_channel])
            for dot in np.arange(0, 16, compress):
                # assemble each dot's each channel
                if dots_in_channel is None:
                    dots_in_channel = window_poly(dot)
                else:
                    dots_in_channel = np.vstack((dots_in_channel, window_poly(dot)))
            # assemble each window's each channel data
            if window_extract_data is None:
                window_extract_data = dots_in_channel
            else:
                window_extract_data = np.hstack((window_extract_data, dots_in_channel))

        # assemble each window data
        if seg_poly_fit is None:
            seg_poly_fit = window_extract_data
        else:
            seg_poly_fit = np.vstack((seg_poly_fit, window_extract_data))
        start_ptr += 16
        end_ptr += 16

    return seg_poly_fit

def feature_extract_single(polyfit_data, type_name):
    # 对曲线拟合后的数据进行特征提取 效果更好
    polyfit_data = feature_extract_single_polyfit(polyfit_data, 1)
    window_amount = len(polyfit_data) / WINDOW_SIZE
    windows_data = np.vsplit(polyfit_data, window_amount)
    win_index = 0
    seg_ARC_feat = None
    seg_RMS_feat = None
    seg_ZC_feat = None

    features = [seg_RMS_feat, seg_ZC_feat, seg_ARC_feat, ]
    for Win_Data in windows_data:
        # 依次处理每个window的数据
        win_RMS_feat = np.sqrt(np.mean(np.square(Win_Data), axis=0))
        Win_Data1 = np.vstack((Win_Data[1:, :], np.zeros((1, TYPE_LEN[type_name]))))
        win_ZC_feat = np.sum(np.sign(-np.sign(Win_Data) * np.sign(Win_Data1) + 1), axis=0) - 1
        win_ARC_feat = np.apply_along_axis(ARC, 0, Win_Data)
        # arc 只要后面几个系数的数据
        win_ARC_feat = win_ARC_feat.ravel()[-5:, ]
        # 将每个window特征提取的数据用vstack叠起来
        win_index += 1
        each_feat_type_data = (win_RMS_feat, win_ZC_feat, win_ARC_feat)
        for each in range(len(each_feat_type_data)):
            if features[each] is None:
                features[each] = each_feat_type_data[each]
            else:
                features[each] = np.vstack((features[each], each_feat_type_data[each]))
    #  根据各种数据 各种特征值规律的不同
    #  进行归一化需要设置不同的归一化阈值以及默认scale
    if type_name == 'gyr':
        threshold_list = [15, 1, 20]
        default_scale = [200, 3, 100]
    else:
        threshold_list = [0.15, 1, 0.5]
        default_scale = [2.5, 3, 1.2]
    for each in range(len(threshold_list)):
        features[each] = normalize(features[each],
                                   threshold_list[each],
                                   default_scale[each])

    # 曲线拟合 a + bx + cx^2 + dx^3
    # (x_rms, y_rms, z_rms, x_zc, y_zc, z_zc, x_c, y_c, z_c, x_d, y_d ,z_d)
    seg_all_feat = np.hstack(tuple(features))
    features.append(polyfit_data)
    features.append(seg_all_feat)
    # rms zc arc polyfit all
    return tuple(features)

def ARC(Win_Data):
    Len_Data = len(Win_Data)
    # AR_coefficient = []
    AR_coefficient = np.polyfit(range(Len_Data), Win_Data, 3)
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
def emg_feature_extract(data_set, for_cnn):
    return __emg_feature_extract(data_set, for_cnn)['trans']

def __emg_feature_extract(data_set, for_cnn):
    """
    特征提取
    :param data_set: 来自Load_From_File过程的返回值 一个dict
                     包含一个手语 三种采集数据类型的 多次采集过程的数据
    :return: 一个dict 包含这个数据采集类型的原始数据,3种特征提取后的数据,特征拼接后的特征向量
            仍保持多次采集的数据放在一起
    """
    if for_cnn:
        data_set = [each[16:144, :] for each in data_set['emg']]
    else:
        data_set = data_set['emg']

    data_trans = emg_wave_trans(data_set)
    if for_cnn:
        data_trans = expand_emg_data(data_trans)
    return {
        'type_name': 'emg',
        'raw': data_set,
        'trans': data_trans,
        'append_all': data_trans,
    }

def wavelet_trans(data):
    data = np.array(data).T  # 转换为 通道 - 时序
    data = pywt.threshold(data, 18, 'hard')  # 阈值滤波
    try:
        data = pywt.wavedec(data, wavelet='db3', level=5)  # 小波变换
    except ValueError:
        data = pywt.wavedec(data, wavelet='db2', level=5)
    data = np.vstack((data[0].T, np.zeros(8))).T
    # 转换为 时序-通道 追加一个零点在转换回 通道-时序
    data = pywt.threshold(data, 12, 'hard')  # 再次阈值滤波
    data = data.T
    data = normalize(data, threshold=1, default_scale=60)  # 转换为 时序-通道 以时序轴 对每个通道进行normalize
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

def expand_emg_data(data):
    expnded = []
    for each_data in data:
        each_data_expand = expand_emg_data_single(each_data)
        expnded.append(np.array(each_data_expand))
    return expnded

def expand_emg_data_single(data):
    expanded_data = None
    for each_dot in range(len(data)):
        if each_dot % 2 != 0:
            continue  # 只对偶数点进行左右扩展
        if each_dot - 1 < 0:
            left_val = data[each_dot]
        else:
            left_val = data[each_dot - 1]

        if each_dot + 1 >= len(data):
            right_val = data[each_dot]
        else:
            right_val = data[each_dot + 1]

        center_val = data[each_dot]
        x = np.arange(0, 2, 1)
        y = np.array([left_val, center_val])
        left_line_args = np.polyfit(x, y, 1)
        y = np.array([center_val, right_val])
        right_line_args = np.polyfit(x, y, 1)

        dot_expanded_data = None
        for each_channel in range(8):
            each_channel_dot_expanded = None

            poly_left = np.poly1d(left_line_args[:, each_channel])
            expand_range = []
            for i in range(8):
                expand_range.append(0.125 * i)

            for dot in expand_range:
                if each_channel_dot_expanded is None:
                    each_channel_dot_expanded = np.array(poly_left(dot))
                else:
                    each_channel_dot_expanded = np.vstack((each_channel_dot_expanded, poly_left(dot)))

            poly_right = np.poly1d(right_line_args[:, each_channel])
            for dot in expand_range:
                if each_channel_dot_expanded is None:
                    each_channel_dot_expanded = np.array(poly_right(dot))
                else:
                    each_channel_dot_expanded = np.vstack((each_channel_dot_expanded, poly_right(dot)))

            if dot_expanded_data is None:
                dot_expanded_data = each_channel_dot_expanded
            else:
                dot_expanded_data = np.hstack((dot_expanded_data, each_channel_dot_expanded))

        if expanded_data is None:
            expanded_data = dot_expanded_data
        else:
            expanded_data = np.vstack((expanded_data, dot_expanded_data))

    #  data padding
    # expanded_data = np.vstack((expanded_data[0,:], expanded_data))
    # expanded_data = np.vstack((expanded_data, expanded_data[-1,:]))

    return expanded_data

# data scaling
normalize_scaler = preprocessing.MinMaxScaler()
normalize_scale_collect = []

def normalize(data, threshold, default_scale):
    normalize_scaler.fit(data)
    scale_adjust(threshold, default_scale)
    data = normalize_scaler.transform(data)
    # 记录每次的scale情况
    # curr_scale = [each for each in normalize_scaler.scale_]
    # normalize_scale_collect.append(curr_scale)
    return data

"""
maxmin scale = (val - min) / (max - min) 
即数据在最大值最小值直接的比例
scale值阈值的设置是根据 scikit MinMax的的处理方法
scale数组中实际存储的是最大值减最小值的倒数  值越大 说明数据波动越小
如果scale时最大最小值相差很小 则不进行min max 的缩放scale 避免放大噪声
min 数组中存的是最小值 乘以scale 数组的值 相当于数据基准偏移量
数据一般都有一个小偏移量 所以数据最好都进行一下偏移修正
在不进行scale时 偏移量应还原成数据自身的偏移量 所以做之前乘法的逆运算 获取原始偏移量

由于RNN CNN scale数据的特性不同 决定是否进行scale的阈值也不同 
"""

def scale_adjust(threshold, default_scale):
    """
    根据scale的情况判断是否需要进行scale
    scale的大小是由这个数据的max - min的得出 如果相差不大 就不进行scale
    通过修改scale和min的值使其失去scale的作用
    @:parameter threshold 过滤阈值 当最大最小值之差小于这个阈值 不进行归一化
    note: scale 的大小是max - min 的倒数
    """
    threshold = 1 / threshold
    default_scale = 1 / default_scale
    curr_scale = normalize_scaler.scale_
    curr_min = normalize_scaler.min_
    for each_val in range(len(curr_scale)):
        if curr_scale[each_val] > threshold:
            # 当最大最小值不满足一般数据规律时 设置为默认归一化的scale
            curr_min[each_val] = curr_min[each_val] * default_scale / curr_scale[each_val]
            curr_scale[each_val] = default_scale

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

