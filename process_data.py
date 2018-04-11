# coding:utf-8
# py3
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pywt
from matplotlib import font_manager
from matplotlib.legend_handler import HandlerLine2D
from sklearn import preprocessing

Width_EMG = 9
Width_ACC = 3
Width_GYR = 3
LENGTH = 160
WINDOW_SIZE = 16
EMG_WINDOW_SIZE = 3
SIGN_COUNT = 14
FEATURE_LENGTH = 44
DATA_DIR_PATH = os.getcwd() + '\\data'

myfont = font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
mpl.rcParams['axes.unicode_minus'] = False

TYPE_LEN = {
    'acc': 3,
    'gyr': 3,
    'emg': 8
}

CAP_TYPE_LIST = ['acc', 'gyr', 'emg']  # 直接在这里修改可去除emg
# CAP_TYPE_LIST = ['acc', 'gyr', 'emg']
GESTURES_TABLE = ['肉 ', '鸡蛋 ', '喜欢 ', '您好 ', '你 ', '什么 ', '想 ', '我 ', '很 ', '吃 ',
                  '老师 ', '发烧 ', '谢谢 ', '空手语', '大家', '支持', '我们', '创新', '医生', '交流',
                  '团队', '帮助', '聋哑人', '请', ]

def file2matrix(filename, data_col_num):
    del_sign = '()[]'
    separator = ','
    try:
        fr = open(filename, 'r')
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

'''
读取数据 从文件中读取数据
数据格式:
{
    'acc': [ 每次手语的160个数据组成的nparrayy 一共20个 一次采集一个ndarray] 
            ndarray中每个数据的形式以 [x, y, z] 排列
    'gyr': [ 同上 ] 
    'emg': [ 形式同上 但是有8个维度]
}

py2 py3 pickle 不能通用

'''

def load_data(sign_id, batch_num):
    """
    从采集文件夹中读取采集数据 并对数据进行裁剪
    :param sign_id: 需要取得的手语id
    :param batch_num: 需要取得的手语所在的batch数
    :return: 返回dict  包含这个手语的三种采集数据, 多次采集的数据矩阵的list
    """
    # Load and return data
    # initialization
    Path = DATA_DIR_PATH
    file_num = sign_id
    file_emg = Path + '\\' + str(batch_num) + '\\Emg\\' + str(file_num) + '.txt'
    data_emg = file2matrix(file_emg, Width_EMG)
    file_acc = Path + '\\' + str(batch_num) + '\\Acceleration\\' + str(file_num) + '.txt'
    data_acc = file2matrix(file_acc, Width_ACC)
    file_gyr = Path + '\\' + str(batch_num) + '\\Gyroscope\\' + str(file_num) + '.txt'
    data_gyr = file2matrix(file_gyr, Width_GYR)

    processed_data_emg = []
    processed_data_acc = []
    processed_data_gyr = []
    if len(data_emg) != 0:
        capture_tag_list = list(data_emg[:, -1])
        capture_length_book = {}
        for i in capture_tag_list:
            capture_length_book[i] = capture_length_book.get(i, 0) + 1
        Index = 0
        capture_times = len(capture_length_book.keys())
        capture_times = capture_times if capture_times < 20 else 21
        for i in range(1, capture_times):
            resize_data_emg = length_adjust(data_emg[Index:Index + capture_length_book[i], 0:8])
            processed_data_emg.append(resize_data_emg)  # init
            # processed_data_emg.append(standardize(resize_data_emg))
            # processed_data_emg.append(normalize(resize_data_emg))

            resize_data_acc = length_adjust(data_acc[Index:Index + capture_length_book[i], :])
            processed_data_acc.append(resize_data_acc)
            # processed_data_acc.append(standardize(resize_data_acc))
            # processed_data_acc.append(normalize(resize_data_acc))

            resize_data_gyr = length_adjust(data_gyr[Index:Index + capture_length_book[i], :])
            processed_data_gyr.append(resize_data_gyr)
            # processed_data_gyr.append(standardize(resize_data_gyr))
            # processed_data_gyr.append(normalize(resize_data_gyr))

            Index += capture_length_book[i]
        print('Load done , batch num: %d, sign id: %d, ' % (batch_num, sign_id,))

    return {
        'emg': processed_data_emg,  # 包含这个手语多次采集的数据矩阵的list
        'acc': processed_data_acc,
        'gyr': processed_data_gyr,
    }

def length_adjust(A):
    tail_len = len(A) - LENGTH
    if tail_len < 0:
        print('Length Error')
        A1 = A
    else:
        # 前后各去掉多出来长度的一半
        End = len(A) - tail_len / 2
        Begin = tail_len / 2
        A1 = A[int(Begin):int(End), :]
    return A1

def trans_data_to_time_seqs(data_set):
    return data_set.T

normalize_scaler = preprocessing.MinMaxScaler()
normalize_scale_collect = []
def normalize(data):
    normalize_scaler.fit(data)
    scale_adjust()
    data = normalize_scaler.transform(data)
    # 记录每次的scale情况
    # curr_scale = [each for each in normalize_scaler.scale_]
    # normalize_scale_collect.append(curr_scale)
    return data

standardize_scaler = preprocessing.StandardScaler()
def standardize(data):
    standardize_scaler.fit(data)
    data = standardize_scaler.transform(data)
    return data

def scale_adjust():
    """
    根据scale的情况判断是否需要进行scale
    scale的大小是由这个数据的max - min的得出 如果相差不大 就不进行scale
    通过修改scale和min的值使其失去scale的作用

    note: scale 的大小时max - min 的倒数
    """
    curr_scale = normalize_scaler.scale_
    curr_min = normalize_scaler.min_

    for each_val in range(len(curr_scale)):
        if curr_scale[each_val] > 1:
            curr_scale[each_val] = 1
        if abs(curr_min[each_val]) < 2:
            curr_min[each_val] = 0



def print_plot(data_set, data_cap_type, data_feat_type):
    for dimension in range(TYPE_LEN[data_cap_type]):
        fig_ = plt.figure()
        fig_.add_subplot(111, title='%s %s dim%s' % (data_feat_type, data_cap_type, str(dimension + 1)))
        capture_times = len(data_set[data_feat_type])
        capture_times = capture_times if capture_times < 20 else 20
        # 最多只绘制十次采集的数据 （太多了会看不清）

        handle_lines_map = {}
        for capture_num in range(0, capture_times):
            single_capture_data = trans_data_to_time_seqs(data_set[data_feat_type][capture_num])
            data = single_capture_data[dimension]
            l = plt.plot(range(len(data)), data, '.-', label='cap %d' % capture_num, )
            handle_lines_map[l[0]] = HandlerLine2D(numpoints=1)
            plt.pause(0.01)
        plt.legend(handler_map=handle_lines_map)
        plt.pause(0.008)
    plt.show()

def pickle_to_file(batch_num, feedback_data=None):
    """
    从采集生成的文件夹中读取数据 存为python对象

    采用追加的方式 当采集文件夹数大于当前数据对象batch最大值时 进行数据的追加
    :param batch_num: 当前需要提取的采集数据文件夹数
    :param online_mode: 是否生成online识别的短数据
    :param feedback_data 是否将之前feedback数据纳入训练集
    :return: None  过程函数
    """
    data_set_file_name = 'data_set'

    try:
        file = open(DATA_DIR_PATH + '\\' + data_set_file_name, 'r+b')
        train_data = pickle.load(file)
        file.close()
    except IOError:
        train_data = (0, [])

    # (batch_amount, data_set_emg)
    # 元组的形式保存数据数据对象
    # [0] 是当前数据对象所包含的batch数
    # [1] 是数据对象的实例
    # 每次都从上一次读取的数据集对象的基础上进行更新
    curr_batch_num = train_data[0]
    train_data = train_data[1]

    for each_batch in range(curr_batch_num + 1, batch_num + 1):
        for each_sign in range(1, len(GESTURES_TABLE) + 1):
            # 一个手势一个手势的读入数据
            raw_data_set = load_data(batch_num=each_batch, sign_id=each_sign)
            extracted_data_set = []
            # 根据数据采集种类 提取特征
            for each_cap_type in CAP_TYPE_LIST:
                extracted_data_set.append(feature_extract(raw_data_set, each_cap_type)['append_all'])

            # 拼接特征 使其满足RNN的输入要求
            batch_list = append_feature_vector(extracted_data_set)
            for each_data_mat in batch_list:
                train_data.append((each_sign, each_data_mat))
    curr_data_set_cont = batch_num

    if feedback_data is not None:
        train_data_from_feedback = []
        for sign_id in range(len(feedback_data)):
            extracted_data_set = []
            sign_raw_data = feedback_data[sign_id]

            if len(sign_raw_data) == 0:
                continue
            for each_cap_type in CAP_TYPE_LIST:
                extracted_data_set.append(feature_extract(sign_raw_data, each_cap_type)['append_all'])

            batch_list = append_feature_vector(extracted_data_set)
            for each_data_mat in batch_list:
                train_data_from_feedback.append((sign_id, each_data_mat))
        # 将feedback的结果追加在后面
        train_data = (curr_data_set_cont, train_data, train_data_from_feedback)
    else:
        train_data = (curr_data_set_cont, train_data)

    # train_data format:
    # (batch_amount, data_set_emg)
    file = open(DATA_DIR_PATH + '\\' + data_set_file_name, 'w+b')
    pickle.dump(train_data, file)
    file.close()



def emg_feature_extract(data_set):
    return __emg_feature_extract(data_set)['trans']

def __emg_feature_extract(data_set):
    """
    特征提取
    :param data_set: 来自Load_From_File过程的返回值 一个dict
                     包含一个手语 三种采集数据类型的 多次采集过程的数据
    :param type_name: 数据采集的类型 决定nparray的长度
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


'''
提取一个手势的一个batch的某一信号种类的全部数据
数据形式保存不变 只改变数值和每次采集ndarray的长度
（特征提取会改变数据的数量）
'''

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
    data = length_adjust(data)
    window_amount = len(data) / WINDOW_SIZE
    # windows_data = data.reshape(window_amount, WINDOW_SIZE, TYPE_LEN[type_name])
    windows_data = np.vsplit(data[0:160], window_amount)
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
        if is_first:
            is_first = False
            seg_all_feat = All_Seg_Feat
        else:
            seg_all_feat = np.vstack((seg_all_feat, All_Seg_Feat))

    seg_all_feat = normalize(seg_all_feat)
    # seg_all_feat = np.abs(seg_all_feat)
    seg_RMS_feat = seg_all_feat[:, 0:3]
    seg_ZC_feat = seg_all_feat[:, 3:6]
    seg_ARC_feat = seg_all_feat[:, 6:18]
    seg_ARC_feat = np.hsplit(seg_ARC_feat, 4)
    seg_ARC_feat = np.vstack(tuple(seg_ARC_feat))
    return seg_ARC_feat, seg_RMS_feat, seg_ZC_feat, seg_all_feat

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


def load_from_file_feed_back():
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

def wavelet_trans(data):
    data = np.array(data).T  # 转换为 通道 - 时序
    data = pywt.threshold(data, 30, mode='hard')  # 阈值滤波
    data = pywt.wavedec(data, wavelet='db3', level=5)  # 小波变换
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

def get_feat_std_scales():
    scales = [each for each in standardize_scale_collect]
    return scales

def print_scale(cap_type, scale_feat_name):
    """
    获取特征提取后 向量归一化使用的scale
    通过设置手语id的范围 和 batch id的范围，
    来在不同散点图显示 各种手语在不同batch的scale情况
    并能计算其平均值和中位数 保存至文件 在 在线识别的时候作为参照和默认归一化向量使用
    :param cap_type:  数据采集的类型
    :param scale_feat_name: 数据的特征类型 rms zc arc all （emg 只用all就好）
    """
    scale_list = []
    fig = plt.figure()
    fig.add_subplot(111, title='%s %s scales' % (cap_type, scale_feat_name))

    # 设置batch的range
    batch_range = list(range(1, 20))
    batch_range.extend(list(range(40, 75)))

    # 扫描各个batch的各个手语
    for batch_id in batch_range:  # batch id range
        for sign_id in range(1, 25):  # sign id 或者是 batch id 的 range
            if sign_id == 14:
                continue

            # scale_list = []
            data_set = load_data(sign_id=sign_id, batch_num=batch_id)  # 循环从采集文件获取数据
            feature_extract(data_set, cap_type)
            if cap_type != 'emg':
                scales = get_feat_norm_scales()
                if scale_feat_name == 'arc':
                    scale_list.extend(scales['all'])
                    for each_val in range(len(scale_list)):
                        scale_list[each_val] = scale_list[each_val][0:12]
                else:
                    scale_list.extend(scales[scale_feat_name])
            else:
                # emg 时 scale只有一个
                scale_list.extend([each for each in normalize_scale_collect])
            # 显示单个手语的scale的平均值
            # scale_list = np.mean(scale_list, axis=0)
            # plt.scatter(np.array(range(len(scale_list))), scale_list, marker='.')

        # 输出每个手语的scale
        # for each in scale_list:
        #     plt.scatter(np.array(range(len(each))), each, marker='.')

        # 一个手语一个散点图
        # fig = plt.figure()
        # fig.add_subplot(111, title='sign id: %d %s scales' % (i, scale_feat_name))

    # 计算所有手语的scale 平均值 和 中位数
    scale_list_mean = np.mean(scale_list, axis=0)
    scale_list_median = np.median(scale_list, axis=0)

    # 将scale存入文件
    file_ = open(DATA_DIR_PATH + '\\scale_mean', 'w+b')
    pickle.dump(scale_list_mean, file_, protocol=2)
    file_.close()
    file_ = open(DATA_DIR_PATH + '\\scale_median', 'w+b')
    pickle.dump(scale_list_median, file_, protocol=2)
    file_.close()

    # 绘制出scale平均值和中位数
    # fig = plt.figure()
    # fig.add_subplot(111, title='mean scale')
    # plt.scatter(np.array(range(len(scale_list_mean))), scale_list_mean, marker='.', )
    #
    # fig_ = plt.figure()
    # fig_.add_subplot(111, title='median scale')
    # plt.scatter(np.array(range(len(scale_list_median))), scale_list_median, marker='.', )

    # plt.show()

def print_data_plot(sign_id, batch_num, data_cap_type, data_feat_type):
    data_set = load_data(sign_id=sign_id, batch_num=batch_num)  # 从采集文件获取数据
    data_set = feature_extract(data_set, data_cap_type)
    print_plot(data_set, data_cap_type, data_feat_type)

def main():
    # 从feedback文件获取数据
    # data_set = load_from_file_feed_back()[sign_id]

    print_data_plot(sign_id=10,
                    batch_num=24,
                    data_cap_type='gyr',  # 数据特征类型 zc rms arc trans(emg)
                    data_feat_type='arc')  # 数据采集类型 emg acc gyr

    # print_scale('acc', 'all')
    #
    # 将采集数据转换为训练数据
    # pickle_to_file(batch_num=91, online_mode=True)
    # pickle_to_file(batch_num=91)

if __name__ == "__main__":
    main()
