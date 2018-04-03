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

def Load_ALL_Data(sign_id, batch_num):
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
    print('Load ALL Data')
    file_emg = Path + '\\' + str(batch_num) + '\\Emg\\' + str(file_num) + '.txt'
    data_emg = file2matrix(file_emg, Width_EMG)
    file_acc = Path + '\\' + str(batch_num) + '\\Acceleration\\' + str(file_num) + '.txt'
    data_acc = file2matrix(file_acc, Width_ACC)
    file_gyr = Path + '\\' + str(batch_num) + '\\Gyroscope\\' + str(file_num) + '.txt'
    data_gyr = file2matrix(file_gyr, Width_GYR)
    print('Load done')
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
            resize_data_emg = Length_Adjust(data_emg[Index:Index + capture_length_book[i], 0:8])
            processed_data_emg.append(resize_data_emg)  # init
            # processed_data_emg.append(standardize(resize_data_emg))
            # processed_data_emg.append(normalize(resize_data_emg))

            resize_data_acc = Length_Adjust(data_acc[Index:Index + capture_length_book[i], :])
            processed_data_acc.append(resize_data_acc)
            # processed_data_acc.append(standardize(resize_data_acc))
            # processed_data_acc.append(normalize(resize_data_acc))

            resize_data_gyr = Length_Adjust(data_gyr[Index:Index + capture_length_book[i], :])
            processed_data_gyr.append(resize_data_gyr)
            # processed_data_gyr.append(standardize(resize_data_gyr))
            # processed_data_gyr.append(normalize(resize_data_gyr))

            Index += capture_length_book[i]

        print("data resized")
    return {
        'emg': processed_data_emg,  # 包含这个手语多次采集的数据矩阵的list
        'acc': processed_data_acc,
        'gyr': processed_data_gyr,
    }

def Length_Adjust(A):
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

def normalize(data):
    normalize_scaler.fit(data)
    data = normalize_scaler.transform(data)
    return data

standardize_scaler = preprocessing.StandardScaler()

def standardize(data):
    standardize_scaler.fit(data)
    data = standardize_scaler.transform(data)
    return data



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
        plt.pause(0.01)
    plt.show()

def pickle_to_file(batch_num, feedback_data=None):
    """
    从采集生成的文件夹中读取数据 存为python对象

    采用追加的方式 当采集文件夹数大于当前数据对象batch最大值时 进行数据的追加
    :param batch_num: 当前需要提取的采集数据文件夹数
    :param feedback_data 是否将之前feedback数据纳入训练集
    :return: None  过程函数
    """
    try:
        file = open(DATA_DIR_PATH + '\\data_set', 'r+b')
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
            raw_data_set = Load_ALL_Data(batch_num=each_batch, sign_id=each_sign)
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

    # (batch_amount, data_set_emg)
    file = open(DATA_DIR_PATH + '\\data_set', 'w+b')
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

#
# def emg_feature_extract_single(data):
#     data = Length_Adjust(data)
#     window_amount = len(data) / EMG_WINDOW_SIZE
#     # windows_data = data.reshape(window_amount, WINDOW_SIZE, TYPE_LEN[type_name])
#     window_rest = len(data) % EMG_WINDOW_SIZE
#     data_len = (len(data) - window_rest) if window_rest != 0 else len(data)
#     windows_data = np.vsplit(data[0:data_len, :], window_amount)
#     win_index = 0
#     is_first = True
#     seg_all_feat = []
#     seg_RMS_feat = []
#     seg_VAR_feat = []
#
#     for Win_Data in windows_data:
#         # 依次处理每个window的数据
#         win_RMS_feat = np.sqrt(np.mean(np.square(Win_Data), axis=0))
#
#         win_avg_value = np.sum(Win_Data, axis=0) / EMG_WINDOW_SIZE
#         win_avg = np.array([win_avg_value for j in range(EMG_WINDOW_SIZE)])
#
#         diff = Win_Data - win_avg
#         square = np.square(diff)
#
#         win_VAR_feat = np.mean(square, axis=0)
#         # 将每个window特征提取的数据用vstack叠起来
#         if win_index == 0:
#             seg_RMS_feat = win_RMS_feat
#             seg_VAR_feat = win_VAR_feat
#         else:
#             seg_RMS_feat = np.vstack((seg_RMS_feat, win_RMS_feat))
#             seg_VAR_feat = np.vstack((seg_VAR_feat, win_VAR_feat))
#         win_index += 1
#
#         # 将三种特征拼接成一个长向量
#         # 层叠 转置 遍历展开
#         Seg_Feat = np.vstack((win_RMS_feat, win_VAR_feat))
#         All_Seg_Feat = Seg_Feat.T.ravel()
#
#         if is_first:
#             is_first = False
#             seg_all_feat = All_Seg_Feat
#         else:
#             seg_all_feat = np.vstack((seg_all_feat, All_Seg_Feat))
#
#     return seg_RMS_feat, seg_VAR_feat, seg_all_feat




'''
提取一个手势的一个batch的某一信号种类的全部数据
数据形式保存不变 只改变数值和每次采集ndarray的长度
（特征提取会改变数据的数量）
'''

def feature_extract(data_set, type_name):
    """
    特征提取
    :param data_set: 来自Load_From_File过程的返回值 一个dict
                     包含一个手语 三种采集数据类型的 多次采集过程的数据
    :param type_name: 数据采集的类型 决定nparray的长度
    :return: 一个dict 包含这个数据采集类型的原始数据,3种特征提取后的数据,特征拼接后的特征向量
            仍保持多次采集的数据放在一起
    """
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
    data = Length_Adjust(data)
    window_amount = len(data) / WINDOW_SIZE
    # windows_data = data.reshape(window_amount, WINDOW_SIZE, TYPE_LEN[type_name])
    windows_data = np.vsplit(data[0:160], window_amount)
    win_index = 0
    is_first = True
    seg_all_feat = []
    seg_ARC_feat = []
    seg_RMS_feat = []
    seg_ZC_feat = []

    for Win_Data in windows_data:
        # 依次处理每个window的数据
        win_RMS_feat = np.sqrt(np.mean(np.square(Win_Data), axis=0))
        Win_Data1 = np.vstack((Win_Data[1:, :], np.zeros((1, TYPE_LEN[type_name]))))
        win_ZC_feat = np.sum(np.sign(-np.sign(Win_Data) * np.sign(Win_Data1) + 1), axis=0) - 1
        win_ARC_feat = np.apply_along_axis(ARC, 0, Win_Data)
        # 将每个window特征提取的数据用vstack叠起来
        if win_index == 0:
            seg_RMS_feat = win_RMS_feat
            seg_ZC_feat = win_ZC_feat
            seg_ARC_feat = win_ARC_feat
        else:
            seg_RMS_feat = np.vstack((seg_RMS_feat, win_RMS_feat))
            seg_ZC_feat = np.vstack((seg_ZC_feat, win_ZC_feat))
            seg_ARC_feat = np.vstack((seg_ARC_feat, win_ARC_feat))
        win_index += 1

        # 将三种特征拼接成一个长向量
        # 层叠 转置 遍历展开
        Seg_Feat = np.vstack((win_RMS_feat, win_ZC_feat, win_ARC_feat))
        All_Seg_Feat = Seg_Feat.ravel()

        if is_first:
            is_first = False
            seg_all_feat = All_Seg_Feat
        else:
            seg_all_feat = np.vstack((seg_all_feat, All_Seg_Feat))

    seg_ARC_feat = normalize(seg_ARC_feat)
    seg_RMS_feat = normalize(seg_RMS_feat)
    seg_ZC_feat = normalize(seg_ZC_feat)
    seg_all_feat = normalize(seg_all_feat)

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
    feedback_data_set = pickle.load(file_)
    # [ (sign_id, data), .....  ]
    file_.close()
    data_set = list(range(SIGN_COUNT))
    for each_cap in feedback_data_set:
        data_set[each_cap[0]] = each_cap[1]
    return data_set

def wavelet_trans(data):
    data = np.array(data).T
    data = pywt.threshold(data, 30, mode='hard')
    data = pywt.wavedec(data, wavelet='db3', level=5)
    data = np.vstack((data[0].T, np.zeros(8))).T
    # 再次阈值滤波
    data = pywt.threshold(data, 20, mode='hard')
    data = standardize(data)
    data = eliminate_zero_shift(data)
    data = np.abs(data)
    # cap = normalize(cap)
    return data.T

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



def main():
    sign_id = 16
    # 从采集文件获取数据
    data_set = Load_ALL_Data(sign_id=sign_id, batch_num=44)
    # 从feedback文件获取数据
    # data_set = load_from_file_feed_back()[sign_id]

    # 数据采集类型 emg acc gyr
    data_cap_type = 'emg'

    # 数据特征类型 zc rms arc
    data_feat_type = 'trans'

    data_set = feature_extract(data_set, data_cap_type)

    # print_plot(data_set, data_cap_type, data_feat_type)

    # 将采集数据转换为训练数据
    pickle_to_file(batch_num=81)

if __name__ == "__main__":
    main()
