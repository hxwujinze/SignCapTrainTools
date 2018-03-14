# coding:utf-8
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

Width_EMG = 9
Width_ACC = 3
Width_GYR = 3
LENGTH = 160
WINDOW_SIZE = 16
SYS_PATH = os.getcwd()

TYPE_LEN = {
    'acc': 3,
    'gyr': 3,
    'emg': 8
}
CAP_TYPE_LIST = ['acc', 'emg', 'gyr']

def file2matrix(filename, data_col_num):
    del_sign = '()[]'
    separator = ','
    fr = open(filename, 'r')
    all_array_lines = fr.readlines()
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
'''

def Load_ALL_Data(sign_id, batch_num):
    # Load and return data
    # initialization
    Path = SYS_PATH
    file_num = sign_id
    print('Load ALL Data')
    file_emg = Path + '\\' + str(batch_num) + '\\Emg\\' + str(file_num) + '.txt'
    data_emg = file2matrix(file_emg, Width_EMG)
    file_acc = Path + '\\' + str(batch_num) + '\\Acceleration\\' + str(file_num) + '.txt'
    data_acc = file2matrix(file_acc, Width_ACC)
    file_gyr = Path + '\\' + str(batch_num) + '\\Gyroscope\\' + str(file_num) + '.txt'
    data_gyr = file2matrix(file_gyr, Width_GYR)
    print('Load done')
    capture_tag_list = list(data_emg[:, -1])
    capture_length_book = {}
    for i in capture_tag_list:
        capture_length_book[i] = capture_length_book.get(i, 0) + 1
    Index = 0
    resize_data_emg = []
    resize_data_acc = []
    resize_data_gyr = []
    for i in range(20):
        resize_data_emg.append(Length_Adjust(data_emg[Index:Index + capture_length_book[i], 0:8]))
        resize_data_acc.append(Length_Adjust(data_acc[Index:Index + capture_length_book[i], :]))
        resize_data_gyr.append(Length_Adjust(data_gyr[Index:Index + capture_length_book[i], :]))
        Index += capture_length_book[i]

    print("data resized")
    return {
        'emg': resize_data_emg,
        'acc': resize_data_acc,
        'gyr': resize_data_gyr
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

def print_plot(data_set, data_cap_type, data_feat_type):
    for dimension in range(TYPE_LEN[data_cap_type]):
        fig_acc = plt.figure()
        fig_acc.add_subplot(111, title='%s %s dim%s' % (data_feat_type, data_cap_type, str(dimension + 1)))
        for capture_num in range(1, 11):
            single_capture_data = trans_data_to_time_seqs(data_set[data_feat_type][capture_num])
            data = single_capture_data[dimension]
            plt.plot(range(len(data)), data)

    plt.show()

'''
提取一个手势的一个batch的某一信号种类的全部数据
数据形式保存不变 只改变数值和每次采集ndarray的长度
（特征提取会改变数据的数量）
'''

def feature_extract(data_set, type_name):
    data_set_rms_feat = []
    data_set_zc_feat = []
    data_set_arc_feat = []
    data_set = data_set[type_name]
    for data in data_set:
        Num_Window = len(data) / WINDOW_SIZE
        Win_Seg_Data = np.vsplit(data, Num_Window)
        Win_index = 0
        for Win_Data in Win_Seg_Data:
            RMS_Feat = np.sqrt(np.mean(np.square(Win_Data), axis=0))
            Win_Data1 = np.vstack((Win_Data[1:, :], np.zeros((1, TYPE_LEN[type_name]))))
            ZC_Feat = np.sum(np.sign(-np.sign(Win_Data) * np.sign(Win_Data1) + 1), axis=0) - 1
            ARC_Feat = np.apply_along_axis(ARC, 0, Win_Data)
            if Win_index == 0:
                Seg_RMS_Feat = RMS_Feat
                Seg_ZC_Feat = ZC_Feat
                Seg_ARC_Feat = ARC_Feat
            else:
                Seg_RMS_Feat = np.vstack((Seg_RMS_Feat, RMS_Feat))
                Seg_ZC_Feat = np.vstack((Seg_ZC_Feat, ZC_Feat))
                Seg_ARC_Feat = np.vstack((Seg_ARC_Feat, ARC_Feat))
            Win_index += 1
        data_set_arc_feat.append(Seg_ARC_Feat)
        data_set_rms_feat.append(Seg_RMS_Feat)
        data_set_zc_feat.append(Seg_ZC_Feat)
    return {
        'type_name': type_name,
        'raw': data_set,
        'arc': data_set_arc_feat,
        'rms': data_set_rms_feat,
        'zc': data_set_zc_feat
    }

def ARC(Win_Data):
    Len_Data = len(Win_Data)
    AR_coefficient = np.polyfit(range(Len_Data), Win_Data, 3)
    return AR_coefficient

def pickle_to_file(batch_num):
    for each_batch in range(1, batch_num + 1):
        for each_sign in range(1, 14):
            raw_data_set = Load_ALL_Data(batch_num=each_batch, sign_id=each_sign)
            for each_cap_type in CAP_TYPE_LIST:
                data_set = feature_extract(raw_data_set, each_cap_type)
                file = open('.\\data_set_pickle\\(%d,%d,%s)' % (each_batch, each_sign, each_cap_type), 'w+b')
                pickle.dump(data_set, file)
                file.close()

def load_from_file(batch_num):
    batch_list = []
    for each_batch in range(1, batch_num + 1):
        sign_list = []
        for each_sign in range(1, 14):
            data_cap_list = {}
            for each_cap_type in CAP_TYPE_LIST:
                try:
                    file = open('.\\data_set_pickle\\(%d,%d,%s)' % (each_batch, each_sign, each_cap_type), 'r+b')
                except FileNotFoundError:
                    print('找不到文件 检查data_set_pickle文件夹在当前目录且batch参数设置正确')
                    print("batch_num: %d" % batch_num)
                    return
                data_set = pickle.load(file)
                data_cap_list[each_cap_type] = data_set
            sign_list.append(data_cap_list)
        batch_list.append(sign_list)
    return batch_list



def main():
    data_set = Load_ALL_Data(sign_id=4, batch_num=1)
    data_cap_type = 'emg'
    # 数据采集类型 emg acc gyr
    data_feat_type = 'arc'
    # 数据特征类型 zc rms arc
    data_set = feature_extract(data_set, data_cap_type)
    print_plot(data_set, data_cap_type, data_feat_type)

    # pickle_to_file(batch_num=9)
    # data = load_from_file(batch_num=2)





if __name__ == "__main__":
    main()
