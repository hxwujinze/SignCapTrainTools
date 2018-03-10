# coding:utf-8
from __future__ import print_function

import os

from numpy import *

u'''
测试数据长度是否ok
如果出现 i, j 的输出说明第i个手势的第j次采集数据长度不够
修复办法 把这次识别与其他识别次数的数据集的交界处该位这个数据短的次数的数据

'''

SIGN_COUNT = 14
BATCH_NUM = '2'

def file2matrix(filename, del_sign, separator, Data_Columns):
    fr = open(filename, 'r')
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, Data_Columns))
    index = 0
    for line in arrayOLines:
        line = line.strip()
        line = line.strip(del_sign)
        try:
            listFromLine = line.split(separator)
            returnMat[index, :] = listFromLine[0:Data_Columns]
        except ValueError:
            print('occur error \nin file name %s' % filename)
            print('line %s' % line)
            print('listFromLine %s' % str(listFromLine))
        index += 1
    return returnMat

def Load_ALL_Data(Path, Num_SLR, Width_EMG, Width_ACC, Width_GYR):
    # Load and return data
    # initialization
    Data_EMG = []
    Data_ACC = []
    Data_GYR = []
    print('Load ALL Data')

    for i in range(Num_SLR):
        File_EMG = Path + '\\' + BATCH_NUM + '\\Emg\\' + str(i + 1) + '.txt'
        Data_EMG.append(file2matrix(File_EMG, '()[]', ',', Width_EMG))
        File_ACC = Path + '\\' + BATCH_NUM + '\\Acceleration\\' + str(i + 1) + '.txt'
        Data_ACC.append(file2matrix(File_ACC, '()[]', ',', Width_ACC))
        File_GYR = Path + '\\' + BATCH_NUM + '\\Gyroscope\\' + str(i + 1) + '.txt'
        Data_GYR.append(file2matrix(File_GYR, '()[]', ',', Width_GYR))

    print('Load done')

    return Data_EMG, Data_ACC, Data_GYR

def Segmentation(Data, Num_Seg):
    Num = {}
    Seg_index = list(Data[:, -1])
    for i in Seg_index:
        Num[i] = Num.get(i, 0) + 1
    Seg = range(Num_Seg)
    Index = Num[0]
    for i in range(Num_Seg):
        Seg[i] = Data[Index: Index + Num[i + 1]]
        Index = Index + Num[i + 1]
    return Seg

if __name__ == "__main__":

    Path = os.getcwd()

    EMG, ACC, GYR = Load_ALL_Data(Path, SIGN_COUNT, 9, 3, 3)

    print('data length')
    for i in range(len(EMG)):
        print('in batch num %d' % (i + 1))
        Seg = Segmentation(EMG[i], 20)
        for j in range(len(Seg)):
            print(len(Seg[j]), end=', ')
        print('\n')

    print('unsatisfied data length')
    for i in range(len(EMG)):
        Seg = Segmentation(EMG[i], 20)
        result = ''
        for j in range(len(Seg)):
            if len(Seg[j]) < 165:
                result += ('{0}, '.format(j + 1))
        if result != '':
            result = 'in batch num %d\n' % (i + 1) + result
            print(result)
        # else:
        # print len(Seg[j]),
