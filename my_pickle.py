# coding:utf-8
import numpy as np

"""
将np array <-> string  便于在pipe中传递 而不用生成硬盘的文件 提高运行速度
"""


def loads(string):
    """
    将string 重新转换为np.array
    :param string:
    :return:
    """
    rows = string.split(';')
    data_mat = []
    for each_row in rows:
        if each_row == '':
            continue
        row_data = []
        for each_col in each_row.split(','):
            if each_col == '':
                continue
            row_data.append(float(each_col))
        data_mat.append(row_data)
    return np.array(data_mat)


def dumps(array):
    """
    将任意2维np array转为string
    :param array: np.array
    :return:  string
    """
    res = ''
    for row in array:
        for col in row:
            res += str(col) + ','
        res = '%s;' % res
    return res
