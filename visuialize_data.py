import matplotlib.pyplot as plt
import numpy as np

def file2matrix(filename, del_sign, separator, Data_Columns):
    fr = open(filename, 'r')
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, Data_Columns), dtype=float)
    index = 0
    for line in arrayOLines:
        line = line.strip()
        line = line.strip(del_sign)
        listFromLine = line.split(separator)
        returnMat[index, :] = listFromLine[0:Data_Columns]
        index += 1
    return returnMat

def Load_ALL_Data(Path, sign_id, batch_num):
    # Load and return data
    # initialization
    file_num = sign_id

    print('Load ALL Data')
    File_EMG = Path + '\\' + str(batch_num) + '\\Emg\\' + str(file_num) + '.txt'
    Data_EMG = file2matrix(File_EMG, '()[]', ',', Width_EMG)
    File_ACC = Path + '\\' + str(batch_num) + '\\Acceleration\\' + str(file_num) + '.txt'
    Data_ACC = file2matrix(File_ACC, '()[]', ',', Width_ACC)
    File_GYR = Path + '\\' + str(batch_num) + '\\Gyroscope\\' + str(file_num) + '.txt'
    Data_GYR = file2matrix(File_GYR, '()[]', ',', Width_GYR)
    print('Load done')

    return Data_EMG, Data_ACC, Data_GYR

def get_single_capture_time_seqs(target_capture_tag: int, data_set: tuple):
    is_first = True
    single_capture_emg = np.zeros(8)
    single_capture_gyr = np.zeros(3)
    single_capture_acc = np.zeros(3)

    iter = 0
    for each_cap_emg in data_set[0]:
        if each_cap_emg[-1] == target_capture_tag:
            if is_first:
                single_capture_emg = each_cap_emg[:-1]
                single_capture_acc = data_set[1][iter]  # data acc
                single_capture_gyr = data_set[2][iter]  # data gyr
                is_first = False
            else:
                single_capture_emg = np.vstack((single_capture_emg, each_cap_emg[:-1]))
                single_capture_acc = np.vstack((single_capture_acc, data_set[1][iter]))
                single_capture_gyr = np.vstack((single_capture_gyr, data_set[2][iter]))
        iter += 1
    return {'gyr': single_capture_gyr.T,
            'emg': single_capture_emg.T,
            'acc': single_capture_acc.T}

def print_plot(data_type: str, data_set):
    type_len = {
        'acc': 3,
        'gyr': 3,
        'emg': 8
    }
    for dimension in range(type_len[data_type]):
        fig_acc = plt.figure()
        fig_acc.add_subplot(111, title=data_type + ' dim ' + str(dimension + 1))
        for capture_num in range(0, 10):
            single_capture_data = get_single_capture_time_seqs(capture_num, data_set)
            data_len = len(single_capture_data[data_type][dimension])
            plt.plot(range(data_len), single_capture_data[data_type][dimension], )
    plt.show()

import os

Width_EMG = 9
Width_ACC = 3
Width_GYR = 3

def main():
    data_set = Load_ALL_Data(os.getcwd(), sign_id=1, batch_num=1)
    print_plot('emg', data_set)

if __name__ == "__main__":
    main()
