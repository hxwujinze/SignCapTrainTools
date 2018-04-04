# coding:utf-8

from numpy import *
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.svm import SVC

DATA_DIR_PATH = os.getcwd() + '\\data'



def file2matrix(filename, del_sign, separator, Data_Columns):
    fr = open(filename, 'r')
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, Data_Columns))
    index = 0
    for line in arrayOLines:
        line = line.strip()
        line = line.strip(del_sign)
        listFromLine = line.split(separator)
        returnMat[index, :] = listFromLine[0:Data_Columns]
        index += 1
    return returnMat

def Load_ALL_Data(Path, Num_SLR, Width_EMG, Width_ACC, Width_GYR, Index_Folder):
    # Load and return data
    # initialization
    Data_EMG = []
    Data_ACC = []
    Data_GYR = []
    print('Load ALL Data')

    for i in range(Num_SLR):
        File_EMG = Path + '\\' + str(Index_Folder) + '\\Emg\\' + str(i + 1) + '.txt'
        Data_EMG.append(file2matrix(File_EMG, '()[]', ',', Width_EMG))
        File_ACC = Path + '\\' + str(Index_Folder) + '\\Acceleration\\' + str(i + 1) + '.txt'
        Data_ACC.append(file2matrix(File_ACC, '()[]', ',', Width_ACC))
        File_GYR = Path + '\\' + str(Index_Folder) + '\\Gyroscope\\' + str(i + 1) + '.txt'
        Data_GYR.append(file2matrix(File_GYR, '()[]', ',', Width_GYR))

    print('Load done')

    return Data_EMG, Data_ACC, Data_GYR

def Length_Adjust(A, Standard_Length):
    Leng = len(A) - Standard_Length
    if Leng < 0:
        print ('Length Error')
        A1 = A
    else:
        End = len(A) - Leng / 2
        Re = Leng % 2
        Begin = Leng / 2 + Re
        A1 = A[Begin:End, :]
    return A1

def Segmentation(Data_EMG, Data_ACC, Data_GYR, Num_Seg):
    Num = {}
    Seg_EMG_Data = []
    Seg_ACC_Data = []
    Seg_GYR_Data = []
    for Index_SLR in range(len(Data_EMG)):
        Num = {}
        Single_EMG = Data_EMG[Index_SLR]
        Single_ACC = Data_ACC[Index_SLR]
        Single_GYR = Data_GYR[Index_SLR]
        Seg_index = list(Single_EMG[:, -1])
        for i in Seg_index:
            Num[i] = Num.get(i, 0) + 1
        Seg_EMG = []
        Seg_ACC = []
        Seg_GYR = []
        Index = Num[0]
        for i in range(Num_Seg):
            Seg_EMG.append(Length_Adjust(Single_EMG[Index:Index + Num[i + 1], 0:8], 160))
            Seg_ACC.append(Length_Adjust(Single_ACC[Index:Index + Num[i + 1], 3:6], 160))
            Seg_GYR.append(Length_Adjust(Single_GYR[Index:Index + Num[i + 1], 3:6], 160))
            Index = Index + Num[i + 1]
        Seg_EMG_Data.append(Seg_EMG)
        Seg_ACC_Data.append(Seg_ACC)
        Seg_GYR_Data.append(Seg_GYR)
    return Seg_EMG_Data, Seg_ACC_Data, Seg_GYR_Data

def Folder_combination(Path, Num_Folder):
    for Index_Folder in range(Num_Folder):
        EMG_Data, ACC_Data, GYR_Data = Load_ALL_Data(Path, 12, 17, 6, 6, Index_Folder + 1)
        Seg_EMG, Seg_ACC, Seg_GYR = Segmentation(EMG_Data, ACC_Data, GYR_Data, 20)
        if Index_Folder == 0:
            Seg_EMG_Data = Seg_EMG
            Seg_ACC_Data = Seg_ACC
            Seg_GYR_Data = Seg_GYR
        else:
            for Index_SLR in range(len(Seg_EMG)):
                for Index_Seg in range(len(Seg_EMG[0])):
                    Seg_EMG_Data[Index_SLR].append(Seg_EMG[Index_SLR][Index_Seg])
                    Seg_ACC_Data[Index_SLR].append(Seg_ACC[Index_SLR][Index_Seg])
                    Seg_GYR_Data[Index_SLR].append(Seg_GYR[Index_SLR][Index_Seg])
    return Seg_EMG_Data, Seg_ACC_Data, Seg_GYR_Data

def ARC(Win_Data):
    Len_Data = len(Win_Data)
    AR_coefficient = polyfit(range(Len_Data), Win_Data, 3)
    return AR_coefficient

def Feature_Etraction(Seg_data, Size_Window, width_data):
    print('Feature Etraction')
    All_Feat = []
    for Single_Gesture in Seg_data:
        Gest_index = 0
        for Seg_Gesture in Single_Gesture:
            Num_Window = len(Seg_Gesture) / Size_Window
            Win_Seg_Data = vsplit(Seg_Gesture, Num_Window)
            Win_index = 0
            for Win_Data in Win_Seg_Data:
                RMS_Feat = sqrt(mean(square(Win_Data), axis=0))
                Win_Data1 = vstack(((Win_Data[1::, :]), zeros((1, width_data))))
                ZC_Feat = sum(sign(-sign(Win_Data) * sign(Win_Data1) + 1), axis=0) - 1
                ARC_Feat = apply_along_axis(ARC, 0, Win_Data)
                if Win_index == 0:
                    Seg_RMS_Feat = RMS_Feat
                    Seg_ZC_Feat = ZC_Feat
                    Seg_ARC_Feat = ARC_Feat
                else:
                    Seg_RMS_Feat = vstack((Seg_RMS_Feat, RMS_Feat))
                    Seg_ZC_Feat = vstack((Seg_ZC_Feat, ZC_Feat))
                    Seg_ARC_Feat = vstack((Seg_ARC_Feat, ARC_Feat))
                Win_index += 1

            Seg_Feat = vstack((Seg_RMS_Feat, Seg_ZC_Feat, Seg_ARC_Feat))
            All_Seg_Feat = (Seg_Feat.T).ravel()
            if Gest_index == 0:
                All_Single_Feat = All_Seg_Feat
            else:
                All_Single_Feat = vstack((All_Single_Feat, All_Seg_Feat))

            Gest_index += 1
        All_Feat.append(All_Single_Feat)
    print('Feature Etraction done')
    return All_Feat

def Combine_Feature(Feat1, Feat2, Feat3):
    Com_Feat = []
    for k in range(12):
        Size_Ges = min(len(Feat1[k]), len(Feat2[k]), len(Feat3[k]))
        Feat = hstack((Feat1[k][0:Size_Ges], Feat2[k][0:Size_Ges], Feat3[k][0:Size_Ges]))
        Com_Feat.append(Feat)
    return Com_Feat

def Add_Label(Com_Feat):
    index = 0
    for Single_Feat in Com_Feat:
        # norm_step_level = ges_all_feat
        Num_Seg = len(Single_Feat)
        Single_Label = zeros(Num_Seg) + index
        if index == 0:
            All_Feat = Single_Feat
            All_Label = Single_Label
        if index > 0:
            All_Feat = vstack((All_Feat, Single_Feat))
            All_Label = hstack((All_Label, Single_Label))
        index += 1
    return All_Feat, All_Label

def Standardization(All_Feat):
    max_abs_scaler = preprocessing.MaxAbsScaler()
    Norm_Feat = max_abs_scaler.fit_transform(All_Feat)
    savetxt('scale_rnn.txt', max_abs_scaler.scale_)
    print(len(max_abs_scaler.scale_))
    return Norm_Feat

def Pre_processing():
    Path = os.getcwd()
    Seg_EMG, Seg_ACC, Seg_GYR = Folder_combination(Path, 5)

    EMG_Feat = Feature_Etraction(Seg_EMG, 16, 8)
    ACC_Feat = Feature_Etraction(Seg_ACC, 16, 3)
    GYR_Feat = Feature_Etraction(Seg_GYR, 16, 3)

    Combine_Feat = Combine_Feature(EMG_Feat, ACC_Feat, GYR_Feat)

    All_Feat, All_Label = Add_Label(Combine_Feat)

    Norm_Feat = Standardization(All_Feat)
    return Norm_Feat, All_Label

#  直接从rnn训练用的data_set里读取数据
def trans_to_SVM_input(raw_data):
    input_data = []
    input_label = []
    for (each_label, each_data) in raw_data[1]:
        # 直接将数据进行展开
        input_data.append(each_data.ravel())
        input_label.append(each_label)
    return input_data, input_label

def Train_SVM(Norm_Feat, All_Label, C0=256, G0=0.0175):
    clf = SVC(kernel='rbf', C=C0, gamma=G0)
    scores = []
    for i in range(10):
        data_train, data_test, target_train, target_test = train_test_split(Norm_Feat, All_Label)
        # clf.fit(Norm_Feat, All_Label)
        clf.fit(data_train, target_train)
        joblib.dump(clf, "train_model.m")
        scores1 = cross_validation.cross_val_score(clf, data_test, target_test, cv=4)
        scores += list(scores1)
    return scores


def main():
    # load data
    f = open(DATA_DIR_PATH + '\\data_set', 'r+b')
    raw_data = pickle.load(f)
    f.close()

    input_data, input_label = trans_to_SVM_input(raw_data)

    scores = Train_SVM(input_data, input_label)
    scores_means = mean(scores)
    scores_var = 10000 * var(scores)

    print ('scores=', scores)
    print ('scores_means=', scores_means)
    print ('scores_var=', scores_var)

if __name__ == '__main__':
    main()
