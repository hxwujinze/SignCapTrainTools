# Siamese-Networks train

import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

from models.verify_model import SiameseNetwork, ContrastiveLoss, \
    BATCH_SIZE, WEIGHT_DECAY
from process_data import DataScaler

DATA_DIR_PATH = os.path.join('..', 'data')
# 输入数据是个tuple  (label, data)
class SiameseNetworkTrainDataSet:
    """
    生成随机的相同或者不同的数据对进行训练
    """

    def __init__(self, data):
        self.data_len = len(data)
        self.data = data
        self.data_dict = {}
        for (each_label, each_data) in data:
            if self.data_dict.get(each_label) is None:
                self.data_dict[each_label] = [each_data]
            else:
                self.data_dict[each_label].append(each_data)
        self.class_cnt = len(self.data_dict.keys())

    # 要保证network尽可能见过最多class 并且能对他们进行正误的分辨
    def __getitem__(self, item):
        x1_ = random.choice(self.data)
        x1_label = x1_[0]
        x1_ = x1_[1]
        get_same = bool(random.randint(0, 1))
        if get_same:
            x2_ = random.choice(self.data_dict[x1_label])
        else:
            x2_label = x1_label
            while x2_label == x1_label:
                x2_label = random.randint(1, self.class_cnt)
                while self.data_dict.get(x2_label) is None:
                    x2_label = random.randint(1, self.class_cnt)
            x2_ = random.choice(self.data_dict[x2_label])

        # if get_same:
        #     x2_ = random.choice(self.data_dict[x1_label])
        # else:
        #     x2_label = x1_label
        #     while x2_label == x1_label:
        #         x2_label = random.randint(1, self.class_cnt)
        #     x2_ = random.choice(self.data_dict[x2_label])
        return x1_, \
               x2_, \
               np.array([0 if get_same else 1], dtype=np.float32)

    # 输出几个训练数据看看
    def look_input_data(self):
        import matplotlib.pyplot as plt
        for i in range(20):
            each_sample = self[i]
            fig = plt.figure()
            fig.add_subplot(111, title=str(each_sample[2]))
            plt.plot(range(len(each_sample[0][0])), each_sample[0][0])
            plt.plot(range(len(each_sample[1][0])), each_sample[1][0])
        plt.show()


    def __len__(self):
        return self.data_len

def train():
    # load data
    f = open(os.path.join(DATA_DIR_PATH, 'new_train_data'), 'r+b')
    raw_data = pickle.load(f)
    f.close()
    data = []
    scaler = DataScaler(DATA_DIR_PATH)
    for each in raw_data:
        data.append((each[1], scaler.normalize(each[0], 'cnn')))

    # try:
    #     raw_data = raw_data[1].extend(raw_data[2])
    # except IndexError:
    #     raw_data = raw_data[1]
    # train_data => (batch_amount, data_set_emg)
    raw_data = data
    random.shuffle(raw_data)
    input_len = len(raw_data[0][1])
    if input_len == 10:
        # 数据为RNN输入时不需要转置成x轴的时序信号
        for each in range(len(raw_data)):
            raw_data[each] = (raw_data[each][0],  # label
                              raw_data[each][1])  # data
    else:
        input_len = len(raw_data[0][1][0])
        for each in range(len(raw_data)):
            raw_data[each] = (raw_data[each][0],  # label
                              raw_data[each][1].T)  # data

    print('data_len: %s' % len(raw_data))
    print('each input len: %s' % input_len)

    siamese_data_set = SiameseNetworkTrainDataSet(raw_data[:40000])
    test_data_set = SiameseNetworkTrainDataSet(raw_data[40000:])

    # siamese_data_set.look_input_data()

    data_loader = Data.DataLoader(siamese_data_set,
                                  shuffle=True,
                                  batch_size=BATCH_SIZE,
                                  num_workers=1)
    test_data_loader = Data.DataLoader(test_data_set,
                                       shuffle=True,
                                       batch_size=1, )

    model = SiameseNetwork(train=True)
    LEARNING_RATE = model.LEARNING_RATE
    print('pick up last progress?')

    res = input()
    if res == 'y':
        model_tmp = load_model_param(model, 'verify_model')
        if model_tmp is not None:
            model = model_tmp
        else:
            print('cant find last progress')
        LEARNING_RATE = 0.0000001
    model.train()
    model.cuda()

    loss_func = ContrastiveLoss()
    print("lr: %f " % LEARNING_RATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    start_time_raw = time.time()
    start_time = time.strftime('%H:%M:%S', time.localtime(start_time_raw))
    print('start_at: %s' % start_time)
    EPOCH = model.EPOCH
    try:
        for epoch in range(EPOCH + 1):
            if epoch % 10 == 0 and epoch != 0:
                LEARNING_RATE *= 0.1
                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=LEARNING_RATE,
                                             weight_decay=WEIGHT_DECAY)
            loss_his = []
            for x1, x2, label in data_loader:
                x1 = Variable(x1).float().cuda()
                x2 = Variable(x2).float().cuda()
                out1, out2 = model.forward(x1, x2)
                label = Variable(label).cuda()
                loss = loss_func(out1, out2, label)
                optimizer.zero_grad()
                loss.backward()
                loss_his.append(loss.data.float()[0])
                optimizer.step()
            loss_val = np.mean(np.array(loss_his))
            print("epoch %d loss %s" % (epoch, str(loss_val)))

            if epoch % 10 == 0:
                model.eval()
                same_arg = []
                diff_arg = []
                for test_data_x1, test_data_x2, label in test_data_loader:

                    test_input_x1 = Variable(test_data_x1).float().cuda()
                    test_input_x2 = Variable(test_data_x2).float().cuda()
                    # 使用方法就是将数据放入模型 然后将得到的编码与要对比类型的编码
                    # 使用F.pairwise_distance 进行计算 相同的手语一般会小于0.5 不同则会大于2

                    test_output = model(test_input_x1, test_input_x2)
                    dissimilarities = F.pairwise_distance(test_output[0], test_output[1])
                    dissimilarities = torch.squeeze(dissimilarities).data[0]

                    label = label[0][0]
                    # print(label)
                    if label == 1.0:
                        diff_arg.append(dissimilarities)
                    if label == 0.0:
                        same_arg.append(dissimilarities)

                same_arg = np.array(same_arg)
                diff_arg = np.array(diff_arg)

                diff_min = np.min(diff_arg)
                diff_max = np.max(diff_arg)
                diff_var = np.var(diff_arg)

                same_max = np.max(same_arg)
                same_min = np.min(same_arg)
                same_var = np.var(same_arg)

                same_arg = np.mean(same_arg, axis=-1)
                diff_arg = np.mean(diff_arg, axis=-1)
                print("****************************")
                print("epoch: %s\nloss: %s\nprogress: %.2f lr: %f" %
                      (epoch, loss_val, 100 * epoch / EPOCH, LEARNING_RATE))
                diff_res = "diff info \n    diff max: %f min: %f, mean: %f var: %f\n " % \
                           (diff_max, diff_min, diff_arg, diff_var) + \
                           "    same max: %f min: %f, mean: %f, same_var %f" % \
                           (same_max, same_min, same_arg, same_var)
                print(diff_res)
    except:
        print("save model ?")
        res = input()
        if res != 'y':
            return
    end_time_raw = time.time()
    end_time = time.strftime('%H:%M:%S', time.localtime(end_time_raw))
    print('end_at: %s' % end_time)

    cost_time = end_time_raw - start_time_raw
    cost_time = time.strftime('%H:%M:%S', time.gmtime(cost_time, ))
    print('cost time: %s' % cost_time)

    end_time = time.strftime('%m-%d,%H-%M', time.localtime(end_time_raw))
    model.cpu()
    model.eval()
    torch.save(model.state_dict(), os.path.join(DATA_DIR_PATH, 'verify_model_%s.pkl' % end_time))

    file = open(os.path.join(DATA_DIR_PATH, 'verify_model_info_%s' % end_time), 'w')

    info = 'data_size:%d\n' % len(siamese_data_set) + \
           'batch_size:%d\n' % BATCH_SIZE + \
           diff_res + '\n' \
                      'loss: %f\n' % loss.data.float()[0] + \
           'Epoch: %d\n' % EPOCH + \
           'learning rate %f\n' % LEARNING_RATE + \
           'weight_decay %f\n' % WEIGHT_DECAY
    info += str(model)
    file.writelines(info)
    file.close()

def load_model_param(model, model_name):
    for root, dirs, files in os.walk(DATA_DIR_PATH):
        for file_ in files:
            file_name_split = os.path.splitext(file_)
            if file_name_split[1] == '.pkl' and file_name_split[0].startswith(model_name):
                print('load model params %s' % file_)
                file_ = os.path.join(DATA_DIR_PATH, file_)
                model.load_state_dict(torch.load(file_))
                model.eval()
                return model


if __name__ == '__main__':
    try:
        DATA_DIR_PATH = sys.argv[1]
    except IndexError:
        pass
    train()
