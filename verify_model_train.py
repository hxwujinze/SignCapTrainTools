# Siamese-Networks train

import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

from verify_model import SiameseNetwork, ContrastiveLoss, \
    EPOCH, BATCH_SIZE, WEIGHT_DECAY

DATA_DIR_PATH = os.path.join(os.getcwd(), 'data')

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
            x2_ = random.choice(self.data_dict[x2_label])
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

def train(verify_model_type):
    # load data
    f = open(os.path.join(DATA_DIR_PATH, 'data_set_%s' % verify_model_type), 'r+b')
    raw_data = pickle.load(f)
    f.close()

    try:
        raw_data = raw_data[1].extend(raw_data[2])
    except IndexError:
        raw_data = raw_data[1]
    # train_data => (batch_amount, data_set_emg)

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

    siamese_data_set = SiameseNetworkTrainDataSet(raw_data)
    # siamese_data_set.look_input_data()

    # split test data
    test_label = []
    test_data_x1 = []
    test_data_x2 = []
    # last 300
    for it in range(siamese_data_set.data_len - 300, siamese_data_set.data_len):
        selected_data = siamese_data_set[it]
        test_data_x1.append(selected_data[0])
        test_data_x2.append(selected_data[1])
        test_label.append(selected_data[2][0])

    test_data_x1 = np.array(test_data_x1, dtype=np.float32)
    test_data_x2 = np.array(test_data_x2, dtype=np.float32)
    test_label = np.array(test_label, dtype=np.float32)

    test_data_x1 = torch.from_numpy(test_data_x1)
    test_data_x2 = torch.from_numpy(test_data_x2)
    test_label = torch.from_numpy(test_label)

    data_loader = Data.DataLoader(siamese_data_set,
                                  shuffle=True,
                                  batch_size=BATCH_SIZE)

    model = SiameseNetwork(model_type=verify_model_type, train=True)
    model.train()
    model.cuda()

    loss_func = ContrastiveLoss()
    global LEARNING_RATE
    if type_name == 'rnn':
        LEARNING_RATE = 0.00035
    else:
        LEARNING_RATE = 0.000125
    print("lr: %f " % LEARNING_RATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    start_time_raw = time.time()
    start_time = time.strftime('%H:%M:%S', time.localtime(start_time_raw))
    print('start_at: %s' % start_time)

    for epoch in range(EPOCH + 1):
        for x1, x2, label in data_loader:
            x1 = Variable(x1).float().cuda()
            x2 = Variable(x2).float().cuda()
            out1, out2 = model.forward(x1, x2)
            label = Variable(label).cuda()
            loss = loss_func(out1, out2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            test_input_x1 = Variable(test_data_x1).cuda()
            test_input_x2 = Variable(test_data_x2).cuda()
            # 使用方法就是将数据放入模型 然后将得到的编码与要对比类型的编码
            # 使用F.pairwise_distance 进行计算 相同的手语一般会小于0.5 不同则会大于2

            test_output = model(test_input_x1, test_input_x2)
            dissimilarities = F.pairwise_distance(test_output[0], test_output[1])
            dissimilarities = torch.squeeze(dissimilarities).data

            same_arg = []
            diff_arg = []
            for each in range(len(test_label)):
                if test_label[each] == 1.0:
                    diff_arg.append(dissimilarities[each])
                if test_label[each] == 0.0:
                    same_arg.append(dissimilarities[each])
            same_arg = np.array(same_arg)
            diff_arg = np.array(diff_arg)

            diff_min = np.min(diff_arg)
            diff_max = np.max(diff_arg)
            same_max = np.max(same_arg)
            same_min = np.min(same_arg)
            same_arg = np.mean(same_arg, axis=-1)
            diff_arg = np.mean(diff_arg, axis=-1)
            print("****************************")
            print('epoch %d\nloss: %.6f\nprogress: %.2f' %
                  (epoch, loss.data.float()[0], 100 * epoch / EPOCH))
            diff_res = "diff info \n    diff max: %f min: %f, mean: %f\n" % \
                       (diff_max, diff_min, diff_arg) + \
                       "    same max: %f min: %f, mean: %f" % \
                       (same_max, same_min, same_arg)
            print(diff_res)

    end_time_raw = time.time()
    end_time = time.strftime('%H:%M:%S', time.localtime(end_time_raw))
    print('end_at: %s' % end_time)

    cost_time = end_time_raw - start_time_raw
    cost_time = time.strftime('%H:%M:%S', time.gmtime(cost_time, ))
    print('cost time: %s' % cost_time)

    end_time = time.strftime('%m-%d,%H-%M', time.localtime(end_time_raw))
    model.cpu()
    model.eval()
    torch.save(model.state_dict(), os.path.join(DATA_DIR_PATH, 'verify_model_%s_%s.pkl' %
                                                (verify_model_type, end_time)))

    file = open(os.path.join(DATA_DIR_PATH, 'verify_model_%s_info_%s' %
                             (verify_model_type, end_time)), 'w')

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

if __name__ == '__main__':
    type_name = input()
    train(type_name)
