import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

from CNN_model import RawInputCNN, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCH, get_max_index

DATA_DIR_PATH = os.getcwd() + '\\data'

# load data
f = open(DATA_DIR_PATH + '\\data_setcnn_raw', 'r+b')
raw_data = pickle.load(f)
f.close()

try:
    raw_data = raw_data[1].extend(raw_data[2])
except IndexError:
    raw_data = raw_data[1]
# train_data => (batch_amount, data_set_emg)

random.shuffle(raw_data)

data_input, data_label = [], []
for (each_label, each_data) in raw_data:
    # 需要调整长度以及转置成时序
    data_input.append(each_data[16:144].T)
    data_label.append(each_label - 1)

DATA_SET_SIZE = len(data_input)
INPUT_LEN = len(data_input[0][0])
print('data_len: %s' % DATA_SET_SIZE)
print('each input len: %s' % INPUT_LEN)

data_input = torch.from_numpy(np.array(data_input)).float()
data_label = torch.from_numpy(np.array(data_label))

# split and batch with data loader
# 0~500 test
test_input_init = data_input[:500]
test_label = data_label[:500]

# 500~n train
training_input = data_input[500:]
training_label = data_label[500:]
training_set = Data.TensorDataset(data_tensor=training_input,
                                  target_tensor=training_label)

loader = Data.DataLoader(
    dataset=training_set,
    batch_size=BATCH_SIZE,  # should be tuned when data become bigger
    shuffle=True
)

cnn = RawInputCNN()
cnn.cuda()
cnn.train()

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

start_time_raw = time.time()
start_time = time.strftime('%H:%M:%S', time.localtime(start_time_raw))
print('start_at: %s' % start_time)

# start training
# epoch: 用所有训练数据跑一遍称为一次epoch
for epoch in range(0, EPOCH):
    for batch_x, batch_y in loader:
        batch_x = Variable(batch_x).cuda()
        batch_y = Variable(batch_y).cuda()
        batch_out = cnn(batch_x)
        batch_out = torch.squeeze(batch_out)
        loss = loss_func(batch_out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        cnn.eval()
        # 转换为求值模式
        test_input = Variable(test_input_init).cuda()  # 转换在gpu内跑识别
        # 转换为可读取的输入 Variable
        # 如下进行nn的正向使用 分类
        test_output = cnn(test_input).cpu()  # 从gpu中取回cpu算准确度
        # 需要从gpu的显存中取回内存进行计算正误率
        test_output = get_max_index(test_output)
        # softmax是14个概率的输出
        # test数据是连续的100个输入 于是输出也是一个 100 * 14 的矩阵
        testLabel_ = test_label.numpy()
        right = 0
        error = 0
        for i in range(len(testLabel_)):
            if test_output[i] == testLabel_[i]:
                right += 1
            else:
                error += 1
        result = right / (right + error)
        print("\n\nepoch: %s\naccuracy: %.4f\nloss: %s" % (epoch, result, loss.data.float()[0]))

end_time_raw = time.time()
end_time = time.strftime('%H:%M:%S', time.localtime(end_time_raw))
print('end_at: %s' % end_time)

cost_time = end_time_raw - start_time_raw
cost_time = time.strftime('%H:%M:%S', time.gmtime(cost_time, ))
print('cost time: %s' % cost_time)

end_time = time.strftime('%m-%d,%H-%M', time.localtime(end_time_raw))
model = cnn.cpu()
torch.save(model.state_dict(), DATA_DIR_PATH + '\\raw_input_cnn_model%s.pkl' % end_time)
