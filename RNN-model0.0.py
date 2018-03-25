# coding:utf-8
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

# 由于softmax输出的是十四个概率值 于是取最大的那个就是最可能正确的答案
# 取最大值 并且转换为int
def getMaxIndex(tensor):
    tensor = torch.max(tensor, dim=1)[1]
    # 对矩阵延一个固定方向取最大值
    return torch.squeeze(tensor).data.int()

# normalize
def scale_data_block(data_block):
    for each_line in data_block:
        for each_feat_dim in range(len(SCALE)):
            each_line[each_feat_dim] /= SCALE[each_feat_dim]

CUDA_AVAILABLE = torch.cuda.is_available()
print('cuda_status: %s' % str(CUDA_AVAILABLE))

DATA_DIR_PATH = os.getcwd() + '\\data'
BATCH_SIZE = 64


# load data
f = open(DATA_DIR_PATH + '\\data_set', 'r+b')
rawData = pickle.load(f)
f.close()

# 检查rawData中 feedback数据集是否存在
try:
    rawData = rawData[1].extend(rawData[2])
except IndexError:
    rawData = rawData[1]
# train_data => (batch_amount, data_set_emg)

random.shuffle(rawData)

f = open(DATA_DIR_PATH + '\\scale_rnn', 'r+b')
SCALE = pickle.load(f)
f.close()

f = open(DATA_DIR_PATH + '\\scale_rnn', 'w+b')
pickle.dump(SCALE, f, protocol=2)
f.close()

# process data
dataInput, dataLabel = [], []
for (labelItem, dataItem) in rawData:
    if len(dataItem) == 10:
        # 归一化
        scale_data_block(dataItem)
        dataInput.append(dataItem)
        dataLabel.append(labelItem - 1)
    else:
        print("len error")
print('data_len: %s' % len(dataInput))

dataInput = torch.from_numpy(np.array(dataInput)).float()
dataLabel = torch.from_numpy(np.array(dataLabel))
# split and batch with data loader
# 0~100 test
testInput = dataInput[:100]
testLabel = dataLabel[:100]
# 100~n train
trainingInput = dataInput[101:]
trainingLabel = dataLabel[101:]
trainingSet = Data.TensorDataset(data_tensor=trainingInput,
                                 target_tensor=trainingLabel)
loader = Data.DataLoader(
    dataset=trainingSet,
    batch_size=BATCH_SIZE,  # should be tuned when data become bigger
    shuffle=True
)

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=25,  # feature's number
            # 2*(3+3+3*4) +(8 + 8 +4*8)
            hidden_size=25,  # hidden size of rnn layers
            num_layers=3,  # the number of rnn layers
            # hidden_size=20,  # hidden size of rnn layers
            # num_layers=2,  # the number of rnn layers
            batch_first=True,
            dropout=0.40)
        # dropout :
        # 在训练时，每次随机（如 50% 概率）忽略隐层的某些节点；
        # 这样，我们相当于随机从 2^H 个模型中采样选择模型；同时，由于每个网络只见过一个训练数据
        # 使得模型保存一定的随机性 避免过拟合严重
        self.out = nn.Linear(25, 16)
        self.out2 = nn.Linear(16, 14)
        # use soft max classifier.
        # 在输出层中间加了层softmax 用于分类
        # softmax将会输出这十四个结果每个可能是正确的概率
        # self.out = nn.Linear(20, 12)
        # self.out2 = nn.Linear(12, 14)

    def forward(self, x):
        lstm_out, (h_n, h_c) = self.lstm(x, None)
        out = F.relu(lstm_out)
        out = self.out(lstm_out[:, -1, :])
        out = F.relu(out)
        # return out

        out2 = self.out2(out)
        out2 = F.softmax(out2, dim=1)
        return out2

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(44, 44),
            nn.Tanh(),
            nn.Linear(44, 30),
            nn.Tanh(),
            nn.Linear(30, 25),

        )

        self.decode = nn.Sequential(
            nn.Linear(25, 30),
            nn.Tanh(),
            nn.Linear(30, 44),
            nn.Tanh(),
            nn.Linear(44, 44),
        )

    def forward(self, x):
        encode = self.encode(x)
        decode = self.decode(encode)
        return encode, decode


# define loss function and optimizer
model = LSTM()

encoder = AutoEncoder()
encoder.load_state_dict(torch.load('autoencoder_model03-22,21-15.pkl'))
encoder.eval()

# model.cuda()
# 转换为GPU对象

model.train()
# 转换为训练模式

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0000002)
# learning rate can be tuned for better performance

start_time_raw = time.time()
start_time = time.strftime('%H:%M:%S', time.localtime(start_time_raw))
print('start_at: %s' % start_time)

# start training
# epoch: 用所有训练数据跑一遍称为一次epoch
for epoch in range(0, 681):

    for batch_x, batch_y in loader:
        batch_x = Variable(batch_x)  # .cuda()
        batch_y = Variable(batch_y)  # .cuda()
        new_batch_x = []
        for each_block in batch_x:
            input_block = []
            for each_line in each_block:
                encode, decode = encoder(each_line)
                input_block.append(encode)
            input_block = torch.stack(input_block)
            new_batch_x.append(input_block)
        new_batch_x = torch.stack(new_batch_x)

        batch_out = model(new_batch_x)
        batch_out = torch.squeeze(batch_out)    
        loss = loss_func(batch_out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        model.eval()
        # 转换为求值模式
        testInput_ = Variable(testInput)  #.cuda()  # 转换在gpu内跑识别
        new_batch_x = []
        for each_block in testInput_:
            input_block = []
            is_first_line = True
            for each_line in each_block:
                encode, decode = encoder(each_line)
                input_block.append(encode)
            input_block = torch.stack(input_block)
            new_batch_x.append(input_block)
        new_batch_x = torch.stack(new_batch_x)

        # 转换为可读取的输入 Variable
        # 如下进行nn的正向使用 分类
        testOuput_ = model(new_batch_x)  # .cpu()     # 从gpu中取回cpu算准确度
        # 需要从gpu的显存中取回内存进行计算正误率
        testOuput_ = getMaxIndex(testOuput_)
        # softmax是14个概率的输出
        # test数据是连续的100个输入 于是输出也是一个 100 * 14 的矩阵
        testOuput_ = testOuput_.numpy()
        testLabel_ = testLabel.numpy()
        right = 0
        error = 0
        for i in range(len(testLabel_)):
            if testOuput_[i] == testLabel_[i]:
                right += 1
            else:
                error += 1
        result = right / (right + error)
        print("epoch: %s\naccuracy: %s\nloss: %s" % (epoch, result, loss.data.float()))

end_time_raw = time.time()
end_time = time.strftime('%H:%M:%S', time.localtime(end_time_raw))
print('end_at: %s' % end_time)

cost_time = end_time_raw - start_time_raw
cost_time = time.strftime('%H:%M:%S', time.gmtime(cost_time, ))
print('cost time: %s' % cost_time)

end_time = time.strftime('%m-%d,%H-%M', time.localtime(end_time_raw))
torch.save(model.state_dict(), 'model_param%s.pkl' % end_time)

file = open('models_info_%s' % end_time, 'w')
file.writelines('batch_size:%d\nacc_result:%f' % (BATCH_SIZE, result))
file.close()
# how to read? :
# model.load_state_dict(torch.load('model_param.pkl'))
