# coding:utf-8
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

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

# normalize
def scale_data_block(data_block):
    for each_line in data_block:
        for each_feat_dim in range(len(SCALE)):
            each_line[each_feat_dim] /= SCALE[each_feat_dim]

DATA_DIR_PATH = os.getcwd() + '\\data'
BATCH_SIZE = 256

CUDA_AVAILABLE = torch.cuda.is_available()
print('cuda_status: %s' % str(CUDA_AVAILABLE))

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

# process data
data_input, data_label = [], []
for (labelItem, dataItem) in rawData:
    if len(dataItem) == 10:
        # 归一化
        scale_data_block(dataItem)
        for each_line in dataItem:
            data_input.append(each_line)
    else:
        print("len error")

print('data_len: %s' % len(data_input))

data_input = torch.from_numpy(np.array(data_input)).float()

train_loader = Data.DataLoader(dataset=data_input,
                               batch_size=BATCH_SIZE,
                               shuffle=True)

start_time_raw = time.time()
start_time = time.strftime('%H:%M:%S', time.localtime(start_time_raw))
print('start_at: %s' % start_time)

autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)
loss_func = nn.MSELoss()

for epoch in range(400):
    for x in train_loader:
        init_data = Variable(x.view(-1, 44))

        encoded, decoded = autoencoder(init_data)

        loss = loss_func(decoded, init_data)  # mean square error
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

    if epoch % 20 == 0:
        print('epoch: %d' % epoch)
        print('loss %s' % (str(loss.data.float())))

end_time_raw = time.time()
end_time = time.strftime('%H:%M:%S', time.localtime(end_time_raw))
print('end_at: %s' % end_time)

cost_time = end_time_raw - start_time_raw
cost_time = time.strftime('%H:%M:%S', time.gmtime(cost_time, ))
print('cost time: %s' % cost_time)

end_time = time.strftime('%m-%d,%H-%M', time.localtime(end_time_raw))
torch.save(autoencoder.state_dict(), 'autoencoder_model%s.pkl' % end_time)
