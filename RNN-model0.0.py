import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

def getMaxIndex(tensor):
    tensor = torch.max(tensor, dim=1)[1]
    return torch.squeeze(tensor).data.int()

DATA_DIR_PATH = os.getcwd() + '\\data'

# load data
f = open(DATA_DIR_PATH + '\\data_set', 'r+b')
rawData = pickle.load(f)
f.close()

f = open(DATA_DIR_PATH + '\\scale', 'r+b')
SCALE = pickle.load(f)
f.close()

CUDA_AVAILABLE = torch.cuda.is_available()
print('cuda_status: %s' % str(CUDA_AVAILABLE))

# normalize
def scale_data_block(data_block):
    for each_line in data_block:
        for each_feat_dim in range(len(SCALE)):
            each_line[each_feat_dim] /= SCALE[each_feat_dim]

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
    batch_size=20,  # should be tuned when data become bigger
    shuffle=True
)

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=84,  # feature's number
            # 2*(3+3+3*4) +(8 + 8 +4*8)

            hidden_size=20,  # hidden size of rnn layers
            num_layers=2,  # the number of rnn layers
            batch_first=True,
            dropout=0.5)  # ??

        self.out = nn.Linear(20, 12)  # use soft max classifier.
        self.out2 = nn.Linear(12, 14)

    def forward(self, x):
        lstm_out, (h_n, h_c) = self.lstm(x, None)
        out = F.relu(lstm_out)
        out = self.out(lstm_out[:, -1, :])
        out = F.relu(out)
        out2 = self.out2(out)
        out2 = F.softmax(out2)
        return out2

# define loss function and optimizer
model = LSTM()
model.train()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# learning rate can be tuned for better performance

start_time_raw = time.time()
start_time = time.strftime('%H:%M:%S', time.localtime(start_time_raw))
print('start_at: %s' % start_time)

# start training
# epoch : 用所有训练数据跑一遍称为一次epoch
for epoch in range(0, 1001):
    for batch_x, batch_y in loader:
        # 转换为GPU：
        # Variable(batch_x).cuda  所有Variable全部转换为gpu运算

        batch_x = Variable(batch_x)
        batch_y = Variable(batch_y)
        batch_out = model(batch_x)
        batch_out = torch.squeeze(batch_out)
        loss = loss_func(batch_out, batch_y)
        optimizer.zero_grad()
        #  print(loss)
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        model.eval()
        testInput_ = Variable(testInput)
        testOuput_ = model(testInput_)
        testOuput_ = getMaxIndex(testOuput_)
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
        print("epoch : ", epoch, "result: ", result)

end_time_raw = time.time()
end_time = time.strftime('%H:%M:%S', time.localtime(end_time_raw))
print('end_at: %s' % end_time)

cost_time = end_time_raw - start_time_raw
cost_time = time.strftime('%H:%M:%S', time.gmtime(cost_time, ))
print('cost time: %s' % cost_time)

torch.save(model.state_dict(), 'model_param.pkl')
# how to read? :
# model.load_state_dict(torch.load('model_param.pkl'))
