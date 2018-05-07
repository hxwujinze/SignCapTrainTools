import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

from RNN_model import BATCH_SIZE, EPOCH, NNet_SIZE, NNet_LEVEL, NNet_output_size, \
    CLASS_COUNT, LEARNING_RATE, WEIGHT_DECAY, DROPOUT
from RNN_model import LSTM

# 由于softmax输出的是十四个概率值 于是取最大的那个就是最可能正确的答案
# 取最大值 并且转换为int
def get_max_index(tensor):
    # print('置信度')
    # print(tensor[60].data.float())
    tensor = torch.max(tensor, dim=1)[1]
    # 对矩阵延一个固定方向取最大值
    return torch.squeeze(tensor).data.int()


CUDA_AVAILABLE = torch.cuda.is_available()
print('cuda_status: %s' % str(CUDA_AVAILABLE))

DATA_DIR_PATH = os.path.join(os.getcwd(), 'data')

# load data
f = open(os.path.join(DATA_DIR_PATH, 'data_set_rnn'), 'r+b')
raw_data = pickle.load(f)
f.close()

# 检查rawData中 feedback数据集是否存在
try:
    raw_data = raw_data[1].extend(raw_data[2])
except IndexError:
    raw_data = raw_data[1]
# train_data => (batch_amount, data_set)
random.shuffle(raw_data)


# process data
data_input, data_label = [], []
for (each_label, each_data) in raw_data:
    data_input.append(each_data)
    data_label.append(each_label - 1)

DATA_SET_SIZE = len(data_input)
INPUT_LEN = len(data_input[0])
print('data_len: %s' % DATA_SET_SIZE)
print('each input len: %s' % INPUT_LEN)

data_input = torch.from_numpy(np.array(data_input)).float()
data_label = torch.from_numpy(np.array(data_label))

# split and batch with data loader
# 0~1000 test
test_input_init = data_input[:1000]
test_label = data_label[:1000]
test_label = test_label.numpy()

# 1000~n train
training_input = data_input[1000:]
training_label = data_label[1000:]
training_set = Data.TensorDataset(data_tensor=training_input,
                                  target_tensor=training_label)
loader = Data.DataLoader(
    dataset=training_set,
    batch_size=BATCH_SIZE,  # should be tuned when data become bigger
    shuffle=True
)

# define loss function and optimizer
model = LSTM()
model.cuda()
# 转换为GPU对象

model.train()
# 转换为训练模式

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# learning rate can be tuned for better performance

start_time_raw = time.time()
start_time = time.strftime('%H:%M:%S', time.localtime(start_time_raw))
print('start_at: %s' % start_time)

# start training
# epoch: 用所有训练数据跑一遍称为一次epoch
for epoch in range(EPOCH + 1):
    for batch_x, batch_y in loader:
        batch_x = Variable(batch_x).cuda()
        batch_y = Variable(batch_y).cuda()
        batch_out = model(batch_x)
        batch_out = torch.squeeze(batch_out)
        loss = loss_func(batch_out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        model.eval()
        # 转换为求值模式
        test_input = Variable(test_input_init).cuda()  # 转换在gpu内跑识别
        # 转换为可读取的输入 Variable
        # 如下进行nn的正向使用 分类
        test_output = model(test_input).cpu()  # 从gpu中取回cpu算准确度
        # 需要从gpu的显存中取回内存进行计算正误率
        test_output = get_max_index(test_output)
        # softmax是14个概率的输出
        # test数据是连续的100个输入 于是输出也是一个 100 * 14 的矩阵
        test_output = test_output.numpy()

        test_result = {}
        all_t_cnt = 0
        all_f_cnt = 0
        for i in range(len(test_label)):
            if test_result.get(test_label[i]) is None:
                test_result[test_label[i]] = {
                    't': 0,
                    'f': 0
                }
            if test_output[i] == test_label[i]:
                all_t_cnt += 1
                test_result[test_label[i]]['t'] += 1
            else:
                all_f_cnt += 1
                test_result[test_label[i]]['f'] += 1
        accuracy_res = "accuracy of each sign:\n"

        for each_sign in sorted(test_result.keys()):
            t_cnt = test_result[each_sign]['t']
            f_cnt = test_result[each_sign]['f']
            accuracy_rate = t_cnt / (t_cnt + f_cnt)
            accuracy_res += "sign %d, accuracy %f (%d / %d)\n" % \
                            (each_sign, accuracy_rate, t_cnt, t_cnt + f_cnt)
        accuracy_res += "overall accuracy: %.5f\n" % (all_t_cnt / (all_f_cnt + all_t_cnt))

        print(accuracy_res)
        print("\n\nepoch: %s\nloss: %s\nprogress: %.2f" %
              (epoch, loss.data.float()[0], 100 * epoch / EPOCH))


end_time_raw = time.time()
end_time = time.strftime('%H:%M:%S', time.localtime(end_time_raw))
print('end_at: %s' % end_time)

cost_time = end_time_raw - start_time_raw
cost_time = time.strftime('%H:%M:%S', time.gmtime(cost_time, ))
print('cost time: %s' % cost_time)

end_time = time.strftime('%m-%d,%H-%M', time.localtime(end_time_raw))
model = model.cpu()
torch.save(model.state_dict(), os.path.join(DATA_DIR_PATH, 'rnn_model%s.pkl' % end_time))

file = open(os.path.join(DATA_DIR_PATH, 'rnn_models_info_%s' % end_time), 'w')
info = 'data_set_size:%d\n' % DATA_SET_SIZE + \
       'input_size:%d\n' % INPUT_LEN + \
       'batch_size:%d\n' % BATCH_SIZE + \
       accuracy_res + \
       'loss: %f\n' % loss.data.float()[0] + \
       'NNet:%d x %d\n' % (NNet_LEVEL, NNet_SIZE) + \
       'Epoch: %d\n' % EPOCH + \
       'NNet output size: %d\n' % NNet_output_size + \
       'classes cnt %d\n' % CLASS_COUNT + \
       'learning rate %f\n' % LEARNING_RATE + \
       'weight_decay %f\n' % WEIGHT_DECAY + \
       'dropout %f' % DROPOUT
info += str(model)

file.writelines(info)
file.close()
# how to read? :
# model.load_state_dict(torch.load('model_param.pkl'))
