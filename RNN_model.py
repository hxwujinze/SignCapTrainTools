
import torch.nn as nn
import torch.nn.functional as F

"""
超参数
每次调整模型修改这里
修改完成进行部署时直接将该文件复制到服务器目录
"""
BATCH_SIZE = 64
EPOCH = 1100
NNet_SIZE = 30
NNet_LEVEL = 3
NNet_output_size = 24
CLASS_COUNT = 24
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0000002
DROPOUT = 0.5

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=44,  # feature's number
            # 2*(3+3+3*4) +(8 + 8 +4*8)
            hidden_size=NNet_SIZE,  # hidden size of rnn layers
            num_layers=NNet_LEVEL,  # the number of rnn layers

            batch_first=True,
            dropout=DROPOUT)
        # dropout :
        # 在训练时，每次随机（如 50% 概率）忽略隐层的某些节点；
        # 这样，我们相当于随机从 2^H 个模型中采样选择模型；同时，由于每个网络只见过一个训练数据
        # 使得模型保存一定的随机性 避免过拟合严重
        self.out = nn.Linear(NNet_SIZE, NNet_output_size)
        self.out2 = nn.Linear(NNet_output_size, CLASS_COUNT)
        # use soft max classifier.
        # 在输出层中间加了层softmax 用于分类
        # softmax将会输出这十四个结果每个可能是正确的概率

    def forward(self, x):
        lstm_out, (h_n, h_c) = self.lstm(x, None)
        out = F.relu(lstm_out)
        out = self.out(out[:, -1, :])
        out = F.relu(out)
        out2 = self.out2(out)
        out2 = F.softmax(out2)
        return out2
