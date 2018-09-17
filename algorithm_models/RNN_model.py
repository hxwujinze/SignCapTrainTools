# coding:utf-8
import torch.nn as nn

"""
超参数
每次调整模型修改这里
修改完成进行部署时直接将该文件复制到服务器目录
"""
INPUT_SIZE = 30
BATCH_SIZE = 64

# RNN 使用扁平而浅的网络效果好
NNet_SIZE = 64
NNet_LEVEL = 3
NNet_output_size = 32
EPOCH = 1200
CLASS_COUNT = 24
LEARNING_RATE = 0.0006
WEIGHT_DECAY = 0.000008
DROPOUT = 0.5

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,  # feature's number
            # 2 *（3 + 3 + 5） + 8
            hidden_size=NNet_SIZE,  # hidden size of rnn layers
            num_layers=NNet_LEVEL,  # the number of rnn layers
            batch_first=True,
            dropout=DROPOUT
        )  # dropout :
        # 在训练时，每次随机（如 50% 概率）忽略隐层的某些节点；
        # 这样，我们相当于随机从 2^H 个模型中采样选择模型；同时，由于每个网络只见过一个训练数据
        # 使得模型保存一定的随机性 避免过拟合严重

        self.out = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(NNet_SIZE, NNet_SIZE),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(NNet_SIZE, NNet_output_size),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(NNet_output_size, CLASS_COUNT),
            nn.Softmax(),
            # use soft max classifier.
            # 在输出层中间加了层softmax 用于分类
            # softmax将会输出这十四个结果每个可能是正确的概率
        )


    def forward(self, x):
        rnn_out, h_n = self.rnn(x)
        # GRU model 返回两个值
        #   一个是每次输入一个向量时模型的输出
        #        batch_num x input_len x each_out_vector
        #   一个是模型在最后一刻内部的hidden state信息
        #   LSTM 是三个值 后两个是hidden state 和 cell state
        rnn_out = rnn_out[:, -1, :]
        # 只取最后一个output 这个output是模型输入了所有输入向量时
        # 给出的结果
        out = self.out(rnn_out)
        return out
