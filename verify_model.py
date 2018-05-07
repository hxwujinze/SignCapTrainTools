# Siamese-Networks

import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN: input len -> output len
# Lout=floor((Lin+2∗padding−dilation∗(kernel_size−1)−1)/stride+1)

LEARNING_RATE = 0.000125
WEIGHT_DECAY = 0.0000002
EPOCH = 800
BATCH_SIZE = 64

class SiameseNetwork(nn.Module):
    def __init__(self, train=True, model_type='cnn'):
        """
        用于生成vector 进行识别结果验证
        :param train: 设置是否为train 模式
        :param model_type: 设置验证神经网络的模型种类 有rnn 和cnn两种
        """
        nn.Module.__init__(self)
        if train:
            self.status = 'train'
        else:
            self.status = 'eval'
        self.model_type = model_type

        if model_type == 'cnn':
            self.coding_model = nn.Sequential(
                # nn.BatchNorm1d(14),
                nn.Conv1d(  # 14 x 64
                    in_channels=14,
                    out_channels=32,
                    kernel_size=4,
                    padding=2,
                    stride=1,
                ),  # 32 x 64
                # 通常插入在激活函数和FC层之间 对神经网络的中间参数进行normalization
                nn.BatchNorm1d(32),  # 32 x 64
                nn.LeakyReLU(),
                # only one pooling
                nn.MaxPool1d(kernel_size=3),  # 32 x 21

                nn.Conv1d(
                    in_channels=32,
                    out_channels=40,
                    kernel_size=3,
                    padding=1,
                    stride=1
                ),  # 40 x 21
                nn.BatchNorm1d(40),  # 40 x 21

            )
            self.out = nn.Sequential(
                nn.Dropout(),
                nn.LeakyReLU(),
                nn.Linear(40 * 21, 512),
                nn.Dropout(),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.Dropout(),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
            )

        elif model_type == 'rnn':
            global LEARNING_RATE
            global EPOCH
            global BATCH_SIZE
            LEARNING_RATE = 0.0003
            EPOCH = 1000
            BATCH_SIZE = 64

            INPUT_SIZE = 30  # 2 *（3 + 3 + 5） + 8
            NNet_SIZE = 32
            NNet_LEVEL = 3
            NNet_output_size = 32

            self.coding_model = nn.LSTM(
                input_size=INPUT_SIZE,  # feature's number
                hidden_size=NNet_SIZE,  # hidden size of rnn layers
                num_layers=NNet_LEVEL,  # the number of rnn layers
                batch_first=True,
                dropout=0.5
            )
            self.out = nn.Sequential(
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(NNet_SIZE, NNet_SIZE),
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(NNet_SIZE, NNet_output_size),
            )

    def forward_once(self, x):
        if self.model_type == 'rnn':
            # rnn模型有额外的输入
            lstm_out, (h_n, h_c) = self.coding_model(x)
            x = lstm_out[:, -1, :]
        else:  # cnn的情况
            x = self.coding_model(x)
            x = x.view(x.size(0), -1)
        out = self.out(x)
        return out

    def forward(self, *xs):
        """
        train 模式输出两个vector 进行对比
        eval 模式输出一个vector
        """
        if self.status == 'train':
            out1 = self.forward_once(xs[0])
            out2 = self.forward_once(xs[1])
            return out1, out2
        else:
            return self.forward_once(xs[0])

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance,
                                                                    min=0.0), 2))
        return loss_contrastive
