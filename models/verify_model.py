# Siamese-Networks

import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN: input len -> output len
# Lout=floor((Lin+2∗padding−dilation∗(kernel_size−1)−1)/stride+1)


WEIGHT_DECAY = 0.000002
BATCH_SIZE = 64

class SiameseNetwork(nn.Module):
    def __init__(self, train=True):
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

        self.LEARNING_RATE = 0.00022
        self.EPOCH = 100
        self.coding_model = nn.Sequential(
            nn.Conv1d(  # 14 x 64
                in_channels=14,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
            ),  # 32 x 64
            # 通常插入在激活函数和FC层之间 对神经网络的中间参数进行normalization
            nn.BatchNorm1d(64),  # 32 x 64
            nn.LeakyReLU(),

            nn.Conv1d(  # 14 x 64
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
            ),  # 32 x 64
            # 通常插入在激活函数和FC层之间 对神经网络的中间参数进行normalization
            nn.BatchNorm1d(64),  # 32 x 64
            nn.LeakyReLU(),
            # only one pooling
            nn.MaxPool1d(kernel_size=3, stride=2),  # 32 x 21

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=1
            ),  # 40 x 21
            nn.BatchNorm1d(128),  # 40 x 21
            nn.LeakyReLU(),

            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=1
            ),  # 40 x 21
            nn.BatchNorm1d(128),  # 40 x 21
            nn.MaxPool1d(kernel_size=3, stride=2)

        )
        self.out = nn.Sequential(
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(1920, 1024),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
        )


    def forward_once(self, x):
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
