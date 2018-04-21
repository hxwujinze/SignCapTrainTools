import torch
import torch.nn as nn

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.00000002
EPOCH = 1000
BATCH_SIZE = 64

class CNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # input 14 x 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=14,
                out_channels=56,
                kernel_size=4,
                padding=2,
                stride=1,
            ),  # Lout=floor((Lin+2∗padding−dilation∗(kernel_size−1)−1)/stride+1)
            # output 28 x 64
            nn.BatchNorm1d(56),
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 28 x 32
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=56,
                out_channels=60,
                kernel_size=3,
                padding=1,
                stride=1
            ),  # 32 x 16
            nn.BatchNorm1d(60),
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 60 x 16
        )

        self.out1 = nn.Sequential(
            nn.Linear(60 * 16, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 24),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out1(x)
        return x

def get_max_index(tensor):
    # print('置信度')
    # print(tensor.data.float()[0])
    tensor = torch.max(tensor, dim=1)[1]
    # 对矩阵延一个固定方向取最大值
    return torch.squeeze(tensor).data.int()

def output_len(Lin, padding, kernel_size, stride):
    Lout = (Lin + 2 * padding - (kernel_size - 1) - 1) / stride + 1
    return Lout
