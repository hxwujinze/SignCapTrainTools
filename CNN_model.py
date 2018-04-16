import torch
import torch.nn as nn

LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0000002
EPOCH = 1000
BATCH_SIZE = 64

class RawInputCNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # input 14 x 160
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=14,
                out_channels=28,
                kernel_size=8,
                padding=4,
                stride=4,
            ),  # Lout=floor((Lin+2∗padding−dilation∗(kernel_size−1)−1)/stride+1)
            # output 28 x 40
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 28 x 20
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=28,
                out_channels=32,
                kernel_size=4,
                padding=2,
                stride=1
            ),  # 32 x 20
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 32 x 10
        )

        self.out1 = nn.Sequential(
            nn.Linear(32 * 10, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
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
