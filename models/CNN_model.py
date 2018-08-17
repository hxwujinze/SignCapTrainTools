import torch
import torch.nn as nn
import torch.utils.data.dataloader as DataLoader

from train_util.data_set import generate_data_set

LEARNING_RATE = 0.00018
WEIGHT_DECAY = 0.0000002
EPOCH = 1200
BATCH_SIZE = 64


class CNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # input 14 x 64
        self.conv1 = nn.Sequential(
            # 使用VGGNet架构卷积
            nn.Conv1d(
                in_channels=14,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
            ),  # Lout=floor((Lin+2*padding-dilation*(kernel_size -1 ) - 1)/stride+1)
            # output 28 x 64
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            # output 28 x 64
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=3, stride=2)  # 64 x 32
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=1
            ),  # 32 x 21
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=1
            ),  # 32 x 21
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=3, stride=2)  # 128 x 16
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=1
            ),  # 32 x 21
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=1
            ),  # 32 x 21
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=1
            ),  # 32 x 21
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=3, stride=2),  # 256 x 8
        )

        self.out1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(1920, 1024),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(512, 69),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out1(x)
        return x

    def exc_train(self):
        from train_util.common_train import train
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        loss_func = nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        data_set = generate_data_set(0.1)
        data_loader = {
            'train': DataLoader.DataLoader(data_set['train'], shuffle=True, batch_size=64),
            'test': DataLoader.DataLoader(data_set['test'], shuffle=True, batch_size=1)
        }
        train(model=self,
              model_name='cnn_24',
              EPOCH=100,
              optimizer=optimizer,
              exp_lr_scheduler=lr_scheduler,
              loss_func=loss_func,
              save_dir='.',
              data_set=data_set,
              data_loader=data_loader,
              cuda_mode=True
              )

def get_max_index(tensor):
    # print('置信度')
    # print(tensor.data.float()[0])
    tensor = torch.max(tensor, dim=1)[1]
    # 对矩阵延一个固定方向取最大值
    return torch.squeeze(tensor).data.int()

def output_len(Lin, padding, kernel_size, stride):
    Lout = (Lin + 2 * padding - (kernel_size - 1) - 1) / stride + 1
    return Lout
