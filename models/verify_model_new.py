# Siamese-Networks

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN: input len -> output len
# Lout=floor((Lin+2∗padding−dilation∗(kernel_size−1)−1)/stride+1)


WEIGHT_DECAY = 0.000002
BATCH_SIZE = 64
LEARNING_RATE = 0.00022
EPOCH = 200

class SiameseNetwork(nn.Module):
    def __init__(self, train=True):
        """
        用于生成vector 进行识别结果验证
        :param train: 设置是否为train 模式
        """
        nn.Module.__init__(self)

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

        if train:
            self.work_mode = 'train'
            self.train()
        else:
            self.work_mode = 'eval'
            self.eval()

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
        if self.work_mode == 'train':
            out1 = self.forward_once(xs[0])
            out2 = self.forward_once(xs[1])
            return out1, out2
        else:
            return self.forward_once(xs[0])

    def exc_train(self):
        # only import train staff in training env
        from train_util.data_set import generate_data_set, SiameseNetworkTrainDataSet
        from train_util.common_train import train
        from torch.utils.data import dataloader as DataLoader

        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        loss_func = ContrastiveLoss()
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        data_set = generate_data_set(0.1, SiameseNetworkTrainDataSet)
        data_loader = {
            'train': DataLoader.DataLoader(data_set['train'],
                                           shuffle=True,
                                           batch_size=BATCH_SIZE, ),
            'test': DataLoader.DataLoader(data_set['test'],
                                          shuffle=True,
                                          batch_size=1, )
        }
        train(model=self,
              model_name='verify_68',
              EPOCH=EPOCH,
              optimizer=optimizer,
              exp_lr_scheduler=None,
              loss_func=loss_func,
              save_dir='.',
              data_set=data_set,
              data_loader=data_loader,
              test_result_output_func=test_result_output,
              cuda_mode=0,
              print_inter=2,
              val_inter=10,
              scheduler_step_inter=50
              )

    def load_params(self, path):
        file_list = os.listdir(path)
        target = None
        for each in file_list:
            if each.startswith("verify_68"):
                target = each
                break
        if target is None:
            raise Exception("can't find satisfy model params")

        target = os.path.join(path, target)
        # only load tail part
        if self.is_transfer_style:
            self.out.load_state_dict(torch.load(target))
        # load whole part
        else:
            self.load_state_dict(torch.load(target))

def test_result_output(result_list, epoch, loss):
    same_arg = []
    diff_arg = []
    for each in result_list:
        model_output = each[1]
        target_output = each[0]
        dissimilarities = F.pairwise_distance(model_output[0], model_output[1])
        dissimilarities = torch.squeeze(dissimilarities).data.float()[0]

        if target_output == 1.0:
            diff_arg.append(dissimilarities)
        elif target_output == 0.0:
            same_arg.append(dissimilarities)

    same_arg = np.array(same_arg)
    diff_arg = np.array(diff_arg)

    diff_min = np.min(diff_arg)
    diff_max = np.max(diff_arg)
    diff_var = np.var(diff_arg)

    same_max = np.max(same_arg)
    same_min = np.min(same_arg)
    same_var = np.var(same_arg)

    same_arg = np.mean(same_arg, axis=-1)
    diff_arg = np.mean(diff_arg, axis=-1)
    print("****************************")
    print("epoch: %s\nloss: %s\nprogress: %.2f lr: %f" %
          (epoch, loss, 100 * epoch / EPOCH, LEARNING_RATE))
    diff_res = "diff info \n    diff max: %f min: %f, mean: %f var: %f\n " % \
               (diff_max, diff_min, diff_arg, diff_var) + \
               "    same max: %f min: %f, mean: %f, same_var %f" % \
               (same_max, same_min, same_arg, same_var)
    print(diff_res)

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
