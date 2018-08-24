# Siamese-Networks

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import CNN_model

# CNN: input len -> output len
# Lout=floor((Lin+2∗padding−dilation∗(kernel_size−1)−1)/stride+1)


WEIGHT_DECAY = 0.000002
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCH = 200


class SiameseNetwork(nn.Module):
    def __init__(self, encode_model_path, transfer_style: bool, train=True):
        """
        用于生成vector 进行识别结果验证
        :param train: 设置是否为train 模式
        """
        nn.Module.__init__(self)
        self.encode_model = CNN_model.CNN()
        self.encode_model.load_params(encode_model_path)
        self.encode_model.set_encode_mode(True)
        self.is_transfer_style = transfer_style

        if train:
            self.work_mode = 'train'
            if transfer_style:
                self.encode_model.eval()
            else:
                self.encode_model.train()
        else:
            self.work_mode = 'eval'
            self.eval()

        self.out = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(1920, 1024),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(512, 512),
        )

    def forward_once(self, x):
        x = self.encod_model(x)
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
        loss_func = nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        data_set = generate_data_set(0.1, SiameseNetworkTrainDataSet)
        data_loader = {
            'train': DataLoader.DataLoader(data_set['train'],
                                           shuffle=True,
                                           batch_size=BATCH_SIZE,
                                           num_workers=3),
            'test': DataLoader.DataLoader(data_set['test'],
                                          shuffle=True,
                                          batch_size=1,
                                          num_workers=2)
        }
        train(model=self,
              model_name='verify_68',
              EPOCH=EPOCH,
              optimizer=optimizer,
              exp_lr_scheduler=lr_scheduler,
              loss_func=loss_func,
              save_dir='.',
              data_set=data_set,
              data_loader=data_loader,
              test_result_output_func=test_result_output,
              cuda_mode=1,
              print_inter=2,
              val_inter=50,
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
        dissimilarities = F.pairwise_distance(*model_output)
        dissimilarities = torch.squeeze(dissimilarities).item()

        if target_output == 1.0:
            diff_arg.append(dissimilarities)
        elif target_output == 0.0:
            same_arg.append(dissimilarities)

    same_arg = np.array(same_arg)
    diff_arg = np.array(diff_arg)

    diff_min = np.min(diff_arg)
    diff_max = np.max(diff_arg)
    diff_var = np.var(diff_arg)[0]

    same_max = np.max(same_arg)
    same_min = np.min(same_arg)
    same_var = np.var(same_arg)[0]

    same_arg = np.mean(same_arg, axis=-1)[0]
    diff_arg = np.mean(diff_arg, axis=-1)[0]
    print("****************************")
    print("epoch: %s\nloss: %s\nprogress: %.2f lr: %f" %
          (epoch, loss.data.float()[0], 100 * epoch / EPOCH, LEARNING_RATE))
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
