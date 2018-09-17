# Siamese-Networks
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .make_resnet import my_resnet
from .make_VGG import make_vgg

# CNN: input len -> output len
# Lout=floor((Lin+2∗padding−dilation∗(kernel_size−1)−1)/stride+1)


WEIGHT_DECAY = 0.000002
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
EPOCH = 250


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

        # self.coding_model = my_resnet(layers=[2 ,2], layer_planes=[64, 128])
        # self.coding_model = load_model_from_classify()
        self.coding_model = make_vgg(input_chnl=14, layers=[2, 3], layers_chnl=[64, 128])

        self.out = torch.nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward_once(self, x):
        x = self.coding_model(x)
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

    def train(self, mode=True):
        nn.Module.train(self, mode)
        self.status = 'train'

    def single_output(self):
        self.status = 'eval'

    def exc_train(self):
        # only import train staff in training env
        from train_util.data_set import generate_data_set, SiameseNetworkTrainDataSet
        from train_util.common_train import train
        from torch.utils.data import dataloader as DataLoader
        print("verify model start training")
        print(str(self))

        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        loss_func = ContrastiveLoss()
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)
        data_set = generate_data_set(0.06, SiameseNetworkTrainDataSet)
        data_loader = {
            'train': DataLoader.DataLoader(data_set['train'],
                                           shuffle=True,
                                           batch_size=BATCH_SIZE,
                                           num_workers=1),
            'test': DataLoader.DataLoader(data_set['test'],
                                          shuffle=True,
                                          batch_size=1, )
        }
        train(model=self,
              model_name='verify_68',
              EPOCH=EPOCH,
              optimizer=optimizer,
              exp_lr_scheduler=lr_scheduler,
              loss_func=loss_func,
              save_dir='./params',
              data_set=data_set,
              data_loader=data_loader,
              test_result_output_func=test_result_output,
              cuda_mode=1,
              print_inter=2,
              val_inter=25,
              scheduler_step_inter=50
              )


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
    diff_var = np.var(diff_arg)
    diff_1st = np.percentile(diff_arg, 10)
    diff_med = np.percentile(diff_arg, 50)
    diff_2nd = np.percentile(diff_arg, 90)

    same_max = np.max(same_arg)
    same_min = np.min(same_arg)
    same_var = np.var(same_arg)
    same_1st = np.percentile(same_arg, 10)
    same_med = np.percentile(same_arg, 50)
    same_2nd = np.percentile(same_arg, 90)

    same_arg = np.mean(same_arg, axis=-1)
    diff_arg = np.mean(diff_arg, axis=-1)
    diff_res = "****************************"
    diff_res += "epoch: %s\nloss: %s\nprogress: %.2f lr: %f\n" % \
                (epoch, loss, 100 * epoch / EPOCH, LEARNING_RATE)
    diff_res += "diff info \n    max: %f min: %f, mean: %f var: %f\n " % \
                (diff_max, diff_min, diff_arg, diff_var) \
                + "    1st: %f med: %f 2nd: %f\n" % (diff_1st, diff_med, diff_2nd) \
                + "same info\n    max: %f min: %f, mean: %f, same_var %f\n" % \
                (same_max, same_min, same_arg, same_var) \
                + "    1st: %f med: %f 2nd: %f" % (same_1st, same_med, same_2nd)
    print(diff_res)
    return diff_res


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
