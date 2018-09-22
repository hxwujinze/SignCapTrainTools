# Siamese-Networks
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .make_resnet import my_resnet
from algorithm_models.make_VGG import make_vgg

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
        outs = []
        for each in xs:
            outs.append(self.forward_once(each))
        if len(outs) != 1:
            return tuple(outs)
        else:
            return outs[0]

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
        data_set = generate_data_set(0.06, SiameseNetworkTrainDataSet, 16)
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


class WeightBasedTripleLoss(nn.Module):

    def __init__(self, batch_k, margin=2.0):
        super(WeightBasedTripleLoss, self).__init__()
        self.sampling_layer = WeightSamplingLayer(batch_k)

    def forward(self, x):
        a_s, p_s, n_s = self.sampling_layer(x)
        loss = F.triplet_margin_loss(anchor=a_s, positive=p_s, negative=n_s)
        return loss


class WeightSamplingLayer:
    def __init__(self, batch_k, nonzero_loss_cutoff=1.4, maximum_cutoff=0.5):
        self.n = None
        self.d = None
        self.k = batch_k
        self.nonzero_loss_cutoff = nonzero_loss_cutoff
        self.maximum_cutoff = maximum_cutoff

    def __call__(self, x):
        """
        execute sampling according to the distance
        make sure the dtype is double, otherwise many the calculate error will accumulate
        :param x: 2D array (vectors, data in dim)
        :return: 3 2D tensor, contains anchor vector, positive vector, negative vector
        """
        # L2Normalize
        init_vectors = x
        x = WeightSamplingLayer.L2Normalize(x)
        self.n, self.d = x.shape
        # calculate distance
        distance = WeightSamplingLayer.get_distance(x)
        # mapping to distribution
        weights = self.calculate_weight(distance)
        # execute sampling
        return self.sampling(init_vectors, weights)
        # return triplets

    @staticmethod
    def L2Normalize(data):
        data = data.numpy()
        ret = None
        for i in range(len(data)):
            col = data[i,]
            col = col / (np.sqrt(np.sum(col ** 2)) + 0.00001)
            if ret is None:
                ret = col
            else:
                ret = np.vstack((ret, col))
        ret = torch.from_numpy(ret)
        return ret

    @staticmethod
    def get_distance(data):
        data = data.numpy()
        n = data.shape[0]
        square = np.sum(data ** 2.0, axis=1, keepdims=True)
        # print('square %s' % str(square))
        distance_square = square + square.T - (2.0 * np.dot(data, data.T))
        # print('distance_square %s' % str(distance_square))
        # Adding identity to make sqrt work.
        res = np.sqrt(distance_square + np.identity(n))
        return torch.from_numpy(res)

    def calculate_weight(self, distance):
        distance = distance.numpy()
        log_weights = ((2.0 - float(self.d)) * np.log(distance)
                       - (float(self.d - 3) / 2) * np.log(1.0 - 0.25 * (distance ** 2.0)))
        # use formula trans get the distribution score, according that score we use it as
        # the simulating distribution for sample the (a, p, n) triplet
        # print("log_weights %s" % str(log_weights))

        weights = np.exp(log_weights - np.max(log_weights))
        # use the softmax-like exp transform the score to the probabilities
        # Sample only negative examples by setting weights of
        # the same-class examples to 0.

        mask = np.ones(weights.shape)
        k = self.k
        for i in range(0, self.n, k):
            mask[i:i + k, i:i + k] = 0
        # print("weights before norm\n%s" % str(weights))
        weights = weights * np.array(mask) * (distance < self.nonzero_loss_cutoff)
        # mapping the mask to the matrix
        weights = weights / np.sum(weights, axis=1, keepdims=True)  # normalize
        return weights

    def sampling(self, vectors, weights):
        a_indices = []
        p_indices = []
        n_indices = []
        # encoding all the input to the vectors, but only output part of them for compute loss
        # the select method according to the weight based probabilities.

        n = len(vectors)
        print(n)
        for i in range(len(vectors)):
            block_idx = i // self.k  # which block that this data belong

            try:
                n_indices += np.random.choice(n, self.k - 1, p=weights[i]).tolist()
            except:
                n_indices += np.random.choice(n, self.k - 1).tolist()
            for j in range(block_idx * self.k, (block_idx + 1) * self.k):
                if j != i:
                    a_indices.append(i)
                    p_indices.append(j)

        # print("p_indices \n%s" % str(p_indices))
        # print("n_indices \n%s" % str(n_indices))
        # print('a_indices \n%s' % str(a_indices))
        return vectors[a_indices], vectors[p_indices], vectors[n_indices]


'''
import algorithm_models.verify_model as vm
from weight_based_sampling_demo import model as wm
import torch
import  numpy as np
import mxnet as mx
import mxnet.ndarray as nd
arr = torch.from_numpy(np.array([[1.,1.,1.,],
                                 [2.,2.,2.,],
                                 [3.,3.,3.,],
                                 [1.,2.,3.],
                                 [2.1,1.1,2.3],
                                 [1,2,2.1],
                                 [1.4,2.1,.6],
                                 [1.4,2.4,1.],
                                 [1.6,.9,1.],
                                 [1.44,2.1,.98],
                                 [.88,1.1,1.777],
                                 [.4,.9,1.1]])).double()
arr_mx = nd.array(arr)
vm.L2Normalize(arr)
wsl = wm.MarginSampler(3)
my_wsl = vm.WeightSamplingLayer(3)
'''
