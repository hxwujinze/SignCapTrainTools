# coding:utf-8
import math

import torch.nn as nn


def conv_filter3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)  # 这里不需要 bias ？？


class BasicBlock(nn.Module):
    """
    resnet 的基本块
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        生成基本块
        :param inplanes: 输入的通道
        :param planes: 输出的通道
        :param stride: filter 步长 设成1 就行
        :param downsample: identity map是否进行 downsample
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv_filter3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_filter3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        """
        可以看到 是由两个卷积加bn作为运算的F(x), downsample 作为identity mapping        
        """

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, layer_planes):
        """
        生成resnet
        :param block: 用什么样的block？
        :param layers: block里面放几层？
        :param layer_planes: the output channel in each block
        :param num_classes: 分多少类别？
        """

        if len(layers) != len(layer_planes):
            raise Exception('the length of layers cnt args and planes args should same')

        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv1d(14, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        block_to_add = []
        for each in range(len(layers)):
            if each == 0:
                block_to_add.append(self._make_layer(block, 64, layers[each]))
            else:
                block_to_add.append(self._make_layer(block, layer_planes[each], layers[each]))
        self.blocks = nn.Sequential(*block_to_add)

        self.avgpool = nn.AdaptiveAvgPool1d(2)

        for m in self.modules():
            # 对中间的conv 和 BN层进行初始化
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        # 如果下一层的输入与上一层不同了 使用downsample 进行调整
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.blocks(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def my_resnet(layers, layer_planes):
    """
    make a resnet model
    :param layers: the layers amount in each block
    :param layer_planes: the output channel in each block
    :return:
    """
    model = ResNet(BasicBlock, layers, layer_planes)
    return model
