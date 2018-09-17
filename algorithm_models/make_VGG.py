# coding:utf-8
import torch.nn as nn


class VGGBlock(nn.Module):
    def __init__(self, input_chnl, output_chnl, layers):
        super(VGGBlock, self).__init__()
        layers_to_add = []
        for each in range(layers):

            if each == 0:
                tmp_in = input_chnl
            else:
                tmp_in = output_chnl

            layer = nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv1d(
                    in_channels=tmp_in,
                    out_channels=output_chnl,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.BatchNorm1d(output_chnl)
            )
            layers_to_add.append(layer)

            if each == layers - 1:
                layers_to_add.append(nn.MaxPool1d(
                    kernel_size=3,
                    stride=2,
                ))

        self.block = nn.Sequential(
            *layers_to_add
        )

    def forward(self, x):
        x = self.block(x)
        return x


class VGGNet(nn.Module):
    def __init__(self, layers, layer_chnl, input_plane):
        """
        生成VGGNet
        :param block: 用什么样的block？
        :param layers: block里面放几层？
        :param layer_chnl: the output channel in each block
        """

        if len(layers) != len(layer_chnl):
            raise Exception('the length of layers cnt args and planes args should same')
        super(VGGNet, self).__init__()

        block_to_add = []
        for each in range(len(layers)):
            if each == 0:
                input_chnl = input_plane
            else:
                input_chnl = layer_chnl[each - 1]
            block_to_add.append(self.__make_layer(input_chnl, layer_chnl[each], layers[each]))
        self.blocks = nn.Sequential(*block_to_add)
        self.out = nn.AdaptiveAvgPool1d(2)

    def forward(self, x):
        x = self.blocks(x)
        x = self.out(x)
        x = x.view(x.size(0), -1)
        return x

    @staticmethod
    def __make_layer(input_plane, output_plane, layers):
        return VGGBlock(input_chnl=input_plane, output_chnl=output_plane, layers=layers)


def make_vgg(input_chnl, layers, layers_chnl):
    return VGGNet(input_plane=input_chnl, layers=layers, layer_chnl=layers_chnl)
