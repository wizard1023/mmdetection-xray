import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from mmdet.models.backbones import ResNet
from mmdet.registry import MODELS
from mmdet.models.backbones.LatentGNN.LatentGNN import LatentGNNV1
@MODELS.register_module()
class ResNet_LatenGNN(ResNet):
    def __init__(self, **kwargs):
        super(ResNet_LatenGNN, self).__init__(**kwargs)
        if self.depth in (18, 34):
            self.dims = (64, 128, 256, 512)
        elif self.depth in (50, 101, 152):
            self.dims = (256, 512, 1024, 2048)
        else:
            raise Exception()

        self.latentgnn = LatentGNNV1(in_channels=self.dims[2], latent_dims=[100,100], channel_stride=8,
                                  num_kernels=2)

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in [2]:
                x = self.latentgnn(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


if __name__ == "__main__":
    # network = LatentGNNV1(in_channels=1024,
    #                       latent_dims=[100, 100],
    #                       channel_stride=8,
    #                       num_kernels=2,
    #                       mode='asymmetric',
    #                       graph_conv_flag=False)
    # dump_inputs = torch.rand((8, 1024, 30, 30))
    # print(str(network))
    # output = network(dump_inputs)
    # print(output.shape)
    from mmdet.models import ResNet_LatenGNN
    import torch

    self = ResNet_LatenGNN(depth=50)
    self.eval()
    inputs = torch.rand(1, 3, 32, 32)
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))