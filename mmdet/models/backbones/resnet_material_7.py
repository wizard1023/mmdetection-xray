import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from mmdet.models.backbones import ResNet
from mmdet.registry import MODELS
from mmdet.models.backbones.LatentGNN.Material_Guidance_7 import Material_Guidance_7

@MODELS.register_module()
class ResNet_Material_7(ResNet):
    def __init__(self, **kwargs):
        super(ResNet_Material_7, self).__init__(**kwargs)
        self.material_guidance = Material_Guidance_7()

    def forward(self, x):
        origin_img = x
        #print(x.shape)
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
            if i in self.out_indices:
                outs.append(x)
        outs = self.material_guidance(origin_img, outs)
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
    import torch

    self = ResNet_Material_7(depth=50)
    self.eval()
    inputs = torch.rand(2, 3, 640, 640)
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))