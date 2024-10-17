import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from mmdet.models.backbones import ResNet
from mmdet.registry import MODELS
from mmdet.models.backbones.LatentGNN.Density_Guidance_add import Density_Guidance_add

@MODELS.register_module()
class ResNet_hsv_cat(ResNet):
    def __init__(self, **kwargs):
        super(ResNet_hsv_cat, self).__init__(**kwargs)
        # self.material_guidance = Density_Guidance_add()
        self.conv = nn.Conv2d(6, 3, kernel_size=1, stride=1, padding=0)

    def rgb2hsv_torch(self, rgb: torch.Tensor):
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
        cmax_idx[delta == 0] = 3
        hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsv_h[cmax_idx == 3] = 0.
        hsv_h /= 6.
        hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
        hsv_v = cmax
        return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

    def forward(self, x):
        # origin_img = x
        hsv_x = self.rgb2hsv_torch(x)
        x = torch.cat((x, hsv_x), dim=1)
        x = self.conv(x)
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
        # outs = self.material_guidance(origin_img, outs)
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