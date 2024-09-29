# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.model import BaseModule
from mmdet.registry import MODELS

import torch
import torch.nn as nn


@ MODELS.register_module()
class FPN_LIM(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels=256,
                 num_outs=5,

                 ):
        super(FPN_LIM, self).__init__()
        C2_size, C3_size, C4_size, C5_size = in_channels
        num_out = num_outs
        feature_size = out_channels
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_1_dual = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_1_dual = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_upsampled_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_1_dual = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C2
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_1_dual = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        # self.P7_1 = nn.ReLU()
        # self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        # self.conv_to_1 = nn.Conv2d(feature_size, 16, kernel_size=1)
        # self.conv_resume = nn.Conv2d(16, feature_size, kernel_size=1)

        self.corner_proc_C2 = Boundary_Aggregation(C2_size)
        self.corner_proc_C3 = Boundary_Aggregation(C3_size)
        self.corner_proc_C4 = Boundary_Aggregation(C4_size)


        self.conv_p3_to_p4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.conv_p2_to_p3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.conv_p2_to_p4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.gamma_2 = nn.Parameter(torch.zeros(1))
        self.gamma_3 = nn.Parameter(torch.zeros(1))
        self.gamma_4 = nn.Parameter(torch.zeros(1))
        self.gamma_5 = nn.Parameter(torch.zeros(1))

        self.L2Norm2 = L2Norm(C2_size, 80)
        self.L2Norm3 = L2Norm(C3_size, 40)
        self.L2Norm1 = L2Norm(C4_size, 20)

    def forward(self, inputs):
        # global count
        C2, C3, C4, C5 = inputs  # 150，75,38,19
        C2 = self.L2Norm2(C2)
        C3 = self.L2Norm3(C3)
        C4 = self.L2Norm1(C4)

        C2_BA = self.corner_proc_C2(C2)
        C3_BA = self.corner_proc_C3(C3)
        C4_BA = self.corner_proc_C4(C4)

        P5_x = self.P5_1(C5)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P4_x  # 38
        P4_upsampled_x_1 = self.P4_upsampled_1(P4_x) # 75
        P4_upsampled_x_2 = self.P4_upsampled_2(P4_upsampled_x_1)#150
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x_1  # 75
        P3_upsampled_x_1 = self.P3_upsampled(P3_x)  # 150
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x_1 + P4_upsampled_x_2 # 75
        P2_x = self.P2_2(P2_x)

        P2_dual = self.P2_1_dual(C2_BA)
        P2_downsample_x_1 = self.conv_p2_to_p3(P2_dual)
        P2_downsample_x_2 = self.conv_p2_to_p4(P2_downsample_x_1)


        P3_dual = self.P3_1_dual(C3_BA)
        P3_dual = P3_dual + P2_downsample_x_1
        P3_downsample_x_1 = self.conv_p3_to_p4(P3_dual)

        P4_dual = self.P4_1_dual(C4_BA) + P3_downsample_x_1 + P2_downsample_x_2

        P5_dual = self.P5_1_dual(C5)

        P6_x = self.P6(C5)  # 10

        O2_x = self.gamma_2 * P2_dual + (1 - self.gamma_2) * P2_x
        O3_x = self.gamma_3 * P3_dual + (1 - self.gamma_3) * P3_x
        O4_x = self.gamma_4 * P4_dual + (1 - self.gamma_4) * P4_x
        O5_x = self.gamma_5 * P5_dual + (1 - self.gamma_5) * P5_x

        return (O2_x, O3_x, O4_x, O5_x, P6_x)



class Boundary_Aggregation(nn.Module):
    def __init__(self, in_channels):
        super(Boundary_Aggregation, self).__init__()
        self.conv = nn.Conv2d(in_channels * 5, in_channels, 1)

    def forward(self, x_batch: torch.tensor):
        in_channels, height, width = x_batch.size()[1:4]
        x_clk_rot = torch.rot90(x_batch, -1, [2, 3])
        x1 = self.up_to_bottom(x_batch, in_channels, height)
        x2 = self.bottom_to_up(x_batch, in_channels, height)
        x3 = self.left_to_right(x_clk_rot, in_channels, width)
        x4 = self.right_to_left(x_clk_rot, in_channels, width)
        x_con = torch.cat((x_batch, x1, x2, x3, x4), 1)
        x_merge = self.conv(x_con)
        return x_merge

    def left_to_right(self, x_clk_rot: torch.tensor, in_channels: int,
                      height: int):
        x = torch.clone(x_clk_rot)
        x = self.up_to_bottom(x, in_channels, height)
        x = torch.rot90(x, 1, [2, 3])
        return x

    def right_to_left(self, x_clk_rot: torch.tensor, in_channels: int,
                      height: int):
        x = torch.clone(x_clk_rot)
        x = self.bottom_to_up(x, in_channels, height)
        x = torch.rot90(x, 1, [2, 3])
        return x

    def bottom_to_up(self, x_raw: torch.tensor, in_channels: int, height: int):
        x = torch.clone(x_raw)
        for i in range(height - 1, -1, -1):
            x[:, :, i] = torch.max(x[:, :, i:], 2, True)[0].squeeze(2)
        return x

    def up_to_bottom(self, x_raw: torch.tensor, in_channels: int, height: int):
        x = torch.clone(x_raw)
        for i in range(height):
            x[:, :, i] = torch.max(x[:, :, :i + 1], 2, True)[0].squeeze(2)
        return x


class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


# if __name__ == '__main__':
#     lim = SSDLIMNeck(256, 512, 1024, 2048)
#     x = [torch.randn(3, 256, 340, 340), torch.randn(3, 512, 170, 170), torch.randn(3, 1024, 85, 85), torch.randn(3, 2048, 43, 43)]
#     y = lim(x)
#
#     print(y[0].shape, y[1].shape, y[2].shape, y[3].shape, y[4].shape)
