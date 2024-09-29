import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
#from mmcv.runner import BaseModule
from mmengine.model import BaseModule
from mmdet.models.necks.ssd_neck import SSDNeck
import torch.nn.init as init
#from ..builder import NECKS
from mmdet.registry import MODELS
@MODELS.register_module()
#@NECKS.register_module()
class SSDLIMNeck(SSDNeck):
    def __init__(self, in_channels, out_channels=256,
                 level_strides=(2, 2, 1, 1),
                 level_paddings=(1, 1, 0, 0),
                 l2_norm_scale=20
                 ):
        super(SSDLIMNeck, self).__init__(
            in_channels=(512, 1024),
            out_channels=(512, 1024, 512, 256, 256, 256),
            level_strides=(2, 2, 1, 1),
            level_paddings=(1, 1, 0, 0),
            l2_norm_scale=20)
        C3_size, C4_size, C5_size = in_channels
        feature_size = out_channels
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_1_dual = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled_1 = nn.Upsample(scale_factor=2, mode='nearest')
        #self.P5_upsampled_2 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P5_upsampled_3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_1_dual = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled_1 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_upsampled_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_1_dual = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        #         self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C2
        # self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        # self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        #
        # # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        # self.P7_1 = nn.ReLU()
        # self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        # self.conv_to_1 = nn.Conv2d(feature_size, 16, kernel_size=1)
        # self.conv_resume = nn.Conv2d(16, feature_size, kernel_size=1)

        self.corner_proc_C3 = Boundary_Aggregation(256)
        self.corner_proc_C4 = Boundary_Aggregation(512)
        self.corner_proc_C5 = Boundary_Aggregation(1024)

        self.conv_p3_to_p4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv_p4_to_p5 = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.conv_p3_to_p5 = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0)

        # self.sig_3 = nn.Sigmoid()
        # self.sig_4 = nn.Sigmoid()
        # self.sig_5 = nn.Sigmoid()
        #self.gamma_3 = nn.Parameter(torch.zeros(1))
        self.gamma_4 = nn.Parameter(torch.zeros(1))
        self.gamma_5 = nn.Parameter(torch.zeros(1))

        # self.conv_down_3 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        # self.conv_down_4 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        # self.conv_down_5 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.conv_upch_4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.conv_upch_5 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)

        self.L2Norm3 = L2Norm(256, 40)
        self.L2Norm1 = L2Norm(512, 20)

    def forward(self, inputs):
        # global count
        C3, C4, C5 = inputs  # 150，75,38,19
        C3 = self.L2Norm3(C3)
        C4 = self.L2Norm1(C4)
        print(C4.shape)
        print(C5.shape)
        print('------')


        C3_BA = self.corner_proc_C3(C3)
        C4_BA = self.corner_proc_C4(C4)
        C5_BA = self.corner_proc_C5(C5)

        P5_x = self.P5_1(C5)
        P5_upsampled_x_1 = self.P5_upsampled_1(P5_x)  # 38
        #P5_upsampled_x_2 = self.P5_upsampled_2(P5_upsampled_x_1)[:, :, :-1, :-1]  # 75
        # P5_upsampled_x_3 = self.P5_upsampled_3(P5_upsampled_x_2)#150
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        print('P4_x,P5_up')
        print(P4_x.shape)
        print(P5_upsampled_x_1.shape)
        P4_x = P4_x + P5_upsampled_x_1  # 38
        #P4_upsampled_x_1 = self.P4_upsampled_1(P4_x)[:, :, :-1, :-1]  # 75
        # P4_upsampled_x_2 = self.P4_upsampled_2(P4_upsampled_x_1)#150
        P4_x = self.P4_2(P4_x)

        # P3_x = self.P3_1(C3)
        # P3_x = P3_x + P4_upsampled_x_1 + P5_upsampled_x_2  # 75
        # # P3_upsampled_x_1 = self.P3_upsampled(P3_x)  # 150
        # P3_x = self.P3_2(P3_x)

        P3_dual = self.P3_1_dual(C3_BA)
        P3_downsample_x_1 = self.conv_p3_to_p4(P3_dual)
        P3_downsample_x_2 = self.conv_p3_to_p5(P3_downsample_x_1)

        P4_dual = self.P4_1_dual(C4_BA)
        P4_dual = P4_dual + P3_downsample_x_1
        P4_downsample_x_1 = self.conv_p4_to_p5(P4_dual)

        P5_dual = self.P5_1_dual(C5_BA)
        P5_dual = P5_dual + P4_downsample_x_1 + P3_downsample_x_2

        # P2_x = self.P2_1(C2)
        # P2_x = P2_x + P3_upsampled_x_1 +  P4_upsampled_x_2 + P5_upsampled_x_3# 75
        # P2_x = self.P2_2(P2_x)

        # P6_x = self.P6(C5)  # 10
        #
        # P7_x = self.P7_1(P6_x)
        # P7_x = self.P7_2(P7_x)  # 5

        # print(P3_dual.shape)
        # img = P3_dual[0][0].cpu().detach().numpy()
        # for i in range(1,P3_dual.shape[1]):
        #    img = img + P3_dual[0][i].cpu().detach().numpy()
        # print('img_shape',img.shape)
        # cv2.imwrite('test/test'+str(count)+'.jpg',img * 255)
        # time.sleep(10)
        #O3_x = self.gamma_3 * P3_dual + (1 - self.gamma_3) * P3_x
        O4_x = self.gamma_4 * P4_dual + (1 - self.gamma_4) * P4_x
        O5_x = self.gamma_5 * P5_dual + (1 - self.gamma_5) * P5_x

        O4_x = self.conv_upch_4(O4_x)
        O5_x = self.conv_upch_5(O5_x)

        outs = [O4_x,O5_x]
        #return (O3_x, O4_x, O5_x, P6_x, P7_x)
        if hasattr(self, 'l2_norm'):
            outs[0] = self.l2_norm(outs[0])

        feat = outs[-1]
        for layer in self.extra_layers:
            feat = layer(feat)
            outs.append(feat)
        return tuple(outs)
        #return O4_x, O5_x



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


# class L2Norm(nn.Module):
#
#     def __init__(self, n_dims, scale=20., eps=1e-10):
#         """L2 normalization layer.
#
#         Args:
#             n_dims (int): Number of dimensions to be normalized
#             scale (float, optional): Defaults to 20..
#             eps (float, optional): Used to avoid division by zero.
#                 Defaults to 1e-10.
#         """
#         super(L2Norm, self).__init__()
#         self.n_dims = n_dims
#         self.weight = nn.Parameter(torch.Tensor(self.n_dims))
#         self.eps = eps
#         self.scale = scale
#
#     def forward(self, x):
#         """Forward function."""
#         # normalization layer convert to FP32 in FP16 training
#         x_float = x.float()
#         norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
#         return (self.weight[None, :, None, None].float().expand_as(x_float) *
#                 x_float / norm).type_as(x)

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

if __name__=='__main__':
    self = SSDLIMNeck(in_channels=(256, 512, 1024),
        out_channels=256,
        level_strides=(2, 2, 1, 1),
        level_paddings=(1, 1, 0, 0),
        l2_norm_scale=20)
    inputs = (torch.rand(1, 256, 200, 292),
              torch.rand(1, 512, 100, 167),
              torch.rand(1, 1024, 50, 84))
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))