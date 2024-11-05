import torch.nn as nn

import torch

import torch.nn.functional as F

from mmdet.models.backbones.LatentGNN.LatentGNN_ch import LatentGNNV1_ch

class Multiscale_16(nn.Module):
    def __init__(self):
        super(Multiscale_16, self).__init__()

        # for level_3
        self.dilated_conv_3_1 = nn.Conv2d(256, 256,kernel_size=3,stride=2,padding=2,dilation=2)
        self.dilated_conv_3_2 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=2, dilation=2)
        self.dilated_conv_3_3 = nn.Conv2d(1024, 256, kernel_size=3, stride=2, padding=2, dilation=2)
        self.conv_11_3 = nn.Conv2d(2048,256, kernel_size=1)

        # self.weight_3_0 = nn.Conv2d(2048,8,kernel_size=1)
        # self.weight_3_1 = nn.Conv2d(2048,8,kernel_size=1)
        # self.weight_3_2 = nn.Conv2d(2048,8,kernel_size=1)
        # self.weight_3_3 = nn.Conv2d(2048,8,kernel_size=1)

        # self.weight_levels_3 = nn.Conv2d(8 * 4, 4, kernel_size=1, stride=1, padding=0)

        # for level_2
        # self.conv_1 = nn.Conv2d(2048,1024,kernel_size=1)
        self.dilated_conv_2_1 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=2, dilation=2)
        self.dilated_conv_2_2 = nn.Conv2d(256,256, kernel_size=3, stride=2, padding=2, dilation=2)

        self.upsample_2 = nn.ConvTranspose2d(2048,256,kernel_size=4,stride=2,padding=1)

        self.conv_11_2 = nn.Conv2d(1024, 256, kernel_size=1)
        #
        # self.weight_2_0 = nn.Conv2d(1024, 8, kernel_size=1)
        # self.weight_2_1 = nn.Conv2d(1024, 8, kernel_size=1)
        # self.weight_2_2 = nn.Conv2d(1024, 8, kernel_size=1)
        # self.weight_2_3 = nn.Conv2d(1024, 8, kernel_size=1)
        #
        # self.weight_levels_2 = nn.Conv2d(8 * 4, 4, kernel_size=1, stride=1, padding=0)

        # for level_1
        # self.conv_2 = nn.Conv2d(2048,512,kernel_size=1)
        # self.conv_3 = nn.Conv2d(1024, 512, kernel_size=1)
        self.dilated_conv_1_1 = nn.Conv2d(256,256, kernel_size=3, stride=2, padding=2, dilation=2)

        self.upsample_1_0 = nn.ConvTranspose2d(2048,256,kernel_size=8,stride=4,padding=2)
        self.upsample_1_1 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)

        self.conv_11_1 = nn.Conv2d(512, 256, kernel_size=1)

        # self.weight_1_0 = nn.Conv2d(512, 8, kernel_size=1)
        # self.weight_1_1 = nn.Conv2d(512, 8, kernel_size=1)
        # self.weight_1_2 = nn.Conv2d(512, 8, kernel_size=1)
        # self.weight_1_3 = nn.Conv2d(512, 8, kernel_size=1)
        #
        # self.weight_levels_1 = nn.Conv2d(8 * 4, 4, kernel_size=1, stride=1, padding=0)

        # for level_0
        # self.conv_4 = nn.Conv2d(2048,256,kernel_size=1)
        # self.conv_5 = nn.Conv2d(1024,256,kernel_size=1)
        # self.conv_6 = nn.Conv2d(512,256,kernel_size=1)

        # self.weight_0_0 = nn.Conv2d(256, 8, kernel_size=1)
        # self.weight_0_1 = nn.Conv2d(256, 8, kernel_size=1)
        # self.weight_0_2 = nn.Conv2d(256, 8, kernel_size=1)
        # self.weight_0_3 = nn.Conv2d(256, 8, kernel_size=1)

        self.upsample_0_0 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1)
        self.upsample_0_01 = nn.ConvTranspose2d(256, 256, kernel_size=8, stride=4, padding=2)
        self.upsample_0_1 = nn.ConvTranspose2d(1024, 256, kernel_size=8, stride=4, padding=2)
        self.upsample_0_2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)

        self.conv_11_0 = nn.Conv2d(256, 256, kernel_size=1)

        # self.weight_levels_0 = nn.Conv2d(8 * 4, 4, kernel_size=1, stride=1, padding=0)

        self.share_fuse = nn.Conv2d(1024,512,kernel_size=1)

        self.conv_11_up_3 = nn.Conv2d(512,2048,kernel_size=1)
        self.conv_11_up_2 = nn.Conv2d(512, 1024, kernel_size=1)
        self.conv_11_up_1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv_11_up_0 = nn.Conv2d(512, 256, kernel_size=1)


        self.maxpool1 = nn.AdaptiveMaxPool2d(output_size=(320, 200))
        self.latent_gnn_ch1 = LatentGNNV1_ch(
            in_channels=512,
            in_spatial=160 * 100,
            latent_dims=[100, 100],
            num_kernels=2,
            mode='asymmetric',
            graph_conv_flag=False)

        self.maxpool2 = nn.AdaptiveMaxPool2d(output_size=(160, 100))
        self.latent_gnn_ch2 = LatentGNNV1_ch(
            in_channels=512,
            in_spatial=80 * 50,
            latent_dims=[100, 100],
            num_kernels=2,
            mode='asymmetric',
            graph_conv_flag=False)

        self.maxpool3 = nn.AdaptiveMaxPool2d(output_size=(80, 50))
        self.latent_gnn_ch3 = LatentGNNV1_ch(
            in_channels=512,
            in_spatial=40 * 25,
            latent_dims=[100, 100],
            num_kernels=2,
            mode='asymmetric',
            graph_conv_flag=False)

        self.maxpool4 = nn.AdaptiveMaxPool2d(output_size=(40, 25))
        self.latent_gnn_ch4 = LatentGNNV1_ch(
            in_channels=512,
            in_spatial=20 * 13,
            latent_dims=[100, 100],
            num_kernels=2,
            mode='asymmetric',
            graph_conv_flag=False)



    def forward(self,feat):
        level_0,level_1,level_2,level_3 = feat[0],feat[1],feat[2],feat[3]

        # for level_3
        level_3_d = self.conv_11_3(level_3)
        level_2_d_3 = self.dilated_conv_3_3(level_2)
        level_1_d_3 = self.dilated_conv_3_2(F.max_pool2d(level_1, 3, stride=2, padding=1))
        level_0_d_3 = self.dilated_conv_3_1(F.max_pool2d(level_0, 4, stride=4, padding=1))
        fused_out_3 = torch.cat((level_0_d_3 ,level_1_d_3 ,level_2_d_3 ,level_3_d),dim=1)
        fused_out_3 = self.share_fuse(fused_out_3)
        fused_out_3_pool = self.maxpool4(fused_out_3)
        fused_out_3_att = self.conv_11_up_3(self.latent_gnn_ch4(fused_out_3_pool)*fused_out_3)+level_3


        # level_2_d_3_w = self.weight_3_0(level_2_d_3)
        # level_1_d_3_w = self.weight_3_1(level_1_d_3)
        # level_0_d_3_w = self.weight_3_2(level_0_d_3)
        # level_3_w = self.weight_3_3(level_3)
        # levels_3_w_cat = torch.cat((level_2_d_3_w, level_1_d_3_w, level_0_d_3_w,level_3_w), 1)
        # levels_3_weights = self.weight_levels_3(levels_3_w_cat)
        # levels_3_weights = F.softmax(levels_3_weights, dim=1)
        # fused_out_3 = level_0_d_3 * levels_3_weights[:, 0:1, :, :] + \
        #                     level_1_d_3 * levels_3_weights[:, 1:2, :, :] + \
        #                     level_2_d_3 * levels_3_weights[:, 2:3, :, :] + \
        #                     level_3 * levels_3_weights[:, 3:, :, :]


        # for level_2
        # level_3_u_2 = F.interpolate(self.conv_1(level_3), scale_factor=2, mode='nearest')
        level_2_d = self.conv_11_2(level_2)
        level_3_u_2 = self.upsample_2(level_3)
        level_1_d_2 = self.dilated_conv_2_1(level_1)
        level_0_d_2 = self.dilated_conv_2_2(F.max_pool2d(level_0, 3, stride=2, padding=1))
        fused_out_2 = torch.cat((level_3_u_2 ,level_1_d_2 ,level_0_d_2 ,level_2_d),dim=1)
        fused_out_2 = self.share_fuse(fused_out_2)
        fused_out_2_pool = self.maxpool3(fused_out_2)
        fused_out_2_att = self.conv_11_up_2(self.latent_gnn_ch3(fused_out_2_pool) * fused_out_2) + level_2

        # level_3_u_2_w = self.weight_2_0(level_3_u_2)
        # level_1_d_2_w = self.weight_2_1(level_1_d_2)
        # level_0_d_2_w = self.weight_2_2(level_0_d_2)
        # level_2_w = self.weight_2_3(level_2)
        # levels_2_w_cat = torch.cat((level_3_u_2_w, level_1_d_2_w, level_0_d_2_w, level_2_w), 1)
        # levels_2_weights = self.weight_levels_2(levels_2_w_cat)
        # levels_2_weights = F.softmax(levels_2_weights, dim=1)
        # fused_out_2 = level_3_u_2 * levels_2_weights[:, 0:1, :, :] + \
        #               level_1_d_2 * levels_2_weights[:, 1:2, :, :] + \
        #               level_0_d_2 * levels_2_weights[:, 2:3, :, :] + \
        #               level_2 * levels_2_weights[:, 3:, :, :]

        # for level_1
        # level_3_u_1 = F.interpolate(self.conv_2(level_3), scale_factor=4, mode='nearest')
        # level_2_u_1 = F.interpolate(self.conv_3(level_2), scale_factor=2, mode='nearest')
        level_1_d = self.conv_11_1(level_1)
        level_3_u_1 = self.upsample_1_0(level_3)
        level_2_u_1 = self.upsample_1_1(level_2)
        level_0_d_1 = self.dilated_conv_1_1(level_0)
        fused_out_1 = torch.cat((level_3_u_1 ,level_2_u_1 ,level_0_d_1 ,level_1_d),dim=1)
        fused_out_1 = self.share_fuse(fused_out_1)
        fused_out_1_pool = self.maxpool2(fused_out_1)
        fused_out_1_att = self.conv_11_up_1(self.latent_gnn_ch2(fused_out_1_pool) * fused_out_1) + level_1

        # level_3_u_1_w = self.weight_1_0(level_3_u_1)
        # level_2_u_1_w = self.weight_1_1(level_2_u_1)
        # level_0_d_1_w = self.weight_1_2(level_0_d_1)
        # level_1_w = self.weight_1_3(level_1)
        # levels_1_w_cat = torch.cat((level_3_u_1_w, level_2_u_1_w, level_0_d_1_w, level_1_w), 1)
        # levels_1_weights = self.weight_levels_1(levels_1_w_cat)
        # levels_1_weights = F.softmax(levels_1_weights, dim=1)
        # fused_out_1 = level_3_u_1 * levels_1_weights[:, 0:1, :, :] + \
        #               level_2_u_1 * levels_1_weights[:, 1:2, :, :] + \
        #               level_0_d_1 * levels_1_weights[:, 2:3, :, :] + \
        #               level_1 * levels_1_weights[:, 3:, :, :]

        # for level_0
        # level_3_u_0 = F.interpolate(self.conv_4(level_3), scale_factor=8, mode='nearest')
        # level_2_u_0 = F.interpolate(self.conv_5(level_2), scale_factor=4, mode='nearest')
        # level_1_u_0 = F.interpolate(self.conv_6(level_1), scale_factor=2, mode='nearest')
        level_0_d = self.conv_11_0(level_0)
        level_3_u_0 = self.upsample_0_01(self.upsample_0_0(level_3))
        level_2_u_0 = self.upsample_0_1(level_2)
        level_1_u_0 = self.upsample_0_2(level_1)
        fused_out_0 = torch.cat((level_3_u_0,level_2_u_0,level_1_u_0,level_0_d),dim=1)
        fused_out_0 = self.share_fuse(fused_out_0)
        fused_out_0_pool = self.maxpool1(fused_out_0)
        fused_out_0_att = self.conv_11_up_0(self.latent_gnn_ch1(fused_out_0_pool) * fused_out_0) + level_0

        # level_3_u_0_w = self.weight_0_0(level_3_u_0)
        # level_2_u_0_w = self.weight_0_1(level_2_u_0)
        # level_1_u_0_w = self.weight_0_2(level_1_u_0)
        # level_0_w = self.weight_0_3(level_0)
        # levels_0_w_cat = torch.cat((level_3_u_0_w, level_2_u_0_w, level_1_u_0_w, level_0_w), 1)
        # levels_0_weights = self.weight_levels_0(levels_0_w_cat)
        # levels_0_weights = F.softmax(levels_0_weights, dim=1)
        # fused_out_0 = level_3_u_0 * levels_0_weights[:, 0:1, :, :] + \
        #               level_2_u_0 * levels_0_weights[:, 1:2, :, :] + \
        #               level_1_u_0 * levels_0_weights[:, 2:3, :, :] + \
        #               level_0 * levels_0_weights[:, 3:, :, :]

        outs = []
        outs.append(fused_out_0_att)
        outs.append(fused_out_1_att)
        outs.append(fused_out_2_att)
        outs.append(fused_out_3_att)

        # fusion weight

        # print(level_2_d_3.shape)
        # print(level_1_d_3.shape)
        # print(level_0_d_3.shape)
        # print(level_3_u_2.shape)
        # print(level_1_d_2.shape)
        # print(level_0_d_2.shape)
        # print(level_3_u_1.shape)
        # print(level_2_u_1.shape)
        # print(level_0_d_1.shape)
        # print(level_3_u_0.shape)
        # print(level_2_u_0.shape)
        # print(level_1_u_0.shape)

        return outs


if __name__=='__main__':
    img = torch.rand(2, 3, 640, 640).to("cuda:2")
    feat = [torch.rand(1, 256, 160, 160).to("cuda:2"),
            torch.rand(1, 512, 80, 80).to("cuda:2"),
            torch.rand(1, 1024, 40, 40).to("cuda:2"),
            torch.rand(1, 2048, 20, 20).to("cuda:2")]
    model = Multiscale_16().to("cuda:2")
    outs = model(feat)
    print([i.shape for i in outs])

