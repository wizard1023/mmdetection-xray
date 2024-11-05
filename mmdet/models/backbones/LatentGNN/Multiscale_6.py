import torch.nn as nn

import torch

import torch.nn.functional as F
from mmdet.models.backbones.LatentGNN.ostu import otsu_threshold

class Multiscale_6(nn.Module):
    def __init__(self):
        super(Multiscale_6, self).__init__()

        # for level_3
        # self.dilated_conv_3_1 = nn.Sequential(
        #     nn.Conv2d(256, 2048,kernel_size=3,stride=2,padding=2,dilation=2),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU())
        # self.dilated_conv_3_2 = nn.Sequential(
        #     nn.Conv2d(512, 2048, kernel_size=3, stride=2, padding=2, dilation=2),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU())
        self.dilated_conv_3_3 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=2, dilation=2),
            nn.BatchNorm2d(2048),
            nn.ReLU())

        self.weight_3_0 = nn.Sequential(
            nn.Conv2d(2048,8,kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU())
        # self.weight_3_1 = nn.Sequential(
        #     nn.Conv2d(2048,8,kernel_size=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU())
        # self.weight_3_2 = nn.Sequential(
        #     nn.Conv2d(2048,8,kernel_size=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU())
        self.weight_3_3 = nn.Sequential(
            nn.Conv2d(2048,8,kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU())

        self.weight_levels_3 = nn.Conv2d(8 * 2, 2, kernel_size=1, stride=1, padding=0)

        # for level_2
        self.conv_1 = nn.Sequential(
            nn.Conv2d(2048,1024,kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU())
        self.dilated_conv_2_1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=2, dilation=2),
            nn.BatchNorm2d(1024),
            nn.ReLU())
        # self.dilated_conv_2_2 = nn.Sequential(
        #     nn.Conv2d(256,1024, kernel_size=3, stride=2, padding=2, dilation=2),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU())

        self.weight_2_0 = nn.Sequential(
            nn.Conv2d(1024, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.weight_2_1 = nn.Sequential(
            nn.Conv2d(1024, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU())
        # self.weight_2_2 = nn.Sequential(
        #     nn.Conv2d(1024, 8, kernel_size=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU())
        self.weight_2_3 = nn.Sequential(
            nn.Conv2d(1024, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU())

        self.weight_levels_2 =nn.Conv2d(8 * 3, 3, kernel_size=1, stride=1, padding=0)

        # for level_1
        # self.conv_2 = nn.Sequential(
        #     nn.Conv2d(2048,512,kernel_size=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU())
        self.conv_3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.dilated_conv_1_1 = nn.Sequential(
            nn.Conv2d(256,512, kernel_size=3, stride=2, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU())

        # self.weight_1_0 = nn.Sequential(
        #     nn.Conv2d(512, 8, kernel_size=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU())
        self.weight_1_1 = nn.Sequential(
            nn.Conv2d(512, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.weight_1_2 = nn.Sequential(
            nn.Conv2d(512, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.weight_1_3 = nn.Sequential(
            nn.Conv2d(512, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU())

        self.weight_levels_1 = nn.Conv2d(8 * 3, 3, kernel_size=1, stride=1, padding=0)

        # for level_0
        # self.conv_4 = nn.Sequential(
        #     nn.Conv2d(2048,256,kernel_size=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU())
        # self.conv_5 = nn.Sequential(
        #     nn.Conv2d(1024,256,kernel_size=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU())
        self.conv_6 = nn.Sequential(
            nn.Conv2d(512,256,kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        # self.weight_0_0 = nn.Sequential(
        #     nn.Conv2d(256, 8, kernel_size=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU())
        # self.weight_0_1 = nn.Sequential(
        #     nn.Conv2d(256, 8, kernel_size=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU())
        self.weight_0_2 = nn.Sequential(
            nn.Conv2d(256, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.weight_0_3 = nn.Sequential(
            nn.Conv2d(256, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU())

        self.weight_levels_0 =nn.Conv2d(8 * 2, 2, kernel_size=1, stride=1, padding=0)

        # hsv
        self.conv_128 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())  # /2
        self.conv_256 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())  # /2
        self.conv_512 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())  # /2
        self.conv_1024 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())  # /2
        self.conv_2048 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU())  # /2

        self.down_ch_1 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.down_ch_2 = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.down_ch_3 = nn.Sequential(
            nn.Conv2d(1536, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU())
        self.down_ch_4 = nn.Sequential(
            nn.Conv2d(3072, 2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048),
            nn.ReLU())

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

    def forward(self,im,feat):
        # hsv
        hsv = self.rgb2hsv_torch(im)
        hsv_ostu = otsu_threshold(hsv)


        hsv_128 = self.conv_128(hsv_ostu)
        hsv_256 = self.conv_256(hsv_128)
        hsv_512 = self.conv_512(hsv_256)
        hsv_1024 = self.conv_1024(hsv_512)
        hsv_2048 = self.conv_2048(hsv_1024)
        # hsv_level_3_w = self.weight_hsv_3(hsv_2048)
        # hsv_level_2_w = self.weight_hsv_2(hsv_1024)
        # hsv_level_1_w = self.weight_hsv_1(hsv_512)
        # hsv_level_0_w = self.weight_hsv_0(hsv_256)

        level_0, level_1, level_2, level_3 = feat[0], feat[1], feat[2], feat[3]

        # for level_3
        level_2_d_3 = self.dilated_conv_3_3(level_2)
        # level_1_d_3 = self.dilated_conv_3_2(F.max_pool2d(level_1, 3, stride=2, padding=1))
        # level_0_d_3 = self.dilated_conv_3_1(F.max_pool2d(level_0, 4, stride=4, padding=1))

        level_2_d_3_w = self.weight_3_0(level_2_d_3)
        # level_1_d_3_w = self.weight_3_1(level_1_d_3)
        # level_0_d_3_w = self.weight_3_2(level_0_d_3)
        level_3_w = self.weight_3_3(level_3)
        levels_3_w_cat = torch.cat((level_2_d_3_w, level_3_w), 1)
        levels_3_weights = self.weight_levels_3(levels_3_w_cat)
        levels_3_weights = F.softmax(levels_3_weights, dim=1)
        fused_out_3 = level_2_d_3 * levels_3_weights[:, 0:1, :, :] + \
                      level_3 * levels_3_weights[:, 1:2, :, :]

        fused_out_3 = torch.cat((fused_out_3,hsv_2048),dim=1)
        fused_out_3 = self.down_ch_4(fused_out_3)


        # for level_2
        level_3_u_2 = F.interpolate(self.conv_1(level_3), scale_factor=2, mode='nearest')
        level_1_d_2 = self.dilated_conv_2_1(level_1)
        # level_0_d_2 = self.dilated_conv_2_2(F.max_pool2d(level_0, 3, stride=2, padding=1))

        level_3_u_2_w = self.weight_2_0(level_3_u_2)
        level_1_d_2_w = self.weight_2_1(level_1_d_2)
        # level_0_d_2_w = self.weight_2_2(level_0_d_2)
        level_2_w = self.weight_2_3(level_2)
        levels_2_w_cat = torch.cat((level_3_u_2_w, level_1_d_2_w, level_2_w), 1)
        levels_2_weights = self.weight_levels_2(levels_2_w_cat)
        levels_2_weights = F.softmax(levels_2_weights, dim=1)
        fused_out_2 = level_3_u_2 * levels_2_weights[:, 0:1, :, :] + \
                      level_1_d_2 * levels_2_weights[:, 1:2, :, :] + \
                      level_2 * levels_2_weights[:, 2:3, :, :]

        fused_out_2 = torch.cat((fused_out_2,hsv_1024),dim=1)
        fused_out_2 = self.down_ch_3(fused_out_2)


        # for level_1
        # level_3_u_1 = F.interpolate(self.conv_2(level_3), scale_factor=4, mode='nearest')
        level_2_u_1 = F.interpolate(self.conv_3(level_2), scale_factor=2, mode='nearest')
        level_0_d_1 = self.dilated_conv_1_1(level_0)

        # level_3_u_1_w = self.weight_1_0(level_3_u_1)
        level_2_u_1_w = self.weight_1_1(level_2_u_1)
        level_0_d_1_w = self.weight_1_2(level_0_d_1)
        level_1_w = self.weight_1_3(level_1)
        levels_1_w_cat = torch.cat((level_2_u_1_w, level_0_d_1_w, level_1_w), 1)
        levels_1_weights = self.weight_levels_1(levels_1_w_cat)
        levels_1_weights = F.softmax(levels_1_weights, dim=1)
        fused_out_1 = level_2_u_1 * levels_1_weights[:, 0:1, :, :] + \
                      level_0_d_1 * levels_1_weights[:, 1:2, :, :] + \
                      level_1 * levels_1_weights[:, 2:3, :, :]

        fused_out_1 = torch.cat((fused_out_1,hsv_512),dim=1)
        fused_out_1 = self.down_ch_2(fused_out_1)


        # for level_0
        # level_3_u_0 = F.interpolate(self.conv_4(level_3), scale_factor=8, mode='nearest')
        # level_2_u_0 = F.interpolate(self.conv_5(level_2), scale_factor=4, mode='nearest')
        level_1_u_0 = F.interpolate(self.conv_6(level_1), scale_factor=2, mode='nearest')

        # level_3_u_0_w = self.weight_0_0(level_3_u_0)
        # level_2_u_0_w = self.weight_0_1(level_2_u_0)
        level_1_u_0_w = self.weight_0_2(level_1_u_0)
        level_0_w = self.weight_0_3(level_0)
        levels_0_w_cat = torch.cat((level_1_u_0_w, level_0_w), 1)
        levels_0_weights = self.weight_levels_0(levels_0_w_cat)
        levels_0_weights = F.softmax(levels_0_weights, dim=1)
        fused_out_0 = level_1_u_0 * levels_0_weights[:, 0:1, :, :] + \
                      level_0 * levels_0_weights[:, 1:2, :, :]

        fused_out_0 = torch.cat((fused_out_0,hsv_256),dim=1)
        fused_out_0 = self.down_ch_1(fused_out_0)


        outs = []
        outs.append(fused_out_0)
        outs.append(fused_out_1)
        outs.append(fused_out_2)
        outs.append(fused_out_3)

        # fusion weight


        return outs


if __name__=='__main__':
    img = torch.rand(2, 3, 640, 640).to("cuda:2")
    feat = [torch.rand(2, 256, 160, 160).to("cuda:2"),
            torch.rand(2, 512, 80, 80).to("cuda:2"),
            torch.rand(2, 1024, 40, 40).to("cuda:2"),
            torch.rand(2, 2048, 20, 20).to("cuda:2")]
    model = Multiscale_6().to("cuda:2")
    outs = model(img,feat)
    print([i.shape for i in outs])

