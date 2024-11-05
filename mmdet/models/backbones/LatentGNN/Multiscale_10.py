import torch.nn as nn

import torch

import torch.nn.functional as F

class Multiscale_10(nn.Module):
    def __init__(self):
        super(Multiscale_10, self).__init__()

        # filter
        self.sa_0 = SpatialAttention()
        self.sa_1 = SpatialAttention()
        self.sa_2 = SpatialAttention()
        self.sa_3 = SpatialAttention()


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

        self.weight_levels_2 = nn.Conv2d(8 * 3, 3, kernel_size=1, stride=1, padding=0)


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

        self.weight_levels_0 = nn.Conv2d(8 * 2, 2, kernel_size=1, stride=1, padding=0)


    def forward(self,feat):
        level_0,level_1,level_2,level_3 = feat[0],feat[1],feat[2],feat[3]
        level_0 = self.sa_0(level_0)
        level_1 = self.sa_1(level_1)
        level_2 = self.sa_2(level_2)
        level_3 = self.sa_3(level_3)

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
                            level_3 * levels_3_weights[:, 1:, :, :]


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
                      level_2 * levels_2_weights[:, 2:, :, :]

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
                      level_1 * levels_1_weights[:, 2:, :, :]

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
                      level_0 * levels_0_weights[:, 1:, :, :]

        outs = []
        outs.append(fused_out_0)
        outs.append(fused_out_1)
        outs.append(fused_out_2)
        outs.append(fused_out_3)

        # fusion weight


        return outs

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, flag=True):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.flag = flag
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)*x if self.flag else self.sigmoid(out)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio = 4, flag=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)*x if self.flag else self.sigmoid(out)

if __name__=='__main__':
    img = torch.rand(2, 3, 640, 640).to("cuda:2")
    feat = [torch.rand(2, 256, 160, 160).to("cuda:2"),
            torch.rand(2, 512, 80, 80).to("cuda:2"),
            torch.rand(2, 1024, 40, 40).to("cuda:2"),
            torch.rand(2, 2048, 20, 20).to("cuda:2")]
    model = Multiscale_10().to("cuda:2")
    outs = model(feat)
    print([i.shape for i in outs])

