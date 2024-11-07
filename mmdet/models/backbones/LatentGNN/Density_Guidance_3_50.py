import torch.nn.functional as F

import torch.nn as nn

import torch

from mmdet.models.backbones.LatentGNN.LatentGNN_hsv import LatentGNNV1_hsv

class Density_Guidance_3_50(nn.Module):
    def __init__(self):
        super(Density_Guidance_3_50, self).__init__()

        self.conv_128 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())  # /2
        self.conv_256 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())  # /2
        self.conv_512 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())  # /2
        self.conv_1024 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU())  # /2
        self.conv_2048 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU())  # /2
        #
        # self.down_ch_1 = nn.Sequential(
        #     nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU())
        # self.down_ch_2 = nn.Sequential(
        #     nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU())
        # self.down_ch_3 = nn.Sequential(
        #     nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU())
        # self.down_ch_4 = nn.Sequential(
        #     nn.Conv2d(4096, 2048, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU())


        self.latent_gnn0 = LatentGNNV1_hsv(in_channels=256,
                                       latent_dims=[50, 50],
                                       channel_stride=8,
                                       num_kernels=2,
                                       mode='asymmetric',
                                       graph_conv_flag=False)


        self.latent_gnn1= LatentGNNV1_hsv(in_channels=512,
                                           latent_dims=[50, 50],
                                           channel_stride=8,
                                           num_kernels=2,
                                           mode='asymmetric',
                                           graph_conv_flag=False)


        self.latent_gnn2 = LatentGNNV1_hsv(in_channels=1024,
                                           latent_dims=[50, 50],
                                           channel_stride=8,
                                           num_kernels=2,
                                           mode='asymmetric',
                                           graph_conv_flag=False)


        self.latent_gnn3= LatentGNNV1_hsv(in_channels=2048,
                                           latent_dims=[50, 50],
                                           channel_stride=8,
                                           num_kernels=2,
                                           mode='asymmetric',
                                           graph_conv_flag=False)


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

    def forward(self, im, feat):
        outs = feat
        hsv_outs = []
        # torch.set_printoptions(threshold=torch.inf)
        # print(im)
        hsv = self.rgb2hsv_torch(im)

        hsv_128 = self.conv_128(hsv)
        hsv_256 = self.conv_256(hsv_128)
        hsv_512 = self.conv_512(hsv_256)
        hsv_1024 = self.conv_1024(hsv_512)
        hsv_2048 = self.conv_2048(hsv_1024)

        # hsv_outs.append(hsv_256)
        # hsv_outs.append(hsv_512)
        # hsv_outs.append(hsv_1024)
        # hsv_outs.append(hsv_2048)
        # #
        # #print([hsv_outs[i].shape for i in range(4)])
        #
        # # fusion
        # fusion_outs = [torch.cat((feat[i],hsv_outs[i]),dim=1) for i in range(len(hsv_outs))]
        #
        # # channel down
        # outs = []
        # outs.append(self.down_ch_1(fusion_outs[0]))
        # outs.append(self.down_ch_2(fusion_outs[1]))
        # outs.append(self.down_ch_3(fusion_outs[2]))
        # outs.append(self.down_ch_4(fusion_outs[3]))

        # channel attention
        outs_att = []

        outs_att_1 = self.latent_gnn0(feat[0],hsv_256)


        outs_att_2 = self.latent_gnn1(feat[1],hsv_512)


        outs_att_3 = self.latent_gnn2(feat[2],hsv_1024)


        outs_att_4 = self.latent_gnn3(feat[3],hsv_2048)

        outs_att.append(outs_att_1)
        outs_att.append(outs_att_2)
        outs_att.append(outs_att_3)
        outs_att.append(outs_att_4)

        return outs_att


if __name__=='__main__':
    import cv2
    # img = cv2.imread('/home/xray/LXM/mmdetection/demo/P00170.jpg')
    # img = cv2.resize(img, (640, 640))
    # img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img = torch.rand(2, 3, 640, 640).to("cuda:2")
    feat = [torch.rand(2, 256, 160, 160).to("cuda:2"),
            torch.rand(2, 512, 80, 80).to("cuda:2"),
            torch.rand(2, 1024, 40, 40).to("cuda:2"),
            torch.rand(2, 2048, 20, 20).to("cuda:2")]
    model = Density_Guidance_3_50().to("cuda:2")
    outs = model(img,feat)
    print([i.shape for i in outs])