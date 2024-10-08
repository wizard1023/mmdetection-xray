import torch.nn.functional as F

import torch.nn as nn

import torch

from mmdet.models.backbones.LatentGNN.LatentGNN_ch import LatentGNNV1_ch
from mmdet.models.backbones.LatentGNN.CGA import CGA_2, ChannelAttention_2

class Material_Guidance_10(nn.Module):
    def __init__(self):
        super(Material_Guidance_10, self).__init__()

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

        self.cga = nn.ModuleList([
            CGA_2(dim=256),
            CGA_2(dim=512),
            CGA_2(dim=1024),
            CGA_2(dim=2048)
        ])

        self.conv_1_1 = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)]
        )

        self.ch_att = nn.ModuleList([
            ChannelAttention_2(dim=256, output_size_h=320,output_size_w=200),
            ChannelAttention_2(dim=512, output_size_h=160, output_size_w=100),
            ChannelAttention_2(dim=1024, output_size_h=80, output_size_w=50),
            ChannelAttention_2(dim=2048, output_size_h=40, output_size_w=25)
        ])

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
        hsv_outs = []
        hsv = self.rgb2hsv_torch(im)

        hsv_128 = self.conv_128(hsv)
        hsv_256 = self.conv_256(hsv_128)
        hsv_512 = self.conv_512(hsv_256)
        hsv_1024 = self.conv_1024(hsv_512)
        hsv_2048 = self.conv_2048(hsv_1024)

        hsv_outs.append(hsv_256)
        hsv_outs.append(hsv_512)
        hsv_outs.append(hsv_1024)
        hsv_outs.append(hsv_2048)

        outs = []

        # fusion
        for i in range(len(hsv_outs)):
            f1 = feat[i]+hsv_outs[i]
            w = self.cga[i](f1)
            f2 = hsv_outs[i] * w + feat[i] * (1-w) + hsv_outs[i] + feat[i]
            f3 = self.conv_1_1[i](f2)
            f4 = self.ch_att[i](f3)*f3
            outs.append(f4)

        return outs

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
    model = Material_Guidance_10().to("cuda:2")
    outs = model(img, feat)
    print([i.shape for i in outs])