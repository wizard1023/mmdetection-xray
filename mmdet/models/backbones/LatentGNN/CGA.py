import torch
import torch.nn as nn
import math
from mmdet.models.backbones.LatentGNN.LatentGNN_ch import LatentGNNV1_ch

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, kernel_size=7, padding=3, padding_mode='reflect', bias=True)
    def forward(self,x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim//reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True)
        )
    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn
class ChannelAttention_2(nn.Module):
    def __init__(self, dim, output_size_h, output_size_w):
        super(ChannelAttention_2, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(output_size=(output_size_h,output_size_w))
        self.ca =  LatentGNNV1_ch(
                                       in_channels=dim,
                                       in_spatial=int((output_size_h/2) * math.ceil(output_size_w/2)),
                                       latent_dims=[100, 100],
                                       num_kernels=2,
                                       mode='asymmetric',
                                       graph_conv_flag=False)

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2*dim, dim, 7, padding=3, padding_mode='reflect',groups=dim,bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1],dim=2)  # B, C, 2, H, W
        x2 = x2.reshape(B, -1, H, W)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

class CGA(nn.Module):
    def __init__(self, dim, output_size_h, output_size_w):
        super(CGA, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention_2(dim, output_size_h, output_size_w)
        self.pa = PixelAttention(dim)

    def forward(self,x):
        sattn = self.sa(x)
        cattn = self.ca(x)
        pattn1 = sattn + cattn
        pattn2 = self.pa(x, pattn1)
        return pattn2
class CGA_2(nn.Module):
    def __init__(self, dim):
        super(CGA_2, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim,reduction=4)
        self.pa = PixelAttention(dim)
    def forward(self,x):
        sattn = self.sa(x)
        cattn = self.ca(x)
        pattn1 = sattn + cattn
        pattn2 = self.pa(x, pattn1)
        return pattn2


if __name__=='__main__':
    import cv2
    # img = cv2.imread('/home/xray/LXM/mmdetection/demo/P00170.jpg')
    # img = cv2.resize(img, (640, 640))
    # img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    # img = torch.rand(2, 3, 640, 640).to("cuda:2")
    # feat = [torch.rand(2, 256, 160, 160).to("cuda:2"),
    #         torch.rand(2, 512, 80, 80).to("cuda:2"),
    #         torch.rand(2, 1024, 40, 40).to("cuda:2"),
    #         torch.rand(2, 2048, 20, 20).to("cuda:2")]
    feat = torch.rand(2, 256, 160, 160).to("cuda:2")
    model = CGA(256,40,25).to("cuda:2")
    outs = model(feat)
    print(outs.shape)