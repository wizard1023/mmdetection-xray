import torch.nn.functional as F
import torch.nn as nn
import torch
from mmdet.models.backbones.LatentGNN.C2f import C2f,Conv


class Material_Guidance_5(nn.Module):
    def __init__(self):
        super(Material_Guidance_5, self).__init__()

        self.conv1 = Conv(c1=3,c2=128,k=6,s=2,p=2)  # /2
        self.c2f = C2f(c1=128, c2=128)
        self.conv_256 = Conv(c1=128,c2=256,k=3,s=2,p=1)
        self.conv_512 = Conv(c1=256,c2=512,k=3,s=2,p=1)     # /2
        self.conv_1024 = Conv(c1=512,c2=1024,k=3,s=2,p=1)   # /2
        self.down_sample = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)  # /2

        self.conv_3 = Conv(c1=2048, c2=1024, k=1)
        self.conv_2 = Conv(c1=1024, c2=512, k=1)
        self.conv_1 = Conv(c1=512, c2=256, k=1)


        self.material_att3 = Material_Attention(in_channels=2048)
        self.material_att2 = Material_Attention(in_channels=1024)
        self.material_att1 = Material_Attention(in_channels=512)


        self.channel_down = Conv(c1=2048, c2=512, k=1)
        self.channel_down1 = Conv(c1=1024, c2=256, k=1)

        self.up_sample = nn.Upsample(scale_factor=2)

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
        hsv = self.rgb2hsv_torch(im)

        hsv_conv_128 = self.c2f(self.conv1(hsv))
        hsv_conv_256 = self.conv_256(hsv_conv_128)
        hsv_conv_512 = self.conv_512(hsv_conv_256)
        hsv_conv_1024 = self.conv_1024(hsv_conv_512)
        hsv_conv_1024_downsample = self.down_sample(hsv_conv_1024)

        feat_conv_3 = self.conv_3(feat[3])
        feat_hsv_3 = torch.cat((feat_conv_3, hsv_conv_1024_downsample), dim=1)
        feat_hsv_3 = self.material_att3(feat_hsv_3)

        feat_hsv_3_upsample = self.up_sample(self.channel_down(feat_hsv_3))
        hsv_conv_512_downsample = self.down_sample(hsv_conv_512)
        hsv_2 = hsv_conv_512_downsample + feat_hsv_3_upsample
        feat_conv_2 = self.conv_2(feat[2])
        feat_hsv_2 = torch.cat((feat_conv_2, hsv_2), dim=1)
        feat_hsv_2 = self.material_att2(feat_hsv_2)

        feat_hsv_2_upsample = self.up_sample(self.channel_down1(feat_hsv_2))
        hsv_conv_256_downsample = self.down_sample(hsv_conv_256)
        hsv_1 = hsv_conv_256_downsample + feat_hsv_2_upsample
        feat_conv_1 = self.conv_1(feat[1])
        feat_hsv_1 = torch.cat((feat_conv_1, hsv_1), dim=1)
        feat_hsv_1 = self.material_att1(feat_hsv_1)

        return [feat[0], feat_hsv_1, feat_hsv_2, feat_hsv_3]


class Material_Attention(nn.Module):
    def __init__(self, in_channels, ratio=8, norm_layer=nn.BatchNorm2d):
        super(Material_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        inter_channel = in_channels//ratio
        self.channel_latent = nn.Sequential(
            nn.Conv2d(in_channels,inter_channel,
                    kernel_size=1,padding=0,bias=False),
            norm_layer(inter_channel),
            nn.ReLU(inplace=True))
        self.channel_up = nn.Sequential(
            nn.Conv2d(inter_channel,in_channels,
                    kernel_size=1,padding=0,bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True))

        self.sigmoid = nn.Sigmoid()

    def forward(self, feat):

        channel_avg = self.avg_pool(feat)
        channel_max = self.max_pool(feat)
        channel_feat = channel_max + channel_avg

        channel_latent_feat = self.channel_latent(channel_feat)

        # Generate Dense-connected Graph Adjacency Matrix
        B, C, H, W = channel_latent_feat.shape
        channel_latent_feat = channel_latent_feat.reshape(B, -1, H * W)
        affinity_matrix = torch.bmm(channel_latent_feat,channel_latent_feat.permute(0,2,1))
        affinity_matrix = F.softmax(affinity_matrix, dim=-1)

        channel_node_feat = torch.bmm(affinity_matrix,channel_latent_feat).reshape(B,-1,H,W)

        channel_attention = self.channel_up(channel_node_feat)
        channel_attention = self.sigmoid(channel_attention)

        outs = channel_attention * feat + feat

        return outs



if __name__=='__main__':
    import cv2
    # img = cv2.imread('/home/xray/LXM/mmdetection/demo/P00170.jpg')
    # img = cv2.resize(img, (640, 640))
    # img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img = torch.rand(2, 3, 640, 640).to("cuda:0")
    feat = [torch.rand(2, 256, 160, 160).to("cuda:0"),
            torch.rand(2, 512, 80, 80).to("cuda:0"),
            torch.rand(2, 1024, 40, 40).to("cuda:0"),
            torch.rand(2, 2048, 20, 20).to("cuda:0")]
    model = Material_Guidance_5().to("cuda:0")
    outs = model(img, feat)
    print([i.shape for i in outs])