import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
from mmdet.models.backbones.LatentGNN.LatentGNN import LatentGNNV1
import cv2

class Material_Guidance_1(nn.Module):
    def __init__(self):
        super(Material_Guidance_1, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(1,128,kernel_size=6, stride=2, padding=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=6, stride=2, padding=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=6, stride=2, padding=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        self.latentgnn1 = LatentGNNV1(in_channels=256,
                                       latent_dims=[100, 100],
                                       channel_stride=4,
                                       num_kernels=2,
                                       mode='asymmetric',
                                       graph_conv_flag=False)
        self.channel_dowm = nn.Conv2d(in_channels=256*3,out_channels=256,kernel_size=1)
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3,stride=2, padding=1)
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)

    def rgb2hsv_torch(self,rgb: torch.Tensor):
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
        return hsv_h, hsv_s, hsv_v

    def forward(self, im, feat):

        h, s, v = self.rgb2hsv_torch(im)

        h_conv = self.branch1(h)
        s_conv = self.branch2(s)
        v_conv = self.branch3(v)
        hsv_concate = torch.cat((h_conv,s_conv,v_conv),dim=1)

        material_outs = []

        hsv_inter_1 = self.channel_dowm(hsv_concate)
        hsv_inter_1 = self.latentgnn1(hsv_inter_1)
        hsv_inter_2 = self.conv1(hsv_inter_1)
        hsv_inter_3 = self.conv2(hsv_inter_2)
        hsv_inter_4 = self.conv3(hsv_inter_3)
        material_outs.append(hsv_inter_1)
        material_outs.append(hsv_inter_2)
        material_outs.append(hsv_inter_3)
        material_outs.append(hsv_inter_4)

        # fusion
        outs = [material_outs[i]+feat[i] for i in range(len(material_outs))]

        return outs

if __name__ == "__main__":
    import cv2
    img = cv2.imread('/home/xray/LXM/mmdetection/demo/P00170.jpg')
    img = cv2.resize(img,(640,640))
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img = img.to("cuda:0")
    feat =[torch.rand(1,256,160,160).to("cuda:0"),
           torch.rand(1,512,80,80).to("cuda:0"),
           torch.rand(1,1024,40,40).to("cuda:0"),
           torch.rand(1,2048,20,20).to("cuda:0")]
    model = Material_Guidance_1().to("cuda:0")
    outs= model(img, feat)
    print([i.shape for i in outs])
