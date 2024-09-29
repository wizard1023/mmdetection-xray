import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
from mmdet.models.backbones.LatentGNN.LatentGNN import LatentGNNV1

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

# bcsp = BottleneckCSP(1, 2)
#
class Edge_Guidance_5(nn.Module):
    def __init__(self):
        super(Edge_Guidance_5, self).__init__()
        # horizontal
        kernel_const_hori = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype='float32')
        kernel_const_hori = torch.cuda.FloatTensor(kernel_const_hori).unsqueeze(0)
        self.weight_hori = nn.Parameter(data=kernel_const_hori, requires_grad=False)
        # vertical
        kernel_const_vertical = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype='float32')
        kernel_const_vertical = torch.cuda.FloatTensor(kernel_const_vertical).unsqueeze(0)
        self.weight_vertical = nn.Parameter(data=kernel_const_vertical, requires_grad=False)

        # stem
        self.conv1 = nn.Conv2d(1, 128, kernel_size=6, stride=2, padding=2) # /2
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # /4
        #
        self.conv3 = nn.Conv2d(256, 512, kernel_size=1,stride=1, padding=1) # /8
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.latent_gnn1 = LatentGNNV1(in_channels=512,
                                       latent_dims=[100, 100],
                                       channel_stride=8,
                                       num_kernels=2,
                                       mode='asymmetric',
                                       graph_conv_flag=False)


        self.conv4 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=1)  # /16
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.latent_gnn2 = LatentGNNV1(in_channels=1024,
                                       latent_dims=[100, 100],
                                       channel_stride=8,
                                       num_kernels=2,
                                       mode='asymmetric',
                                       graph_conv_flag=False)


        self.conv5 = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=1) # /32
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)


    def forward(self, im, feat):
        x3 = Variable(im[:, 2].unsqueeze(1))  # 感觉有点问题
        weight_hori = Variable(self.weight_hori)
        weight_vertical = Variable(self.weight_vertical)

        x_hori = F.conv2d(x3, weight_hori, padding=1)
        x_vertical = F.conv2d(x3, weight_vertical, padding=1)


        # get edge_img
        edge_detect = (torch.add(x_hori.pow(2), x_vertical.pow(2))).pow(0.5)

        # edge_detect = edge_detect.squeeze(0).cpu().numpy()
        # image = np.transpose(edge_detect, (1, 2, 0))
        # cv2.imwrite("/home/xray/LXM/mmdetection/demo/edge_00151.jpg", image)

        # latentgnn for edge_img
        edge_outs = []
        edge_detect_conved1 = self.conv1(edge_detect)
        edge_detect_conved1 = self.conv2(edge_detect_conved1)
        edge_detect_conved2 = self.conv3(edge_detect_conved1)
        edge_detect_conved2 = self.pool1(edge_detect_conved2)
        edge_detect_conved2 = self.latent_gnn1(edge_detect_conved2)
        edge_detect_conved3 = self.conv4(edge_detect_conved2)
        edge_detect_conved3 = self.pool2(edge_detect_conved3)
        edge_detect_conved3 = self.latent_gnn2(edge_detect_conved3)
        edge_detect_conved4 = self.conv5(edge_detect_conved3)
        edge_detect_conved4 = self.pool3(edge_detect_conved4)
        edge_outs.append(edge_detect_conved1)
        edge_outs.append(edge_detect_conved2)
        edge_outs.append(edge_detect_conved3)
        edge_outs.append(edge_detect_conved4)

        # fusion
        outs = [edge_outs[i] + feat[i] for i in range(len(edge_outs))]

        return outs

if __name__ == "__main__":
    import cv2
    img = cv2.imread('/home/xray/LXM/mmdetection/demo/P00151.jpg')
    img = cv2.resize(img,(640,640))
    img = torch.tensor(img,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    img = img.to("cuda:0")
    feat =[torch.rand(1, 256, 160, 160).to("cuda:0"),
           torch.rand(1, 512, 80, 80).to("cuda:0"),
           torch.rand(1, 1024, 40, 40).to("cuda:0"),
           torch.rand(1, 2048, 20, 20).to("cuda:0")]
    model = Edge_Guidance_5().to("cuda:0")
    outs = model(img, feat)
    # edge_detect = edge_detect.squeeze(0).cpu().numpy()
    # image = np.transpose(edge_detect, (1, 2, 0))
    # cv2.imwrite("/home/xray/LXM/mmdetection/demo/edge_00151.jpg", image)
    # print([i.shape for i in edge_outs])
    print([i.shape for i in outs])




