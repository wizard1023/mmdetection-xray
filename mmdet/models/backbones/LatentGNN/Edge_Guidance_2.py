import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
from mmdet.models.backbones.LatentGNN.LatentGNN import LatentGNNV1
from torch.nn import init
class Block(nn.Module):
    def __init__(self, kernel_size, in_channel, expand_size, out_channel, nolinear, gnnmodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.gnn = gnnmodule
        # 1*1展开卷积
        self.conv1 = nn.Conv2d(in_channel, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        # 3*3（或5*5）深度可分离卷积
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        # 1*1投影卷积
        self.conv3 = nn.Conv2d(expand_size, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # 注意力模块
        if self.gnn != None:
            out = self.gnn(out)
        # 残差链接
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out

class Edge_Guidance_2(nn.Module):
    def __init__(self):
        super(Edge_Guidance_2, self).__init__()
        # horizontal
        kernel_const_hori = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype='float32')
        kernel_const_hori = torch.cuda.FloatTensor(kernel_const_hori).unsqueeze(0)
        self.weight_hori = nn.Parameter(data=kernel_const_hori, requires_grad=False)
        # vertical
        kernel_const_vertical = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype='float32')
        kernel_const_vertical = torch.cuda.FloatTensor(kernel_const_vertical).unsqueeze(0)
        self.weight_vertical = nn.Parameter(data=kernel_const_vertical, requires_grad=False)

        # stem
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False) # /2
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.block1 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 48, nn.ReLU(inplace=True), None, 2))

        self.block2 = nn.Sequential(
            Block(3, 48, 72, 48, nn.ReLU(inplace=True), None, 1),
            Block(5, 48, 144, 96, nn.ReLU(inplace=True), None, 1),
            Block(5, 96, 120, 96, nn.ReLU(inplace=True),
                    LatentGNNV1(in_channels=96,latent_dims=[100,100],
                                channel_stride=4,num_kernels=2,mode='asymmetric',
                                graph_conv_flag=False),2))
        self.block3 = nn.Sequential(
            Block(3, 96, 240, 120, nn.ReLU(inplace=True),None,1),
            Block(3, 120, 240, 120, nn.ReLU(inplace=True), None, 1),
            Block(3, 120, 240, 160, nn.ReLU(inplace=True),
                  LatentGNNV1(in_channels=160, latent_dims=[100, 100],
                              channel_stride=4, num_kernels=2, mode='asymmetric',
                              graph_conv_flag=False), 2))
        self.block4 = nn.Sequential(
            Block(3, 160, 320, 160, nn.ReLU(inplace=True), None, 1),
            Block(3, 160, 480, 320, nn.ReLU(inplace=True), None, 1),
            Block(3, 320, 672, 320, nn.ReLU(inplace=True),
                  LatentGNNV1(in_channels=320, latent_dims=[100, 100],
                              channel_stride=4, num_kernels=2, mode='asymmetric',
                              graph_conv_flag=False), 2))
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

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
        edge_detect_conved0 = self.relu1(self.bn1(self.conv1(edge_detect))) #/2

        edge_detect_conved1 = self.block1(edge_detect_conved0)#/4

        edge_detect_conved2 = self.block2(edge_detect_conved1)#/8

        edge_detect_conved3 = self.block3(edge_detect_conved2)#/16

        edge_detect_conved4 = self.block4(edge_detect_conved3)#/32

        edge_outs.append(edge_detect_conved1)
        edge_outs.append(edge_detect_conved2)
        edge_outs.append(edge_detect_conved3)
        edge_outs.append(edge_detect_conved4)

        # fusion
        outs = [torch.cat((edge_outs[i],feat[i]), dim=1) for i in range(len(edge_outs))]

        return outs

if __name__ == "__main__":
    import cv2
    img = cv2.imread('/home/xray/LXM/mmdetection/demo/P00151.jpg')
    img = cv2.resize(img,(640,640))
    img = torch.tensor(img,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    img = img.to("cuda:0")
    feat =[torch.rand(1,256,160,160).to("cuda:0"),
           torch.rand(1,512,80,80).to("cuda:0"),
           torch.rand(1,1024,40,40).to("cuda:0"),
           torch.rand(1,2048,20,20).to("cuda:0")]
    model = Edge_Guidance_2().to("cuda:0")
    outs = model(img,feat)
    # edge_detect = edge_detect.squeeze(0).cpu().numpy()
    # image = np.transpose(edge_detect, (1, 2, 0))
    # cv2.imwrite("/home/xray/LXM/mmdetection/demo/edge_00151.jpg", image)
    #print([i.shape for i in edge_outs])
    print([i.shape for i in outs])




