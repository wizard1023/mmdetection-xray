import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
from mmdet.models.backbones.LatentGNN.LatentGNN import LatentGNNV1
from mmdet.models.backbones.LatentGNN.SENet import SELayer
class Edge_Guidance_senet(nn.Module):
    def __init__(self):
        super(Edge_Guidance_senet, self).__init__()
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
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3,stride=2, padding=1) # /8
        self.latent_gnn1 = SELayer(channels=512,ratio=2)

        # self.latent_gnn1 = LatentGNNV1(in_channels=512,
        #                                latent_dims=[100, 100],
        #                                channel_stride=8,
        #                                num_kernels=2,
        #                                mode='asymmetric',
        #                                graph_conv_flag=False)

        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)  # /16
        self.latent_gnn2 = SELayer(channels=1024,ratio=2)
        # self.latent_gnn2 = LatentGNNV1(in_channels=1024,
        #                                latent_dims=[100, 100],
        #                                channel_stride=8,
        #                                num_kernels=2,
        #                                mode='asymmetric',
        #                                graph_conv_flag=False)

        self.conv5 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1) # /32


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
        edge_detect_conved2 = self.latent_gnn1(edge_detect_conved2)
        edge_detect_conved3 = self.conv4(edge_detect_conved2)
        edge_detect_conved3 = self.latent_gnn2(edge_detect_conved3)
        edge_detect_conved4 = self.conv5(edge_detect_conved3)
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
    # img = cv2.resize(img,(640,640))
    img = torch.tensor(img,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    img = img.to("cuda:0")
    feat =[torch.rand(1,256,160,160).to("cuda:0"),
           torch.rand(1,512,80,80).to("cuda:0"),
           torch.rand(1,1024,40,40).to("cuda:0"),
           torch.rand(1,2048,20,20).to("cuda:0")]
    model = Edge_Guidance_senet().to("cuda:0")
    outs = model(img,feat)
    # edge_detect = edge_detect.squeeze(0).cpu().numpy()
    # image = np.transpose(edge_detect, (1, 2, 0))
    # cv2.imwrite("/home/xray/LXM/mmdetection/demo/edge_00151.jpg", image)
    # print([i.shape for i in edge_outs])
    # print([i.shape for i in outs])




