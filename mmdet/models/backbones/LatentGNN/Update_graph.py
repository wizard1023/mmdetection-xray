import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as conv

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

class contextual_layers(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(contextual_layers, self).__init__()
        self.in_feats = in_feats
        # 使用 DGL 的 GATConv 定义三层图注意力网络
        self.gat1 = GATConv(in_feats, h_feats, num_heads=1, activation=F.relu)
        self.gat2 = GATConv(h_feats, h_feats, num_heads=1, activation=F.relu)
        self.gat3 = GATConv(h_feats, h_feats, num_heads=1, activation=F.relu)

    def forward(self, g, in_feat):
        # 图注意力网络的三层传播
        h = self.gat1(g, in_feat).squeeze()  # 使用 .squeeze() 移除单维度
        h = self.gat2(g, h).squeeze()
        h = self.gat3(g, h).squeeze()
        return h

class hierarchical_layers(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(hierarchical_layers, self).__init__()
        self.in_feats = in_feats
        # 使用 DGL 的 GATConv 定义三层图注意力网络
        self.gat1 = GATConv(in_feats, h_feats, num_heads=1, activation=F.relu)
        self.gat2 = GATConv(h_feats, h_feats, num_heads=1, activation=F.relu)
        self.gat3 = GATConv(h_feats, h_feats, num_heads=1, activation=F.relu)

    def forward(self, g, in_feat):
        # 图注意力网络的三层传播
        h = self.gat1(g, in_feat).squeeze()  # 使用 .squeeze() 移除单维度
        h = self.gat2(g, h).squeeze()
        h = self.gat3(g, h).squeeze()
        return h


# 定义输入的图和节点特征
if __name__=='__main__':

    g = dgl.graph(([0, 0], [1, 2]))  # 简单的图结构
    g = dgl.add_self_loop(g)
    in_feat = torch.randn((3, 10))   # 三个节点，每个节点 10 维特征

    # 初始化层
    layer = contextual_layers(in_feats=10, h_feats=20)

    # 前向传播
    output = layer(g, in_feat)
    print(output)


