import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as conv

class ContextualLayers(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(ContextualLayers, self).__init__()
        self.in_feats = in_feats
        # 定义三层 GAT 图卷积层，每层的输出特征维度为 h_feats
        self.gat1 = conv.GATConv(in_feats, h_feats, num_heads=1, activation=nn.ReLU())
        self.gat2 = conv.GATConv(h_feats, h_feats, num_heads=1, activation=nn.ReLU())
        self.gat3 = conv.GATConv(h_feats, h_feats, num_heads=1, activation=nn.ReLU())

    def forward(self, g, in_feat):
        # g 是图结构，in_feat 是输入特征
        h = self.gat1(g, in_feat)  # 第一次 GAT 卷积
        h = h.mean(1)  # 将多头注意力的输出取平均
        h = self.gat2(g, h)  # 第二次 GAT 卷积
        h = h.mean(1)
        h = self.gat3(g, h)  # 第三次 GAT 卷积
        h = h.mean(1)
        return h  # 返回最后的特征图

# 示例用法
g = dgl.graph(([0, 1, 2], [1, 2, 3]))  # 创建一个简单的图
in_feat = torch.randn(4, 10)  # 假设有4个节点，每个节点有10维输入特征

# 创建模型并进行前向传播
model = ContextualLayers(in_feats=10, h_feats=16)
output = model(g, in_feat)

print(output)

