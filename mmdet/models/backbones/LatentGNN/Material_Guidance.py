import torch
import torch.nn as nn
import torch.nn.functional as F

class Material_Guidance(nn.Module):
    def __init__(self, dims):
        super(Material_Guidance, self).__init__()
        in_channel_list = dims
        self.material_att = nn.Sequential(
            Material_Attention(in_channel_list[0],ratio=2),
            Material_Attention(in_channel_list[1],ratio=4),
            Material_Attention(in_channel_list[2],ratio=8),
            Material_Attention(in_channel_list[3],ratio=16)
        )
        # self.material_att0 = Material_Attention(in_channel_list[0])
        # self.material_att1 = Material_Attention(in_channel_list[1])
        # self.material_att2 = Material_Attention(in_channel_list[2])
        # self.material_att3 = Material_Attention(in_channel_list[3])

    def forward(self,feat):
        outs = []
        for i,feat_i in enumerate(feat):
            out = self.material_att[i](feat_i)
            outs.append(out)
        return outs

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
        self.beta = nn.Parameter(torch.zeros(1))

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

        channel_node_feat = torch.bmm(affinity_matrix,channel_latent_feat).reshape(B, -1, H, W)

        channel_attention = self.channel_up(channel_node_feat)

        outs = channel_attention * self.beta + feat

        return outs

if __name__ == "__main__":
    inputs = torch.rand(8, 1024, 30, 30)
    model = Material_Attention(in_channels=1024)
    outs = model(inputs)
    print(outs.shape)


