import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.backbones.LatentGNN.Construct_graph import *
from mmdet.models.backbones.LatentGNN.Update_graph import *
import dgl
class Density_Guidance(nn.Module):
    def __init__(self):
        super(Density_Guidance, self).__init__()

        # hsv backbone
        # self.la = nn.Sequential(
        #     nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU())  # /2
        # self.conv_256 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU())  # /2
        # self.conv_512 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU())  # /2
        # self.conv_1024 = nn.Sequential(
        #     nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU())  # /2
        # self.conv_2048 = nn.Sequential(
        #     nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU())  # /2

        # 削减通道维度
        self.down_ch_1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.down_ch_2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.down_ch_3 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU())

        # 削减空间维度
        self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=(40, 40))
        self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=(20, 20))
        self.avgpool3 = nn.AdaptiveAvgPool2d(output_size=(10, 10))

        # 图
        self.g = simple_birected(build_edges(heterograph("pixel", 256, 2100)))
        self.g = dgl.add_self_loop(self.g, etype="hierarchical")
        self.g = dgl.add_self_loop(self.g, etype="contextual")
        self.subg_h = hetero_subgraph(self.g, "hierarchical")
        self.subg_c = hetero_subgraph(self.g, "contextual")

        # 图更新
        self.context1 = contextual_layers(256, 256)
        self.context2 = contextual_layers(256, 256)
        # self.context3 = contextual_layers(256, 256)
        self.hierarch1 = hierarchical_layers(256, 256)
        self.hierarch2 = hierarchical_layers(256, 256)
        # self.hierarch3 = hierarchical_layers(256, 256)
        self.context4 = contextual_layers(256, 256)
        self.context5 = contextual_layers(256, 256)
        # self.context6 = contextual_layers(256, 256)

        # 空间通道维度复原
        self.conv_transpose_1 = nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=2, stride=2)
        self.conv_transpose_2 = nn.ConvTranspose2d(in_channels=256, out_channels=1024, kernel_size=2, stride=2)
        self.conv_transpose_3 = nn.ConvTranspose2d(in_channels=256, out_channels=2048, kernel_size=2, stride=2)



    def forward(self,feat):
        # print(f'0:{feat[0].shape}')
        # print(f'1:{feat[1].shape}')
        # print(f'2:{feat[2].shape}')
        # print(f'3:{feat[3].shape}')

        outs = []

        feat_40 = self.avgpool1(self.down_ch_1(feat[1]))
        feat_40_reshape = torch.reshape(feat_40,(feat_40.shape[0],feat_40.shape[1],-1)).permute(0,2,1)
        feat_20 = self.avgpool2(self.down_ch_2(feat[2]))
        feat_20_reshape = torch.reshape(feat_20, (feat_20.shape[0], feat_20.shape[1], -1)).permute(0,2,1)
        feat_10 = self.avgpool3(self.down_ch_3(feat[3]))
        feat_10_reshape = torch.reshape(feat_10, (feat_10.shape[0], feat_10.shape[1], -1)).permute(0,2,1)

        bs = feat_10.shape[0]
        # subg_c = []
        # subg_h = []
        feat_40_update_list = []
        feat_20_update_list = []
        feat_10_update_list = []
        for i in range(bs):
            feat_final = torch.cat((feat_40_reshape[i],feat_20_reshape[i],feat_10_reshape[i]),dim=0)
            # print(feat_final.shape)
            self.g = cnn_gnn(self.g, feat_final)
            # subg_c.append(hetero_subgraph(self.g, "contextual"))
            # subg_h.append(hetero_subgraph(self.g, "hierarchical"))
            nodes_update(self.subg_c, self.context1(self.subg_c, self.subg_c.ndata["pixel"]))
            nodes_update(self.subg_c, self.context2(self.subg_c, self.subg_c.ndata["pixel"]))
            # nodes_update(self.subg_c, self.context3(self.subg_c, self.subg_c.ndata["pixel"]))
            nodes_update(self.subg_h, self.hierarch1(self.subg_h, self.subg_h.ndata["pixel"]))
            nodes_update(self.subg_h, self.hierarch2(self.subg_h, self.subg_h.ndata["pixel"]))
            # nodes_update(self.subg_h, self.hierarch3(self.subg_h, self.subg_h.ndata["pixel"]))
            nodes_update(self.subg_c, self.context4(self.subg_c, self.subg_c.ndata["pixel"]))
            nodes_update(self.subg_c, self.context5(self.subg_c, self.subg_c.ndata["pixel"]))
            # nodes_update(self.subg_c, self.context6(self.subg_c, self.subg_c.ndata["pixel"]))
        # bhg_c = dgl.batch(subg_c)
        # bhg_h = dgl.batch(subg_h)
        # nodes_update(bhg_c, self.context1(bhg_c, bhg_c.ndata["pixel"]))
        # nodes_update(bhg_c, self.context2(bhg_c, bhg_c.ndata["pixel"]))
        # nodes_update(bhg_c, self.context3(bhg_c, bhg_c.ndata["pixel"]))
        # nodes_update(bhg_h, self.hierarch1(bhg_h, bhg_h.ndata["pixel"]))
        # nodes_update(bhg_h, self.hierarch2(bhg_h, bhg_h.ndata["pixel"]))
        # nodes_update(bhg_h, self.hierarch3(bhg_h, bhg_h.ndata["pixel"]))
        # nodes_update(bhg_c, self.context4(bhg_c, bhg_c.ndata["pixel"]))
        # nodes_update(bhg_c, self.context5(bhg_c, bhg_c.ndata["pixel"]))
        # nodes_update(bhg_c, self.context6(bhg_c, bhg_c.ndata["pixel"]))

            feat_40_gnn, feat_20_gnn, feat_10_gnn = gnn_cnn(self.g)
            feat_40_gnn = feat_40_gnn.permute(0, 3, 1, 2)
            feat_20_gnn = feat_20_gnn.permute(0, 3, 1, 2)
            feat_10_gnn = feat_10_gnn.permute(0, 3, 1, 2)
            feat_10_upsampled = F.interpolate(feat_10_gnn, scale_factor=2, mode='nearest')
            feat_20_upsampled = F.interpolate(feat_20_gnn, scale_factor=2, mode='nearest')
            feat_10_update = feat_10[i].unsqueeze(0) + feat_10_gnn
            feat_20_update = feat_20[i].unsqueeze(0) + feat_20_gnn + feat_10_upsampled
            feat_40_update = feat_40[i].unsqueeze(0) + feat_40_gnn +feat_20_upsampled
            feat_40_update_list.append(feat_40_update)
            feat_20_update_list.append(feat_20_update)
            feat_10_update_list.append(feat_10_update)

        # 恢复批量的维度
        feat_40_refine = torch.cat(feat_40_update_list,dim=0)
        feat_20_refine = torch.cat(feat_20_update_list,dim=0)
        feat_10_refine = torch.cat(feat_10_update_list,dim=0)

        # 恢复空间通道维度
        feat_0_out = feat[0]
        feat_1_out = self.conv_transpose_1(feat_40_refine) + feat[1]
        feat_2_out = self.conv_transpose_2(feat_20_refine) + feat[2]
        feat_3_out = self.conv_transpose_3(feat_10_refine) + feat[3]
        outs.append(feat_0_out)
        outs.append(feat_1_out)
        outs.append(feat_2_out)
        outs.append(feat_3_out)
        return outs

if __name__=='__main__':
    import cv2
    # img = cv2.imread('/home/xray/LXM/mmdetection/demo/P00170.jpg')
    # img = cv2.resize(img, (640, 640))
    # img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img = torch.rand(2, 3, 640, 640).to("cuda")
    feat = [torch.rand(2, 256, 160, 160).to("cuda"),
            torch.rand(2, 512, 80, 80).to("cuda"),
            torch.rand(2, 1024, 40, 40).to("cuda"),
            torch.rand(2, 2048, 20, 20).to("cuda")]
    model = Density_Guidance().to("cuda")
    outs = model(feat)
    print([i.shape for i in outs])


