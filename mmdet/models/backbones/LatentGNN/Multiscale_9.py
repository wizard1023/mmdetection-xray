import torch.nn as nn

import torch

import torch.nn.functional as F

class Multiscale_9(nn.Module):
    def __init__(self):
        super(Multiscale_9, self).__init__()
        self.down_ch = nn.Sequential(
            nn.Conv2d(in_channels=3840,out_channels=256,kernel_size=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.aspp_0 = ASPP(256)

        self.gate_0 = nn.Conv2d(256,1,kernel_size=1)
        self.gate_1 = nn.Conv2d(256,1,kernel_size=3,stride=2,padding=1)
        self.gate_2 = nn.Conv2d(256,1,kernel_size=3,stride=4)
        self.gate_3 = nn.Conv2d(256,1,kernel_size=3,stride=8)
        self.act = nn.Sigmoid()

    def forward(self,feat):
        level_0, level_1, level_2, level_3 = feat[0], feat[1], feat[2], feat[3]
        level_3_u_0 = F.interpolate(level_3, scale_factor=8, mode='nearest')
        level_2_u_0 = F.interpolate(level_2, scale_factor=4, mode='nearest')
        level_1_u_0 = F.interpolate(level_1, scale_factor=2, mode='nearest')
        levels = torch.concat((level_0,level_1_u_0,level_2_u_0,level_3_u_0),dim=1)
        levels = self.down_ch(levels)
        levels = self.aspp_0(levels)
        gate_0 = self.act(self.gate_0(levels))
        gate_1 = self.act(self.gate_1(levels))
        gate_2 = self.act(self.gate_2(levels))
        gate_3 = self.act(self.gate_3(levels))
        level_0_out = gate_0*level_0+level_0
        level_1_out = gate_1*level_1+level_1
        level_2_out = gate_2*level_2+level_2
        level_3_out = gate_3*level_3+level_3

        return [level_0_out,level_1_out,level_2_out,level_3_out]

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)





if __name__=='__main__':
    img = torch.rand(2, 3, 640, 640).to("cuda:2")
    feat = [torch.rand(2, 256, 160, 160).to("cuda:2"),
            torch.rand(2, 512, 80, 80).to("cuda:2"),
            torch.rand(2, 1024, 40, 40).to("cuda:2"),
            torch.rand(2, 2048, 20, 20).to("cuda:2")]
    model = Multiscale_9().to("cuda:2")
    outs = model(feat)
    print([i.shape for i in outs])

    # aspp = ASPP(256, [6, 12, 18])
    # x = torch.rand(2, 256, 13, 13)
    # print(aspp(x).shape)

