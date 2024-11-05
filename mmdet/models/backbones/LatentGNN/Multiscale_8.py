import torch.nn as nn

import torch

import torch.nn.functional as F

class Multiscale_8(nn.Module):
    def __init__(self):
        super(Multiscale_8, self).__init__()
        self.aspp_0 = ASPP(256)
        self.aspp_1 = ASPP(512)
        self.aspp_2 = ASPP(1024)
        self.aspp_3 = ASPP(2048)

    def forward(self,feat):

        outs_0 = self.aspp_0(feat[0])
        outs_1 = self.aspp_1(feat[1])
        outs_2 = self.aspp_2(feat[2])
        outs_3 = self.aspp_3(feat[3])
        return [outs_0,outs_1,outs_2,outs_3]

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
    model = Multiscale_8().to("cuda:2")
    outs = model(feat)
    print([i.shape for i in outs])

    # aspp = ASPP(256, [6, 12, 18])
    # x = torch.rand(2, 256, 13, 13)
    # print(aspp(x).shape)

