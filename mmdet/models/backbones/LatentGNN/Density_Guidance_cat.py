import torch
import torch.nn as nn
import torch.nn.functional as F

class Density_Guidance(nn.Module):
    def __init__(self):
        super(Density_Guidance, self).__init__()

        # hsv backbone
        self.conv_128 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())  # /2
        self.conv_256 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())  # /2
        self.conv_512 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())  # /2
        self.conv_1024 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU())  # /2
        self.conv_2048 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU())  # /2

        # 削减通道维度
        self.down_ch_1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.down_ch_2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.down_ch_3 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU())
        self.down_ch_4 = nn.Sequential(
            nn.Conv2d(4096, 2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048),
            nn.ReLU())

    def rgb2hsv_torch(self, rgb: torch.Tensor):
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
        return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

    def forward(self,im, feat):
        hsv_outs = []
        hsv = self.rgb2hsv_torch(im)

        hsv_128 = self.conv_128(hsv)
        hsv_256 = self.conv_256(hsv_128)
        hsv_512 = self.conv_512(hsv_256)
        hsv_1024 = self.conv_1024(hsv_512)
        hsv_2048 = self.conv_2048(hsv_1024)

        hsv_outs.append(hsv_256)
        hsv_outs.append(hsv_512)
        hsv_outs.append(hsv_1024)
        hsv_outs.append(hsv_2048)

        fusion_outs = [torch.cat((feat[i], hsv_outs[i]), dim=1) for i in range(len(hsv_outs))]
        outs = []
        outs.append(self.down_ch_1(fusion_outs[0]))
        outs.append(self.down_ch_2(fusion_outs[1]))
        outs.append(self.down_ch_3(fusion_outs[2]))
        outs.append(self.down_ch_4(fusion_outs[3]))

        return outs

if __name__=='__main__':
    import cv2
    # img = cv2.imread('/home/xray/LXM/mmdetection/demo/P00170.jpg')
    # img = cv2.resize(img, (640, 640))
    # img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img = torch.rand(2, 3, 640, 640).to("cuda:2")
    feat = [torch.rand(2, 256, 160, 160).to("cuda:2"),
            torch.rand(2, 512, 80, 80).to("cuda:2"),
            torch.rand(2, 1024, 40, 40).to("cuda:2"),
            torch.rand(2, 2048, 20, 20).to("cuda:2")]
    model = Density_Guidance().to("cuda:2")
    outs = model(img, feat)
    print([i.shape for i in outs])
