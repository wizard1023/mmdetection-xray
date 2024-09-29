import torch.nn.functional as F

import torch.nn as nn

import torch

from mmdet.models.backbones.LatentGNN.LatentGNN_ch import LatentGNNV1_ch

class Material_Guidance_7(nn.Module):
    def __init__(self):
        super(Material_Guidance_7, self).__init__()
        self.height = 1333
        self.width = 800

        self.conv_128 = nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1)  # /2
        self.conv_256 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # /2
        self.conv_512 = nn.Conv2d(256, 512, kernel_size=3,stride=2, padding=1)    # /2
        self.maxpool1 = nn.AdaptiveMaxPool2d(output_size=(160, 100))
        self.latent_gnn_ch1 = LatentGNNV1_ch(
                                       in_channels=512,
                                       in_spatial=80*50,
                                       latent_dims=[100, 100],
                                       num_kernels=2,
                                       mode='asymmetric',
                                       graph_conv_flag=False)
        self.conv_1024 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)   # /2
        self.maxpool2 = nn.AdaptiveMaxPool2d(output_size=(80, 50))
        self.latent_gnn_ch2 = LatentGNNV1_ch(
                                        in_channels=1024,
                                        in_spatial=40*25,
                                        latent_dims=[100, 100],
                                        num_kernels=2,
                                        mode='asymmetric',
                                        graph_conv_flag=False)
        self.conv_2048 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)


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

    def forward(self, im, feat):
        material_outs = []
        hsv = self.rgb2hsv_torch(im)

        hsv_128 = self.conv_128(hsv)
        hsv_256 = self.conv_256(hsv_128)
        hsv_512 = self.conv_512(hsv_256)
        hsv_512_pool = self.maxpool1(hsv_512)
        hsv_latentgnn_ch1 = self.latent_gnn_ch1(hsv_512_pool)
        hsv_512 = hsv_512 + hsv_latentgnn_ch1 * hsv_512
        hsv_1024 = self.conv_1024(hsv_512)
        hsv_1024_pool = self.maxpool2(hsv_1024)
        hsv_latentgnn_ch2 = self.latent_gnn_ch2(hsv_1024_pool)
        hsv_1024 = hsv_1024 + hsv_1024 * hsv_latentgnn_ch2
        hsv_2048 = self.conv_2048(hsv_1024)

        material_outs.append(hsv_256)
        material_outs.append(hsv_512)
        material_outs.append(hsv_1024)
        material_outs.append(hsv_2048)
        #print([material_outs[i].shape for i in range(4)])

        # fusion
        outs = [material_outs[i] + feat[i] for i in range(len(material_outs))]

        return outs


if __name__=='__main__':
    import cv2

    # img = cv2.imread('/home/xray/LXM/mmdetection/demo/P00170.jpg')
    # img = cv2.resize(img, (640, 640))
    # img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img = torch.rand(2, 3, 1333, 800).to("cuda:0")
    feat = [torch.rand(2, 256, 160, 160).to("cuda:0"),
            torch.rand(2, 512, 80, 80).to("cuda:0"),
            torch.rand(2, 1024, 40, 40).to("cuda:0"),
            torch.rand(2, 2048, 20, 20).to("cuda:0")]
    model = Material_Guidance_7().to("cuda:0")
    outs = model(img, feat)
    print([i.shape for i in outs])