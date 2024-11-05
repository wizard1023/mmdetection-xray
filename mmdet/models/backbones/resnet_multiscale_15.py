
from mmdet.models.backbones import ResNet
from mmdet.registry import MODELS
from mmdet.models.backbones.LatentGNN.Multiscale_15 import Multiscale_15

@MODELS.register_module()
class ResNet_Multiscale_15(ResNet):
    def __init__(self, **kwargs):
        super(ResNet_Multiscale_15, self).__init__(**kwargs)
        self.multiscale = Multiscale_15()

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        outs = self.multiscale(outs)
        return tuple(outs)


if __name__ == "__main__":
    # network = LatentGNNV1(in_channels=1024,
    #                       latent_dims=[100, 100],
    #                       channel_stride=8,
    #                       num_kernels=2,
    #                       mode='asymmetric',
    #                       graph_conv_flag=False)
    # dump_inputs = torch.rand((8, 1024, 30, 30))
    # print(str(network))
    # output = network(dump_inputs)
    # print(output.shape)
    from mmdet.models import ResNet_Multiscale_15
    import torch

    self = ResNet_Multiscale_15(depth=50)
    self.eval()
    inputs = torch.rand(1, 3, 32, 32)
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))