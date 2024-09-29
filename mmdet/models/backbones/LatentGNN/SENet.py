import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.utils import digit_version, is_tuple_of
from torch import Tensor

from mmdet.utils import MultiConfig, OptConfigType, OptMultiConfig

class SENET_guidance(nn.Module):
    def __init__(self, dims):
        super(SENET_guidance, self).__init__()
        in_channel_list = dims
        self.senet = nn.Sequential(
            SELayer(in_channel_list[0],ratio=2),
            SELayer(in_channel_list[1], ratio=4),
            SELayer(in_channel_list[2], ratio=8),
            SELayer(in_channel_list[3], ratio=16)
        )

    def forward(self,feat):
        outs = []
        for i, feat_i in enumerate(feat):
            out = self.senet[i](feat_i)
            outs.append(out)
        return outs

class SELayer(BaseModule):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Defaults to 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Defaults to (dict(type='ReLU'), dict(type='Sigmoid'))
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 channels: int,
                 ratio: int = 16,
                 conv_cfg: OptConfigType = None,
                 act_cfg: MultiConfig = (dict(type='ReLU'),
                                         dict(type='Sigmoid')),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for SELayer."""
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out + x