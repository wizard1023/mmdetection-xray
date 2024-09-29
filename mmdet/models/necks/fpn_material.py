from mmdet.models.necks.fpn import FPN
from mmdet.registry import MODELS
import torch.nn.functional as F
import torch
from mmdet.models.backbones.LatentGNN.Material_Guidance import Material_Attention

@MODELS.register_module()

class FPN_Material(FPN):
    def __init__(self, **kwargs):
        super(FPN_Material, self).__init__(**kwargs)
        self.material_guidance = Material_Attention(in_channels=256, ratio=1)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                up_sample_feat = F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
                # lxm add
                up_sample_feat_material = Material_Attention(up_sample_feat)
                laterals[i - 1] = laterals[i - 1] + up_sample_feat_material

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

if __name__ == "__main__":
    in_channels = [256, 512, 1024, 2048]
    scales = [340, 170, 84, 43]
    inputs = [torch.rand(1, c, s, s)
                for c, s in zip(in_channels, scales)]
    self = FPN_Material(in_channels=in_channels, out_channels=256, num_outs=len(in_channels)).eval()
    outputs = self.forward(inputs)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')