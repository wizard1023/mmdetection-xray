# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .cspnext import CSPNeXt
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet
from .resnet_glore import ResNet_GloRe,GloRe_Unit_2D
from .resnet_latentgnn import ResNet_LatenGNN
from .resnet_edge import ResNet_Edge
from .resnet_edge_cbam import ResNet_Edge_cbam
from .resnet_edge_gcnet import ResNet_Edge_gcnet
from .resnet_edge_senet import ResNet_Edge_senet
from .resnet_edge_nonlocal import ResNet_Edge_nonlocal
from .ssd_vgg_DOAM import SSDVGG_DOAM
from .DOAM.DOMA import DOAM
from .resnet_material import ResNet_Material
from .resnet_edge_material import ResNet_Edge_Material
from .resnet_material_1 import ResNet_Material_1
from .resnet_material_2 import ResNet_Material_2
from .resnet_material_3 import ResNet_Material_3
from .resnet_material_4 import ResNet_Material_4
from .resnet_material_5 import ResNet_Material_5
from .resnet_material_6 import ResNet_Material_6
from .resnet_material_7 import ResNet_Material_7
from .resnet_material_8 import ResNet_Material_8
from .resnet_material_9 import ResNet_Material_9
from .resnet_edge_material_1 import ResNet_Edge_Material_1
from .resnet_edge_material_8 import ResNet_Edge_Material_8
from .resnet_edge_material_9 import ResNet_Edge_Material_9
from .resnet_edge_material_10 import ResNet_Edge_Material_10
from .resnet_edge_material_8_200 import ResNet_Edge_Material_8_200
from .resnet_edge_material_8_150 import ResNet_Edge_Material_8_150
from .resnet_edge_material_8_50 import ResNet_Edge_Material_8_50
from .resnet_doam import ResNet_DOAM
from .resnet_edge_material_8_senet import ResNet_Edge_Material_8_SENet
from .resnet_edge_material_8_cbam import ResNet_Edge_Material_8_CBAM
from .resnet_edge_material_8_nonlocal import ResNet_Edge_Material_8_Nonlocal
from .resnet_edge_material_8_gcnet import ResNet_Edge_Material_8_GCNet
from .resnet_edge_material_8_sknet import ResNet_Edge_Material_8_SKNet
from .resnet_density_cat import ResNet_Density_cat
from .resnet_density_add import ResNet_Density_add
from .resnet_hsv_cat import ResNet_hsv_cat
from .resnet_density import ResNet_Density
from .resnet_density_1 import ResNet_Density_1
from .resnet_density_2 import ResNet_Density_2
from .resnet_density_3 import ResNet_Density_3
from .resnet_multiscale import ResNet_Multiscale
from .resnet_multiscale_1 import ResNet_Multiscale_1
from .resnet_multiscale_2 import ResNet_Multiscale_2
from .resnet_multiscale_3 import ResNet_Multiscale_3
from .resnet_multiscale_4 import ResNet_Multiscale_4
from .resnet_multiscale_5 import ResNet_Multiscale_5
from .resnet_multiscale_6 import ResNet_Multiscale_6
from .resnet_multiscale_7 import ResNet_Multiscale_7
from .resnet_multiscale_8 import ResNet_Multiscale_8
from .resnet_multiscale_9 import ResNet_Multiscale_9
from .resnet_multiscale_10 import ResNet_Multiscale_10
from .resnet_multiscale_11 import ResNet_Multiscale_11
from .resnet_multiscale_12 import ResNet_Multiscale_12
from .resnet_multiscale_13 import ResNet_Multiscale_13
from .resnet_multiscale_14 import ResNet_Multiscale_14
from .resnet_multiscale_15 import ResNet_Multiscale_15
from .resnet_multiscale_16 import ResNet_Multiscale_16
from .resnet_multiscale_17 import ResNet_Multiscale_17
from .resnet_multiscale_18 import ResNet_Multiscale_18
from .resnet_multiscale_19 import ResNet_Multiscale_19
from .resnet_multiscale_20 import ResNet_Multiscale_20
from .resnet_edge_multiscale_14_density_3 import ResNet_Edge_Multiscale_14_Density_3
from .resnet_edge_density_3_multiscale_14 import ResNet_Edge_Density_3_Multiscale_14
from .resnet_edge_density_3_multiscale_14_50 import ResNet_Edge_Density_3_Multiscale_14_50
from .resnet_edge_density_3_multiscale_14_150 import ResNet_Edge_Density_3_Multiscale_14_150
from .resnet_edge_density_3_multiscale_14_200 import ResNet_Edge_Density_3_Multiscale_14_200
__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet', 'CSPNeXt',
    'ResNet_GloRe','GloRe_Unit_2D','ResNet_LatenGNN','ResNet_Edge','ResNet_Edge_cbam','ResNet_Edge_gcnet','ResNet_Edge_nonlocal',
    'ResNet_Edge_senet',
    'ResNet_Density_cat','ResNet_Density_add','ResNet_hsv_cat',
    'ResNet_Density','ResNet_Density_1','ResNet_Density_2','ResNet_Density_3',
    'ResNet_Multiscale','ResNet_Multiscale_1','ResNet_Multiscale_2','ResNet_Multiscale_3','ResNet_Multiscale_4',
    'ResNet_Multiscale_5','ResNet_Multiscale_6','ResNet_Multiscale_7','ResNet_Multiscale_8','ResNet_Multiscale_9',
    'ResNet_Multiscale_10','ResNet_Multiscale_11','ResNet_Multiscale_12','ResNet_Multiscale_13','ResNet_Multiscale_14',
    'ResNet_Multiscale_15','ResNet_Multiscale_16','ResNet_Multiscale_17','ResNet_Multiscale_18','ResNet_Multiscale_19',
    'ResNet_Multiscale_20',
    'ResNet_Edge_Multiscale_14_Density_3','ResNet_Edge_Density_3_Multiscale_14','ResNet_Edge_Density_3_Multiscale_14_50',
    'ResNet_Edge_Density_3_Multiscale_14_150','ResNet_Edge_Density_3_Multiscale_14_200',
    'SSDVGG_DOAM','DOAM','ResNet_Material','ResNet_Edge_Material','ResNet_Edge_Material_1',
    'ResNet_Edge_Material_8','ResNet_Edge_Material_9',
    'ResNet_Material_1','ResNet_Material_2','ResNet_Material_3',
    'ResNet_Material_4','ResNet_Material_5','ResNet_Material_6',
    'ResNet_Material_7','ResNet_Material_8','ResNet_Material_9','ResNet_Edge_Material_10',
    'ResNet_Edge_Material_8_200','ResNet_Edge_Material_8_150',
    'ResNet_Edge_Material_8_50',
    'ResNet_DOAM','ResNet_Edge_Material_8_SENet','ResNet_Edge_Material_8_CBAM','ResNet_Edge_Material_8_Nonlocal',
    'ResNet_Edge_Material_8_GCNet','ResNet_Edge_Material_8_SKNet'
]
