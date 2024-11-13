# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from .base_roi_extractor import BaseRoIExtractor
from .GMM import GMMConv
import torch.nn as nn

@MODELS.register_module()
class SingleRoIExtractor_SGNN(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (:obj:`ConfigDict` or dict): Specify RoI layer type and
            arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
            Defaults to 56.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 roi_layer: ConfigType,
                 out_channels: int,
                 featmap_strides: List[int],
                 finest_scale: int = 56,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            roi_layer=roi_layer,
            out_channels=out_channels,
            featmap_strides=featmap_strides,
            init_cfg=init_cfg)
        self.finest_scale = finest_scale
        self.gaussian = GMMConv(in_channels=256,out_channels=256,dim=2,kernel_size=30)
        self.sg_conv_1 = nn.Linear(256, 512)
        self.sg_conv_2 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.embedding_1 = nn.Linear(256*7*7, 512)
        self.embedding_2 = nn.Linear(512, 256)


    def map_roi_levels(self, rois: Tensor, num_levels: int) -> Tensor:
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls


    def roi_relation(self, roi_feats):
        num_rois = roi_feats.shape[0]
        roi_feats_flatten = roi_feats.view(roi_feats.size(0),-1)
        eps = torch.mm(roi_feats_flatten, roi_feats_flatten.t())
        _, indices = torch.topk(eps, k=32, dim=0)
        relation = torch.empty(2, 32 * num_rois, dtype=torch.long)
        relation[0] = torch.Tensor(list(range(num_rois)) * 32)  # , type=torch.long)
        relation[1] = indices.view(-1)
        return relation

    def roi_visual_embedding(self, roi_feats, rois):
        roi_feats_flatten = roi_feats.view(roi_feats.size(0),-1)
        embedding = self.embedding_1(roi_feats_flatten)
        embedding = self.embedding_2(embedding)
        return embedding

    def roi_distance(self, rois, relation):
        rois_ctr = (rois[:, 1:3] + rois[:, 3:5]) / 2
        coord_i = rois_ctr[relation[0]]
        coord_j = rois_ctr[relation[1]]

        # print("coord_i = {}, coord_j= {}".format(coord_i.shape, coord_j.shape))
        d = torch.sqrt((coord_i[:, 0] - coord_j[:, 0]) ** 2 + (coord_i[:, 1] - coord_j[:, 1]) ** 2)
        theta = torch.atan2((coord_j[:, 1] - coord_i[:, 1]), (coord_j[:, 0] - coord_i[:, 0]))
        U = torch.stack([d, theta], dim=1)
        return U

    def sgnn(self, visual_embedding, relation, U):
        device = visual_embedding.device
        relation = relation.to(device)
        U = U.to(device)
        # print("Number of nodes in x:", visual_embedding.device)
        # print("Max index in edge_index:", relation.device)
        # print('U: ', U.device)
        f = self.gaussian(visual_embedding, relation, U)
        f2 = self.relu(self.sg_conv_1(f))
        h = self.relu(self.sg_conv_2(f2))
        return h


    def forward(self,
                feats: Tuple[Tensor],
                rois: Tensor,
                roi_scale_factor: Optional[float] = None):
        """Extractor ROI feats.

        Args:
            feats (Tuple[Tensor]): Multi-scale features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            roi_scale_factor (Optional[float]): RoI scale factor.
                Defaults to None.

        Returns:
            Tensor: RoI feature.
        """
        # convert fp32 to fp16 when amp is on
        # print(f'single_level_roi_extractor_input_feats shape:{[i.shape for i in feats]}')
        # # [[4,256,200,288],[4,256,100,144],[4,256,50,72],[4,256,25,36]) shape不固定
        # print(f'single_level_roi_extractor_input_rois shape:{rois.shape}')
        # [2048,5]
        rois = rois.type_as(feats[0])
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)

        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        # print(f'roi_feats shape:{roi_feats.shape}') # [2048,256,7,7],[2039,256,7,7],[2043,256,7,7]...
        # print(f'the 0 dim of roi_feats:{roi_feats[0]}')
        # step1: 计算roi之间的关系矩阵作为邻接矩阵
        relation = self.roi_relation(roi_feats)
        # step2: 计算roi的visual embedding
        visual_embedding = self.roi_visual_embedding(roi_feats, rois)
        # step3: 计算roi之间的距离和角度
        U = self.roi_distance(rois, relation)
        # step4: 图推理
        f = self.sgnn(visual_embedding, relation, U)

        # step5:当做通道注意力
        ch_att = f.unsqueeze(-1).unsqueeze(-1)
        roi_feats_new = ch_att * roi_feats + roi_feats
        # step5: 与原始的roi_feat拼接
        # roi_feats_new = torch.cat((roi_feats, f), dim=1)

        return roi_feats_new

if __name__=='__main__':
    roi_feats = torch.randn(4, 2, 3, 3)
    rois = torch.tensor([[0,1,1,1,1],[0,2,2,2,2],[1,3,3,3,3],[1,4,4,4,4]])
    inner_product_matrix = (roi_feats, rois)
