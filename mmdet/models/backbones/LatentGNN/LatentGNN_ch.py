import torch.nn as nn
import torch.nn.functional as F
import torch

class LatentGNNV1_ch(nn.Module):
    """
    Latent Graph Neural Network for Non-local Relations Learning

    Args:
        in_channels (int): Number of channels in the input feature
        latent_dims (list): List of latent dimensions
        channel_stride (int): Channel reduction factor. Default: 4
        num_kernels (int): Number of latent kernels used. Default: 1
        mode (str): Mode of bipartite graph message propagation. Default: 'asymmetric'.
        without_residual (bool): Flag of use residual connetion. Default: False
        norm_layer (nn.Module): Module used for batch normalization. Default: nn.BatchNorm2d.
        norm_func (function): Function used for normalization. Default: F.normalize
        graph_conv_flag (bool): Flag of use graph convolution layer. Default: False

    """

    def __init__(self, in_channels, in_spatial, latent_dims,
                 num_kernels=1,
                 mode='asymmetric', without_residual=False,
                 norm_layer=nn.BatchNorm2d, norm_func=F.normalize,
                 graph_conv_flag=False):
        super(LatentGNNV1_ch, self).__init__()
        self.without_resisual = without_residual
        self.num_kernels = num_kernels
        self.mode = mode
        self.norm_func = norm_func


        # Reduce the channel dimension for efficiency
        if mode == 'asymmetric':
            self.down_size_v2l = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=in_channels,
                          kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(in_channels),
            )

            self.down_size_l2v = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=in_channels,
                          kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(in_channels),
            )

        elif mode == 'symmetric':
            self.down_size = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=in_channels,
                          kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(in_channels),
            )
            # nn.init.kaiming_uniform_(self.down_channel[0].weight, a=1)
            # nn.init.kaiming_uniform_(self.down_channel[0].weight, mode='fan_in')
        else:
            raise NotImplementedError

        # Define the latentgnn kernel
        assert len(latent_dims) == num_kernels, 'Latent dimensions mismatch with number of kernels'

        for i in range(num_kernels):
            self.add_module('LatentGNN_Kernel_ch_{}'.format(i),
                            LatentGNN_Kernel_ch(in_spatial=in_spatial,
                                             num_kernels=num_kernels,
                                             latent_dim=latent_dims[i],
                                             norm_layer=norm_layer,
                                             norm_func=norm_func,
                                             mode=mode,
                                             graph_conv_flag=graph_conv_flag))
        # Increase the channel for the output
        self.up_size = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=in_channels * num_kernels,
                      out_channels=in_channels,
                      kernel_size=1, padding=0, bias=False),
            norm_layer(in_channels),
        )

        #max pool
        # V1
        #self.maxpool = nn.AdaptiveMaxPool2d(1)
        # V2
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_spatial * 4, out_channels=1,
                      kernel_size=1,stride=1,padding=0,bias=False),
            norm_layer(1)
        )
        self.sigmoid = nn.Sigmoid()

        # Residual Connection
        #self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, conv_feature):
        # Generate visible space feature
        if self.mode == 'asymmetric':
            v2l_conv_feature = self.down_size_v2l(conv_feature)
            l2v_conv_feature = self.down_size_l2v(conv_feature)
            v2l_conv_feature = self.norm_func(v2l_conv_feature, dim=1)
            l2v_conv_feature = self.norm_func(l2v_conv_feature, dim=1)
        elif self.mode == 'symmetric':
            v2l_conv_feature = self.norm_func(self.down_size(conv_feature), dim=1)
            l2v_conv_feature = None
        out_features = []
        for i in range(self.num_kernels):
            out_features.append(eval('self.LatentGNN_Kernel_ch_{}'.format(i))(v2l_conv_feature, l2v_conv_feature))

        out_features = torch.cat(out_features, dim=1) if self.num_kernels > 1 else out_features[0]

        out_features = self.up_size(out_features)

        # V1
        #channel_attention = self.maxpool(out_features)

        # v2
        B, C, H, W=out_features.shape
        out_features = out_features.reshape(B,C,-1)
        out_features = out_features.permute(0,2,1).unsqueeze(-1)
        channel_attention = self.conv1_1(out_features).transpose(1,2)

        channel_attention = self.sigmoid(channel_attention)

        # if self.without_resisual:
        #     return out_features * channel_attention
        # else:
        #     return conv_feature + out_features * channel_attention
        return channel_attention
class LatentGNNV1_ch_2(nn.Module):
    """
    Latent Graph Neural Network for Non-local Relations Learning

    Args:
        in_channels (int): Number of channels in the input feature
        latent_dims (list): List of latent dimensions
        channel_stride (int): Channel reduction factor. Default: 4
        num_kernels (int): Number of latent kernels used. Default: 1
        mode (str): Mode of bipartite graph message propagation. Default: 'asymmetric'.
        without_residual (bool): Flag of use residual connetion. Default: False
        norm_layer (nn.Module): Module used for batch normalization. Default: nn.BatchNorm2d.
        norm_func (function): Function used for normalization. Default: F.normalize
        graph_conv_flag (bool): Flag of use graph convolution layer. Default: False

    """

    def __init__(self, in_channels, in_spatial, latent_dims,
                 num_kernels=1,
                 mode='asymmetric', without_residual=False,
                 norm_layer=nn.BatchNorm2d, norm_func=F.normalize,
                 graph_conv_flag=False):
        super(LatentGNNV1_ch_2, self).__init__()
        self.without_resisual = without_residual
        self.num_kernels = num_kernels
        self.mode = mode
        self.norm_func = norm_func


        # Reduce the channel dimension for efficiency
        if mode == 'asymmetric':
            self.down_size_v2l = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=in_channels,
                          kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(in_channels),
            )

            self.down_size_l2v = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=in_channels,
                          kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(in_channels),
            )

        elif mode == 'symmetric':
            self.down_size = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=in_channels,
                          kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(in_channels),
            )
            # nn.init.kaiming_uniform_(self.down_channel[0].weight, a=1)
            # nn.init.kaiming_uniform_(self.down_channel[0].weight, mode='fan_in')
        else:
            raise NotImplementedError

        # Define the latentgnn kernel
        assert len(latent_dims) == num_kernels, 'Latent dimensions mismatch with number of kernels'

        for i in range(num_kernels):
            self.add_module('LatentGNN_Kernel_ch_{}'.format(i),
                            LatentGNN_Kernel_ch(in_spatial=in_spatial,
                                             num_kernels=num_kernels,
                                             latent_dim=latent_dims[i],
                                             norm_layer=norm_layer,
                                             norm_func=norm_func,
                                             mode=mode,
                                             graph_conv_flag=graph_conv_flag))
        # Increase the channel for the output
        self.up_size = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=in_channels * num_kernels,
                      out_channels=in_channels,
                      kernel_size=1, padding=0, bias=False),
            norm_layer(in_channels),
        )

        #max pool
        # V1
        #self.maxpool = nn.AdaptiveMaxPool2d(1)
        # V2
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_spatial * 4, out_channels=1,
                      kernel_size=1,stride=1,padding=0,bias=False),
            norm_layer(1)
        )
        # self.sigmoid = nn.Sigmoid()

        # Residual Connection
        #self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, conv_feature):
        # Generate visible space feature
        if self.mode == 'asymmetric':
            v2l_conv_feature = self.down_size_v2l(conv_feature)
            l2v_conv_feature = self.down_size_l2v(conv_feature)
            v2l_conv_feature = self.norm_func(v2l_conv_feature, dim=1)
            l2v_conv_feature = self.norm_func(l2v_conv_feature, dim=1)
        elif self.mode == 'symmetric':
            v2l_conv_feature = self.norm_func(self.down_size(conv_feature), dim=1)
            l2v_conv_feature = None
        out_features = []
        for i in range(self.num_kernels):
            out_features.append(eval('self.LatentGNN_Kernel_ch_{}'.format(i))(v2l_conv_feature, l2v_conv_feature))

        out_features = torch.cat(out_features, dim=1) if self.num_kernels > 1 else out_features[0]

        out_features = self.up_size(out_features)

        # V1
        #channel_attention = self.maxpool(out_features)

        # v2
        B, C, H, W=out_features.shape
        out_features = out_features.reshape(B,C,-1)
        out_features = out_features.permute(0,2,1).unsqueeze(-1)
        channel_attention = self.conv1_1(out_features).transpose(1,2)

        # channel_attention = self.sigmoid(channel_attention)

        # if self.without_resisual:
        #     return out_features * channel_attention
        # else:
        #     return conv_feature + out_features * channel_attention
        return channel_attention

class LatentGNN_Kernel_ch(nn.Module):
    """
    A LatentGNN Kernel Implementation

    Args:

    """

    def __init__(self, in_spatial, num_kernels,
                 latent_dim, norm_layer,
                 norm_func, mode, graph_conv_flag):
        super(LatentGNN_Kernel_ch, self).__init__()
        self.mode = mode
        self.norm_func = norm_func
        # ----------------------------------------------
        # Step1 & 3: Visible-to-Latent & Latent-to-Visible
        # ----------------------------------------------

        if mode == 'asymmetric':
            self.psi_v2l = nn.Sequential(
                nn.Conv2d(in_channels=in_spatial,
                          out_channels=latent_dim,
                          kernel_size=1, padding=0,
                          bias=False),
                norm_layer(latent_dim),
                nn.ReLU(inplace=True),
            )
            # nn.init.kaiming_uniform_(self.psi_v2l[0].weight, a=1)
            # nn.init.kaiming_uniform_(self.psi_v2l[0].weight, mode='fan_in')
            self.psi_l2v = nn.Sequential(
                nn.Conv2d(in_channels=in_spatial,
                          out_channels=latent_dim,
                          kernel_size=1, padding=0,
                          bias=False),
                norm_layer(latent_dim),
                nn.ReLU(inplace=True),
            )

        elif mode == 'symmetric':
            self.psi = nn.Sequential(
                nn.Conv2d(in_channels=in_spatial,
                          out_channels=latent_dim,
                          kernel_size=1, padding=0,
                          bias=False),
                norm_layer(latent_dim),
                nn.ReLU(inplace=True),
            )

        # ----------------------------------------------
        # Step2: Latent Messge Passing
        # ----------------------------------------------
        self.graph_conv_flag = graph_conv_flag
        if graph_conv_flag:
            self.GraphConvWeight = nn.Sequential(
                # nn.Linear(in_channels, in_channels,bias=False),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
                norm_layer(in_channels),
                nn.ReLU(inplace=True),
            )
            nn.init.normal_(self.GraphConvWeight[0].weight, std=0.01)

    def forward(self, v2l_conv_feature, l2v_conv_feature):

        B, C, H, W = v2l_conv_feature.shape

        #  reshape
        v2l_conv_feature = v2l_conv_feature.view(B, C, -1).permute(0, 2, 1).unsqueeze(-1)
        l2v_conv_feature = l2v_conv_feature.view(B, C, -1).permute(0, 2, 1).unsqueeze(-1)

        # Generate Bipartite Graph Adjacency Matrix
        if self.mode == 'asymmetric':
            v2l_graph_adj = self.psi_v2l(v2l_conv_feature).squeeze(-1)
            l2v_graph_adj = self.psi_l2v(l2v_conv_feature).squeeze(-1)
            v2l_graph_adj = self.norm_func(v2l_graph_adj, dim=2)
            l2v_graph_adj = self.norm_func(l2v_graph_adj, dim=1)
            # l2v_graph_adj = self.norm_func(l2v_graph_adj.view(B,-1, H*W), dim=2)
        elif self.mode == 'symmetric':
            assert l2v_conv_feature is None
            l2v_graph_adj = v2l_graph_adj = self.norm_func(self.psi(v2l_conv_feature).squeeze(-1), dim=1)

        # ----------------------------------------------
        # Step1 : Visible-to-Latent
        # ----------------------------------------------
        latent_node_feature = torch.bmm(v2l_graph_adj, v2l_conv_feature.reshape(B, -1, H * W))

        # ----------------------------------------------
        # Step2 : Latent-to-Latent
        # ----------------------------------------------
        # Generate Dense-connected Graph Adjacency Matrix
        latent_node_feature_n = self.norm_func(latent_node_feature, dim=-1)
        affinity_matrix = torch.bmm(latent_node_feature_n, latent_node_feature_n.permute(0, 2, 1))
        affinity_matrix = F.softmax(affinity_matrix, dim=-1)

        latent_node_feature = torch.bmm(affinity_matrix, latent_node_feature)

        # ----------------------------------------------
        # Step3: Latent-to-Visible
        # ----------------------------------------------

        visible_feature = torch.bmm(l2v_graph_adj.permute(0, 2, 1), latent_node_feature).reshape(B, -1, H, W)

        if self.graph_conv_flag:
            visible_feature = self.GraphConvWeight(visible_feature)

        return visible_feature

if __name__ == "__main__":
    network = LatentGNNV1_ch(in_channels=1024,in_spatial=225,
                          latent_dims=[100, 100],
                          num_kernels=2,
                          mode='asymmetric',
                          graph_conv_flag=False)
    dump_inputs = torch.rand((8, 1024, 30, 30))
    print(str(network))
    output = network(dump_inputs)
    print(output.shape)