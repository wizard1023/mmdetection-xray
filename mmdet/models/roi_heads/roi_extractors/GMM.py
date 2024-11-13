import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from mmdet.models.roi_heads.roi_extractors.gmm_inits import reset, glorot, zeros

EPS = 1e-15


class GMMConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dim,
                 kernel_size,
                 bias=True,
                 **kwargs):
        super(GMMConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size

        self.lin = torch.nn.Linear(in_channels,
                                   out_channels * kernel_size,
                                   bias=False)
        self.mu = Parameter(torch.Tensor(kernel_size, dim))
        self.sigma = Parameter(torch.Tensor(kernel_size, dim))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.mu)
        glorot(self.sigma)
        zeros(self.bias)
        reset(self.lin)

    def forward(self, x, edge_index, pseudo):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        out = self.lin(x)

        out = self.propagate(edge_index, x=out, pseudo=pseudo)

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, pseudo):

        x_j = x_j.view(-1, self.kernel_size, self.out_channels)
        (E, D), K = pseudo.size(), self.mu.size(0)

        gaussian = -0.5 * (pseudo.view(E, 1, D) - self.mu.view(1, K, D))**2
        gaussian = gaussian / (EPS + self.sigma.view(1, K, D)**2)
        gaussian = torch.exp(gaussian.sum(dim=-1, keepdim=True))  # [E, K, 1]

        result = (x_j * gaussian).sum(dim=1)

        return result

    def __repr__(self):
        return '{}({}, {}, kernel_size={})'.format(self.__class__.__name__,
                                                   self.in_channels,
                                                   self.out_channels,
                                                   self.kernel_size)

if __name__=='__main__':
    x = torch.randn((6,3))
    num_rois = x.shape[0]
    roi_feats_flatten = x.view(x.size(0), -1)
    eps = torch.mm(roi_feats_flatten, roi_feats_flatten.t())
    _, indices = torch.topk(eps, k=2, dim=0)
    relation = torch.empty(2, 2 * num_rois, dtype=torch.long)
    relation[0] = torch.Tensor(list(range(num_rois)) * 2)
    relation[1] = indices.view(-1)
    gaussian = GMMConv(in_channels=3, out_channels=3, dim=2, kernel_size=4)
    U = torch.randn((num_rois*2,2))
    f = gaussian(x, relation, U)



