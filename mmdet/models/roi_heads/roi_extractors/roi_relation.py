import torch


def roi_relation(roi_feats):
    num_rois = roi_feats.shape[0]
    roi_feats_flatten = roi_feats.view(roi_feats.size(0), -1)
    eps = torch.mm(roi_feats_flatten, roi_feats_flatten.t())
    print(eps)
    _, indices = torch.topk(eps, k=2, dim=0)
    relation = torch.empty(2, 2 * num_rois, dtype=torch.long)
    relation[0] = torch.Tensor(list(range(num_rois)) * 2)  # , type=torch.long)
    relation[1] = indices.view(-1)
    return relation


if __name__ == '__main__':
    # roi_feats = torch.tensor([])
    # rois = torch.tensor([[0, 1, 1, 1, 1], [0, 2, 2, 2, 2], [1, 3, 3, 3, 3], [1, 4, 4, 4, 4]])


    roi_feats = torch.tensor([[[[ 1,  1],
          [ 1,  1]],

         [[ 1, 1],
          [ 1, 1]],

         [[ 1, 1],
          [ 1, 1]]],


        [[[2, 2],
          [ 2, 2]],

         [[2, 2],
          [2, 2]],

         [[ 2, 2],
          [2, 2]]],


        [[[ 3, 3],
          [ 3,  3]],

         [[ 3, 3],
          [3,3]],

         [[3,  3],
          [3, 3]]],


        [[[4, 4],
          [ 4, 4]],

         [[ 4, 4],
          [4,  4]],

         [[ 4, 4],
          [4,  4]]]])
    roi_feats_flatten = roi_feats.view(roi_feats.size(0), -1)
    relation = roi_relation(roi_feats)
    print(roi_feats)
    print(roi_feats_flatten)
    print(relation)
    # print(inner_product_matrix)