import torch


def roi_relation(roi_feats):
    num_rois = roi_feats.shape[0]
    roi_feats_flatten = roi_feats.view(roi_feats.size(0), -1)
    eps = torch.mm(roi_feats_flatten, roi_feats_flatten.t())
    _, indices = torch.topk(eps, k=2, dim=0)
    relation = torch.empty(2, 2 * num_rois, dtype=torch.long)
    relation[0] = torch.Tensor(list(range(num_rois)) * 2)  # , type=torch.long)
    relation[1] = indices.view(-1)
    return relation


def roi_distance(rois, relation):
    rois_ctr = (rois[:,1:3]+rois[:,3:5]) / 2
    print(rois_ctr)
    coord_i = rois_ctr[relation[0]]
    coord_j = rois_ctr[relation[1]]
    print(coord_i)
    print(coord_j)
    # print("coord_i = {}, coord_j= {}".format(coord_i.shape, coord_j.shape))
    d = torch.sqrt((coord_i[:, 0] - coord_j[:, 0]) ** 2 + (coord_i[:, 1] - coord_j[:, 1]) ** 2)
    theta = torch.atan2((coord_j[:, 1] - coord_i[:, 1]), (coord_j[:, 0] - coord_i[:, 0]))
    U = torch.stack([d, theta], dim=1)
    return U
if __name__ == '__main__':
    # roi_feats = torch.tensor([])
    rois = torch.tensor([[0, 1, 1.5, 1, 1.5], [0, 2, 2, 4, 2], [1, 3, 3, 9, 3], [1, 4, 4, 10, 4]])

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
    print(relation)
    u = roi_distance(rois, relation)
    print(u)
    # print(roi_feats)
    # print(roi_feats_flatten)

    # print(inner_product_matrix)