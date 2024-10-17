import torch
import dgl


def heterograph(name_n_feature, dim_n_feature, nb_nodes=2, is_birect=True):
    # 构造图的边关系
    graph_data = {
        ('n', 'contextual', 'n'): (torch.tensor([0]), torch.tensor([1])),
        ('n', 'hierarchical', 'n'): (torch.tensor([0]), torch.tensor([1]))
    }

    # 创建异构图
    g = dgl.heterograph(graph_data, num_nodes_dict={'n': nb_nodes})

    # 初始化节点特征为全零，使用 PyTorch
    g.nodes['n'].data[name_n_feature] = torch.zeros([g.num_nodes(), dim_n_feature])

    # 是否将图转换为双向图
    if is_birect:
        g = dgl.to_bidirected(g, copy_ndata=True)

    # 将图移动到 GPU 上，如果可用
    if torch.cuda.is_available():
        g = g.to('cuda')

    return g


def hetero_add_edges(g, u, v, edges):
    if isinstance(u, int):
        g.add_edges(torch.tensor([u]).to('cuda'), torch.tensor([v]).to('cuda'), etype=edges)
    elif isinstance(u, list):
        g.add_edges(torch.tensor(u).to('cuda'), torch.tensor(v).to('cuda'), etype=edges)
    else:
        g.add_edges(u.to('cuda'), v.to('cuda'), etype=edges)
    return g


def neighbor_9(i, c_shape):
    return torch.tensor([i - c_shape - 1,
                         i - c_shape,
                         i - c_shape + 1,
                         i - 1,
                         i,
                         i + 1,
                         i + c_shape - 1,
                         i + c_shape,
                         i + c_shape + 1], dtype=torch.int64)


def neighbor_25(i, c_shape):
    return torch.tensor([
        i - 2 * c_shape - 2, i - 2 * c_shape - 1, i - 2 * c_shape, i - 2 * c_shape + 1, i - 2 * c_shape + 2,
        i - c_shape - 2, i - c_shape - 1, i - c_shape, i - c_shape + 1, i - c_shape + 2,
        i - 2, i - 1, i, i + 1, i + 2,
        i + c_shape - 2, i + c_shape - 1, i + c_shape, i + c_shape + 1, i + c_shape + 2,
        i + 2 * c_shape - 2, i + 2 * c_shape - 1, i + 2 * c_shape, i + 2 * c_shape + 1, i + 2 * c_shape + 2
    ], dtype=torch.int64)


def build_edges(g, c3_shape=40, c4_shape=20, c5_shape=10):
    c3_size, c4_size, c5_size = c3_shape * c3_shape, c4_shape * c4_shape, c5_shape * c5_shape
    c3 = torch.arange(0, c3_size)
    c4 = torch.arange(c3_size, c3_size + c4_size)
    c5 = torch.arange(c3_size + c4_size, c3_size + c4_size + c5_size)

    # build contextual edges
    for i in range(c3_shape - 1):
        g = hetero_add_edges(g, c3[i * c3_shape: (i + 1) * c3_shape], c3[(i + 1) * c3_shape: (i + 2) * c3_shape],
                             'contextual')  # row-wise edges
        g = hetero_add_edges(g, c3[i: c3_size: c3_shape], c3[i + 1: c3_size: c3_shape],
                             'contextual')  # column-wise edges

    for i in range(c4_shape - 1):
        g = hetero_add_edges(g, c4[i * c4_shape: (i + 1) * c4_shape], c4[(i + 1) * c4_shape: (i + 2) * c4_shape],
                             'contextual')  # row-wise edges
        g = hetero_add_edges(g, c4[i: c4_size: c4_shape], c4[i + 1: c4_size: c4_shape],
                             'contextual')  # column-wise edges

    for i in range(c5_shape - 1):
        g = hetero_add_edges(g, c5[i * c5_shape: (i + 1) * c5_shape], c5[(i + 1) * c5_shape: (i + 2) * c5_shape],
                             'contextual')  # row-wise edges
        g = hetero_add_edges(g, c5[i: c5_size: c5_shape], c5[i + 1: c5_size: c5_shape],
                             'contextual')  # column-wise edges

    # build hierarchical edges
    c3_stride = torch.reshape(c3, (c3_shape, c3_shape))[2:c3_shape:2,
                2:c3_shape:2]  # Reshape and stride for hierarchical edges in C3
    c4_stride = torch.reshape(c4, (c4_shape, c4_shape))[2:c4_shape:2, 2:c4_shape:2]
    c5_stride = torch.reshape(c3, (c3_shape, c3_shape))[2:c3_shape - 4:4, 2:c3_shape - 4:4]

    # Edges between c3 and c4
    counter = 1
    for i in torch.flatten(c3_stride):
        c3_9 = neighbor_9(i.item(), c3_shape)
        g = hetero_add_edges(g, c3_9, c4[counter], 'hierarchical')
        counter += 2 if counter % (c4_shape - 1) == 0 else 1

    # Edges between c4 and c5
    counter = 1
    for i in torch.flatten(c4_stride):
        c4_9 = neighbor_9(i.item(), c4_shape)
        g = hetero_add_edges(g, c4_9, c5[counter], 'hierarchical')
        counter += 2 if counter % (c5_shape - 1) == 0 else 1

    # Edges between c3 and c5
    counter = 1
    for i in torch.flatten(c5_stride):
        c5_9 = neighbor_25(i.item(), c3_shape)
        g = hetero_add_edges(g, c5_9, c5[counter], 'hierarchical')
        counter += 2 if counter % (c5_shape - 1) == 0 else 1

    return g


def simple_birected(g):
    # 将图转换到 CPU
    g = g.to("cpu")

    # 使用 DGL 将图变为简单图，保留节点数据
    g = dgl.to_simple(g, copy_ndata=True)

    # 将图变为双向图，保留节点数据
    g = dgl.to_bidirected(g, copy_ndata=True)

    # 将图转换到 GPU
    g = g.to("cuda")

    return g

def hetero_subgraph(g, edges):
    return dgl.edge_type_subgraph(g, [edges])

def cnn_gnn(g, c):
    g.ndata["pixel"] = c
    return g


import torch


def gnn_cnn(g):
    # 获取节点数据并转换为 PyTorch 张量
    pixel_data = g.ndata["pixel"]

    # p3, p4 和 p5 层的 reshape 操作
    p3 = torch.reshape(pixel_data[:1600], (1, 40, 40, 256))  # 28*28 = 784
    p4 = torch.reshape(pixel_data[1600:2000], (1, 20, 20, 256))  # 14*14 = 196
    p5 = torch.reshape(pixel_data[2000:2100], (1, 10, 10, 256))  # 7*7 = 49

    return p3, p4, p5


def nodes_update(g, val):
    g.apply_nodes(lambda nodes: {'pixel' : val})

if __name__=='__main__':
    g = build_edges(heterograph("pixel", 256, 1029)) # 1029 = 28*28 + 14*14 +7*7
    g = simple_birected(g)
    g = dgl.add_self_loop(g, etype="hierarchical")
    sub_h = hetero_subgraph(g, "hierarchical")
    sub_c = hetero_subgraph(g, 'contextual')
    print(g)
    print(sub_h)
    print(sub_c)

