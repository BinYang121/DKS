import os
import torch
import torch.utils.data
import dgl
import numpy as np
from torch.utils.data import DataLoader

torch.set_printoptions(precision=6)
EPS = 1e-5


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available.')
        device = torch.device("cpu")
    return device


class GraphDGL(torch.utils.data.Dataset):
    def __init__(self, graphs, num_graphs):
        self.graphs = graphs
        self.num_graphs = num_graphs

        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.graphs)
        self._prepare()

    def _prepare(self):
        for graph in self.graphs:
            adj = graph['adj_tensor']
            edge_list = (adj != 0).nonzero(as_tuple=False)

            edge_idx_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idx_in_adj].reshape(-1).long()

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(graph['num_v'])

            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features

            self.graph_lists.append(g)
            self.graph_labels.append(graph['tt'])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    labels = torch.tensor(np.array(labels))
    batched_graph = dgl.batch(graphs)

    return batched_graph, labels


def data_format(data):
    gat_matrices, gat_graphs, gat_targets = [], [], []
    for i in range(len(data)):
        gat_matrices.append(data[i]['node_labels'])
        gat_graphs.append(data[i])
        gat_targets.append(data[i]['target'])

    return gat_matrices, gat_graphs, gat_targets


def graph_loader(graphs, batch_size):
    data = GraphDGL(graphs, num_graphs=len(graphs))
    x_loader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate)
    out_loader = []
    for i, (batch_graphs, batch_targets) in enumerate(x_loader):
        out_loader.append(batch_graphs)

    return out_loader


def matrix_batch(matrices, batch_size):
    i = 0
    batch_matrices = []
    while (i * batch_size) < len(matrices):
        if ((i + 1) * batch_size) <= len(matrices):
            x = torch.tensor(matrices[(i * batch_size)])
            for j in range(1, batch_size):
                x = torch.cat([x, torch.tensor(matrices[(i * batch_size) + j])], 0)
            batch_matrices.append(x)
        else:
            x = torch.tensor(matrices[(i * batch_size)])
            for j in range(1, len(matrices) - i * batch_size):
                x = torch.cat([x, torch.tensor(matrices[(i * batch_size) + j])], 0)
            batch_matrices.append(x)
        i = i + 1

    return batch_matrices


def target_batch(targets, batch_size):
    i = 0
    batch_targets = []
    while (i * batch_size) < len(targets):
        if ((i + 1) * batch_size) <= len(targets):
            x = torch.FloatTensor([targets[(i * batch_size)]])
            for j in range(1, batch_size):
                x = torch.cat([x, torch.FloatTensor([targets[(i * batch_size) + j]])], 0)
            batch_targets.append(x)
        else:
            x = torch.FloatTensor([targets[(i * batch_size)]])
            for j in range(1, len(targets) - i * batch_size):
                x = torch.cat([x, torch.FloatTensor([targets[(i * batch_size) + j]])], 0)
            batch_targets.append(x)
        i = i + 1

    return batch_targets


def gat_batch(matrices, graphs, targets, batch_size):
    batch_loaders = graph_loader(graphs, batch_size)
    batch_matrices = matrix_batch(matrices, batch_size)
    batch_targets = target_batch(targets, batch_size)

    return batch_loaders, batch_matrices, batch_targets


def accuracy(scores, targets):
    scores = scores.detach()
    scores_clone = scores.clone()
    scores_clone[scores_clone >= 0.5] = 1.
    scores_clone[scores_clone < 0.5] = 0.
    acc = torch.sum((scores_clone == targets).view(-1), dtype=torch.int32).item() / targets.numel()
    return acc


def exp_data_batch(exp_data, batch_size):
    exp_batch_data = []
    for i in range(len(exp_data)):
        if len(exp_data[i]) > 0:
            exp_gat_matrices, exp_gat_graphs, exp_gat_targets = data_format(exp_data[i])
            exp_batch_loaders, exp_batch_matrices, exp_batch_targets = gat_batch(exp_gat_matrices, exp_gat_graphs,
                                                                                 exp_gat_targets, batch_size)
            exp_batch_data.append(
                [exp_batch_loaders, exp_batch_matrices, exp_batch_targets, exp_data[i][0]['data_node_id']])

    return exp_batch_data


def predicts_vec(scores):
    vec = scores.mean(dim=0).tolist()
    result = [0 if x < 0.5 else 1 for x in vec]

    return result
