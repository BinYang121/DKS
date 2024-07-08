import dgl
import torch.nn as nn
from gat_layer import GATLayer
from mlp_layer import MLPReadout


class Gat_Net(nn.Module):
    def __init__(self, gat_net_params):
        super().__init__()
        input_dim = gat_net_params['input_dim']
        hidden_dim = gat_net_params['hidden_dim']
        num_heads = gat_net_params['n_heads']
        gat_out_dim = gat_net_params['out_dim']
        in_feat_dropout = gat_net_params['in_feat_dropout']
        dropout = gat_net_params['dropout']
        n_layers = gat_net_params['L']
        self.readout = gat_net_params['readout']
        self.batch_norm = gat_net_params['batch_norm']
        self.residual = gat_net_params['residual']
        output_dim = input_dim

        self.dropout = dropout
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.input_mlp = nn.Linear(input_dim, hidden_dim * num_heads)
        self.gat_layers = nn.ModuleList([GATLayer(hidden_dim * num_heads, hidden_dim, num_heads, dropout,
                                                  self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.gat_layers.append(
            GATLayer(hidden_dim * num_heads, gat_out_dim, 1, dropout, self.batch_norm, self.residual))

        self.output_mlp = MLPReadout(gat_out_dim, output_dim)

    def forward(self, g, gat_matrices):
        h = self.input_mlp(gat_matrices)
        h = self.in_feat_dropout(h)
        for con_v in self.gat_layers:
            h = con_v(g, h)
        g.ndata['h'] = h
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')
        gat_out = self.output_mlp(hg)
        scores = nn.Sigmoid()(gat_out)

        return scores

    def loss(self, scores, targets):
        loss = nn.BCELoss()
        # loss = nn.BCEWithLogitsLoss()
        loss_result = loss(scores, targets)

        return loss_result
