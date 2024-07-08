import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=False, activation=F.elu):
        super().__init__()
        self.residual = residual
        self.activation = activation
        self.batch_norm = batch_norm
            
        if in_dim != (out_dim*num_heads):
            self.residual = False

        self.gatconv = GATConv(in_dim, out_dim, num_heads, dropout, dropout)

        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim * num_heads)

    def forward(self, g, h):
        h_in = h 

        h = self.gatconv(g, h).flatten(1)
            
        if self.batch_norm:
            h = self.batchnorm_h(h)
            
        if self.activation:
            h = self.activation(h)
            
        if self.residual:
            h = h_in + h 

        return h