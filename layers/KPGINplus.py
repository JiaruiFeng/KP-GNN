"""
KP-GIN plus layer
"""
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from .combine import *


class KPGINPlusConv(MessagePassing):
    """KP-GNN with GIN plus convolution kernel
    Args:
        input_size (int): the size of input feature
        output_size (int): the size of output feature
        K (int): number of hop to consider in Convolution layer
        num_hop1_edge (int): number of edge type at 1 hop
        num_pe (int): maximum number of path encoding, larger or equal to 1
    """

    def __init__(self, input_size, output_size, K, num_hop1_edge=1, num_pe=1, combine="independent"):
        super(KPGINPlusConv, self).__init__(node_dim=0)
        self.aggr = "add"
        self.K = K
        self.output_size = output_size
        self.mlp = nn.Sequential(nn.Linear(input_size, output_size),
                                 nn.BatchNorm1d(output_size),
                                 nn.ReLU(),
                                 nn.Linear(output_size, output_size),
                                 nn.BatchNorm1d(output_size),
                                 nn.ReLU())

        self.hop1_edge_emb = torch.nn.Embedding(num_hop1_edge + 2, input_size, padding_idx=0)
        if self.K > 1:
            self.hopk_edge_emb = torch.nn.Embedding(num_pe + 2, input_size, padding_idx=0)
            self.hopk_node_path_emb = torch.nn.Embedding(num_pe, input_size, padding_idx=0)
            if combine == "attention":
                self.combine = AttentionCombine(self.output_size, self.K)
            elif combine == "geometric":
                self.combine = GeometricCombine(self.K, self.output_size)
            else:
                raise ValueError("Not implemented combine function")

        else:
            self.hopk_edge_emb = None
            self.combine = torch.squeeze

        self.reset_parameters()

    def reset_parameters(self):
        self.hop1_edge_emb.reset_parameters()
        self.mlp.apply(self.weights_init)
        if self.K > 1:
            self.hopk_edge_emb.reset_parameters()
            self.hopk_node_path_emb.reset_parameters()
            self.combine.reset_parameters()

    def weights_init(self, m):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def forward(self, x, edge_index, edge_attr, pe_attr=None, peripheral_attr=None):
        # N * K * H
        # h=x[:,0]
        e1_emb = self.hop1_edge_emb(edge_attr[:, :1])  # E * 1 * H
        if self.K > 1:
            if pe_attr is not None:
                pe = self.hopk_node_path_emb(pe_attr)
                x[:, 1:] = x[:, 1:] + pe
            ek_emb = self.hopk_edge_emb(edge_attr[:, 1:])  # E * K-1 * H
            e_emb = torch.cat([e1_emb, ek_emb], dim=-2)  # E * K * H
        else:
            e_emb = e1_emb

        x_n = self.propagate(edge_index, x=x, edge_emb=e_emb, mask=edge_attr)  # N * K * H
        # add peripheral subgraph information
        if peripheral_attr is not None:
            x_n = x_n + peripheral_attr
        h = self.mlp(self.combine(x_n))

        return h

    def message(self, x_j, edge_emb, mask):
        x_j = x_j + edge_emb  # E * K * dk
        mask = mask.unsqueeze(-1)  # E * K * 1
        return x_j.masked_fill_(mask == 0, 0.)

    def update(self, aggr_out):
        return F.gelu(aggr_out)
