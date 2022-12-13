"""
GINE layer
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class GINEConv(MessagePassing):
    """GINE layer, adapted from PyG.
    Args:
        input_size (int): input feature size
        output_size (int): output feature size
        eps (int): epsilon
        num_hop1_edge (int): number of edge type at 1 hop
        train_eps (bool): If true, the epsilon is trainable
    """

    def __init__(self, input_size, output_size, eps=0., num_hop1_edge=1, train_eps=False):
        super(GINEConv, self).__init__(node_dim=0)
        self.input_size = input_size
        self.output_size = output_size
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.mlp = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )
        self.hop1_edge_emb = torch.nn.Embedding(num_hop1_edge + 2, self.input_size, padding_idx=0)
        self.reset_parameters()

    def weights_init(self, m):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.mlp.apply(self.weights_init)
        self.hop1_edge_emb.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_attr):
        x = x.view(-1, 1, self.input_size)  # N * 1 * h
        e_emb = self.hop1_edge_emb(edge_attr)  # E * 1 * h
        out = self.propagate(edge_index, x=x, edge_emb=e_emb, mask=edge_attr)
        out = out + (1 + self.eps) * x
        return self.mlp(out.squeeze())

    def message(self, x_j, edge_emb, mask):
        x_j = x_j + edge_emb  # E * 1 * dk
        mask = mask.unsqueeze(-1)  # E * 1 * 1
        return x_j.masked_fill_(mask == 0, 0.)
