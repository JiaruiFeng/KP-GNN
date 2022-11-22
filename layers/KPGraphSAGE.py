"""
KP-GNN_GraphSAGE layer
"""
import math

import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from .combine import *


class KPGraphSAGEConv(MessagePassing):
    """
    KP-GNN with GraphSAGE kernel
    Args:
        input_size(int): the size of input feature
        output_size(int): the size of output feature
        K(int): number of hop to consider in Convolution layer
        aggr(str): The aggregation function, default is mean for GraphSAGE
        num_hop1_edge(int): number of edge type at 1 hop
        num_pe(int): maximum number of path encoding, larger or equal to 1
        combine(str): combination method for information in different hop. select from(geometric, attention)

    """

    def __init__(self, input_size, output_size, K, aggr="mean", num_hop1_edge=1, num_pe=1, combine="geometric"):
        super(KPGraphSAGEConv, self).__init__(node_dim=0)
        self.aggr = aggr
        self.K = K
        assert input_size % K == 0
        assert output_size % K == 0
        self.input_dk = input_size // K
        self.output_dk = output_size // K
        self.output_size = output_size
        self.hop_proj = torch.nn.Parameter(torch.Tensor(self.K, 2 * self.input_dk, self.output_dk))
        self.hop_bias = torch.nn.Parameter(torch.Tensor(self.K, self.output_dk))

        # edge embedding for 1-hop and k-hop
        # Notice that in hop, there is no actually edge feature, therefore need addtional embedding layer to encode
        # self defined features like path encoding
        self.hop1_edge_emb = torch.nn.Embedding(num_hop1_edge + 2, self.input_dk, padding_idx=0)

        # If K larger than 1, define additional embedding and combine function
        if self.K > 1:
            self.combine_proj = nn.Linear(self.output_dk, output_size)
            self.hopk_edge_emb = torch.nn.Embedding(num_pe + 2, self.input_dk, padding_idx=0)
            self.hopk_node_path_emb = torch.nn.Embedding(num_pe, self.input_dk, padding_idx=0)
            if combine == "attention":
                self.combine = AttentionCombine(self.output_dk, self.K)
            elif combine == "geometric":
                self.combine = GeometricCombine(self.K, self.output_dk)
            else:
                raise ValueError("Not implemented combine function")

        else:
            self.hopk_edge_emb = None
            self.combine = torch.squeeze
            self.combine_proj = nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        self.hop1_edge_emb.reset_parameters()
        if self.K > 1:
            self.hopk_edge_emb.reset_parameters()
            self.hopk_node_path_emb.reset_parameters()
            self.combine.reset_parameters()

        nn.init.kaiming_uniform_(self.hop_proj)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.hop_proj)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.hop_bias, -bound, bound)
        if isinstance(self.combine_proj, nn.Linear):
            self.combine_proj.reset_parameters()

    def forward(self, x, edge_index, edge_attr, pe_attr=None, peripheral_attr=None):

        x = x.view(-1, self.K, self.input_dk)  # N * K * dk

        # embedding of edge
        e1_emb = self.hop1_edge_emb(edge_attr[:, :1])  # E * 1 * dk
        if self.K > 1:
            if pe_attr is not None:
                pe = self.hopk_node_path_emb(pe_attr)
                x[:, 1:] = x[:, 1:] + pe
            ek_emb = self.hopk_edge_emb(edge_attr[:, 1:])  # E * K-1 * dk
            e_emb = torch.cat([e1_emb, ek_emb], dim=-2)  # E * K * dk
        else:
            e_emb = e1_emb

        x_n = self.propagate(edge_index, x=x, edge_emb=e_emb, mask=edge_attr)  # N * K * dk
        # add peripheral subgraph information
        if peripheral_attr is not None:
            x_n = x_n + peripheral_attr

        x = torch.cat([x, x_n], dim=-1).permute(1, 0, 2)  # K * N * 2dk
        x = torch.matmul(x, self.hop_proj) + self.hop_bias.unsqueeze(1)
        x = x.permute(1, 0, 2)  # N * K * H
        x = F.normalize(F.relu(x), p=2, dim=-1)  # N * K * H
        # combine
        x = self.combine_proj(self.combine(x))

        return x

    def message(self, x_j, edge_emb, mask):
        x_j = x_j + edge_emb  # E * K * H
        mask = mask.unsqueeze(-1)  # E * K * 1
        return x_j.masked_fill_(mask == 0, 0.)

    def update(self, aggr_out):
        return aggr_out
