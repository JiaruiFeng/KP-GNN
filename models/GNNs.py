"""
General GNN framework
"""
from copy import deepcopy as c

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, LayerNorm, InstanceNorm, PairNorm, GraphSizeNorm, global_add_pool

from layers.feature_encoder import FeatureConcatEncoder


def clones(module, N):
    """Layer clone function, used for concise code writing
    Args:
        module: the layer want to clone
        N: the time of clone
    """
    return nn.ModuleList(c(module) for _ in range(N))


class GNN(nn.Module):
    """A generalized GNN framework
    Args:
        num_layer(int): the number of GNN layer
        gnn_layer(nn.Module): gnn layer used in GNN model
        init_emb(nn.Module): initial node feature encoding
        num_hop1_edge(int): number of edge type at 1 hop
        max_edge_count(int): maximum count per edge type for encoding
        max_hop_num(int): maximum number of hop to consider in peripheral node configuration
        max_distance_count(int): maximum count per hop for encoding
        JK(str):method of jumping knowledge, last,concat,max or sum
        norm_type(str): method of normalization, batch or layer
        virtual_node(bool): whether to add virtual node in the model
        residual(bool): whether to add residual connection
        use_rd(bool): whether to add resitance distance as additional feature
        wo_peripheral_edge(bool): If true, remove peripheral edge information from model
        wo_oeripheral_configuration(bool): If true, remove peripheral node configuration from model
        drop_prob (float): dropout rate
    """

    def __init__(self, num_layer, gnn_layer, init_emb, num_hop1_edge, max_edge_count, max_hop_num, max_distance_count,
                 JK="last", norm_type="batch", virtual_node=True,
                 residual=False, use_rd=False, wo_peripheral_edge=False, wo_peripheral_configuration=False,
                 drop_prob=0.1):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.hidden_size = gnn_layer.output_size
        self.K = gnn_layer.K
        self.output_dk = gnn_layer.output_dk
        self.dropout = nn.Dropout(drop_prob)
        self.JK = JK
        self.residual = residual
        self.use_rd = use_rd
        self.virtual_node = virtual_node
        self.wo_peripheral_edge = wo_peripheral_edge
        self.wo_peripheral_configuration = wo_peripheral_configuration
        if self.JK == "concat":
            self.output_proj = nn.Sequential(nn.Linear((self.num_layer + 1) * self.hidden_size, self.hidden_size),
                                             nn.ReLU(), nn.Dropout(drop_prob))
        else:
            self.output_proj = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
                                             nn.Dropout(drop_prob))

        if self.JK == "attention":
            self.attention_lstm = nn.LSTM(self.hidden_size, self.num_layer, 1, batch_first=True, bidirectional=True,
                                          dropout=0.)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # embedding start from 1
        self.init_proj = init_emb
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, self.hidden_size)
        if self.virtual_node:
            # set the initial virtual node embedding to 0.
            self.virtualnode_embedding = torch.nn.Embedding(1, self.hidden_size)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            # List of MLPs to transform virtual node at every layer
            self.mlp_virtualnode_list = torch.nn.ModuleList()
            for layer in range(num_layer - 1):
                self.mlp_virtualnode_list.append(torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size, self.hidden_size),
                    torch.nn.BatchNorm1d(self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size, self.hidden_size),
                    torch.nn.BatchNorm1d(self.hidden_size),
                    torch.nn.ReLU()))

        if not wo_peripheral_edge:
            edge_feature_dims = [num_hop1_edge + 2, max_edge_count + 1]
            self.peripheral_edge_embedding = FeatureConcatEncoder(edge_feature_dims, self.output_dk, padding=0)
            self.pew = nn.Parameter(torch.rand(1), requires_grad=True)
        if not wo_peripheral_configuration:
            configuration_feature_dims = [max_distance_count + 1 for _ in range(max_hop_num + 1)]
            self.peripheral_configuration_embedding = FeatureConcatEncoder(configuration_feature_dims, self.output_dk,
                                                                           padding=0)
            self.pcw = nn.Parameter(torch.rand(1), requires_grad=True)

        # gnn layer list
        self.gnns = clones(gnn_layer, num_layer)
        # norm list
        if norm_type == "Batch":
            self.norms = clones(BatchNorm(self.hidden_size), num_layer)
        elif norm_type == "Layer":
            self.norms = clones(LayerNorm(self.hidden_size), num_layer)
        elif norm_type == "Instance":
            self.norms = clones(InstanceNorm(self.hidden_size), num_layer)
        elif norm_type == "GraphSize":
            self.norms = clones(GraphSizeNorm(), num_layer)
        elif norm_type == "Pair":
            self.norms = clones(PairNorm(), num_layer)
        else:
            raise ValueError("Not supported norm method")

        self.reset_parameters()

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def reset_parameters(self):
        self.init_proj.reset_parameters()
        for g in self.gnns:
            g.reset_parameters()
        if self.JK == "attention":
            self.attention_lstm.reset_parameters()

        self.output_proj.apply(self.weights_init)
        if self.use_rd:
            self.rd_projection.reset_parameters()
        if self.virtual_node:
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            self.mlp_virtualnode_list.apply(self.weights_init)
        if not self.wo_peripheral_edge:
            self.peripheral_edge_embedding.reset_parameters()
            nn.init.normal_(self.pew)
        if not self.wo_peripheral_configuration:
            self.peripheral_configuration_embedding.reset_parameters()
            nn.init.normal_(self.pcw)

    def forward(self, data):

        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch
        if "peripheral_edge_attr" in data:
            peripheral_edge_attr = data.peripheral_edge_attr
        else:
            peripheral_edge_attr = None

        if "peripheral_configuration_attr" in data:
            peripheral_configuration_attr = data.peripheral_configuration_attr
        else:
            peripheral_configuration_attr = None

        if "pe_attr" in data:
            pe_attr = data.pe_attr
        else:
            pe_attr = None

        if "rd" in data:
            rd = data.rd
        else:
            rd = None

        # initial projection
        x = self.init_proj(data).squeeze()
        num_nodes = x.size(0)

        if self.use_rd and rd is not None:
            rd_proj = self.rd_projection(rd).squeeze()
            x = x + rd_proj

        peripheral_attr = torch.zeros([num_nodes, self.K, self.output_dk], device=x.device, dtype=x.dtype)
        if (not self.wo_peripheral_edge) and peripheral_edge_attr is not None:
            peripheral_edge_emb = self.peripheral_edge_embedding(peripheral_edge_attr)  # N * K * E' * H
            peripheral_attr += torch.sigmoid(self.pew) * peripheral_edge_emb.sum(-2)  # N * K * H

        if (not self.wo_peripheral_configuration) and peripheral_configuration_attr is not None:
            peripheral_attr += torch.sigmoid(self.pcw) * self.peripheral_configuration_embedding(
                peripheral_configuration_attr)  # N * K * H
        if self.virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
            )

        # forward in gnn layer
        h_list = [x]
        for l in range(self.num_layer):
            if self.virtual_node:
                h_list[l] = h_list[l] + virtualnode_embedding[batch]
            h = self.gnns[l](h_list[l], edge_index, edge_attr, pe_attr, peripheral_attr)
            h = self.norms[l](h)
            # if not the last gnn layer, add dropout layer
            if l != self.num_layer - 1:
                h = self.dropout(h)

            if self.residual:
                h = h + h_list[l]

            h_list.append(h)

            if self.virtual_node:
                # update the virtual nodes
                if l < self.num_layer - 1:
                    virtualnode_embedding_temp = global_add_pool(
                        h_list[l], batch
                    ) + virtualnode_embedding
                    # transform virtual nodes using MLP

                    if self.residual:
                        virtualnode_embedding = virtualnode_embedding + self.dropout(
                            self.mlp_virtualnode_list[l](virtualnode_embedding_temp))
                    else:
                        virtualnode_embedding = self.dropout(self.mlp_virtualnode_list[l](virtualnode_embedding_temp))

        # JK connection
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(-1) for h in h_list]
            node_representation = F.max_pool1d(torch.cat(h_list, dim=-1), kernel_size=self.num_layer + 1).squeeze()
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)
        elif self.JK == "attention":
            h_list = [h.unsqueeze(0) for h in h_list]
            h_list = torch.cat(h_list, dim=0).transpose(0, 1)  # N *num_layer * H
            self.attention_lstm.flatten_parameters()
            attention_score, _ = self.attention_lstm(h_list)  # N * num_layer * 2*num_layer
            attention_score = torch.softmax(torch.sum(attention_score, dim=-1), dim=1).unsqueeze(
                -1)  # N * num_layer  * 1
            node_representation = torch.sum(h_list * attention_score, dim=1)

        return self.output_proj(node_representation)


class GNNPlus(nn.Module):
    """A generalized GNN framework with GINE+ color refinement
    Args:
        num_layer(int): the number of GNN layer
        gnn_layer(nn.Module): gnn layer used in GNN model
        init_emb(nn.Module): initial node feature encoding
        num_hop1_edge(int): number of edge type at 1 hop
        max_edge_count(int): maximum count per edge type for encoding
        max_hop_num(int): maximum number of hop to consider in peripheral node configuration
        max_distance_count(int): maximum count per hop for encoding
        JK(str):method of jumping knowledge, last,concat,max or sum
        norm_type(str): method of normalization, batch or layer
        virtual_node(bool): whether to add virtual node in the model
        residual(bool): whether to add residual connection
        use_rd(bool): whether to add resitance distance as additional feature
        wo_peripheral_edge(bool): If true, remove peripheral edge information from model
        wo_oeripheral_configuration(bool): If true, remove peripheral node configuration from model
        drop_prob (float): dropout rate
    """

    def __init__(self, num_layer, gnn_layer, init_emb, num_hop1_edge, max_edge_count, max_hop_num, max_distance_count,
                 JK="last", norm_type="batch", virtual_node=True,
                 residual=False, use_rd=False, wo_peripheral_edge=False, wo_peripheral_configuration=False,
                 drop_prob=0.1):
        super(GNNPlus, self).__init__()
        self.num_layer = num_layer
        self.hidden_size = gnn_layer[-1].output_size
        self.K = gnn_layer[-1].K
        self.dropout = nn.Dropout(drop_prob)
        self.JK = JK
        self.residual = residual
        self.use_rd = use_rd
        self.virtual_node = virtual_node
        self.wo_peripheral_edge = wo_peripheral_edge
        self.wo_peripheral_configuration = wo_peripheral_configuration
        if self.JK == "concat":
            self.output_proj = nn.Sequential(nn.Linear((self.num_layer + 1) * self.hidden_size, self.hidden_size),
                                             nn.ReLU(), nn.Dropout(drop_prob))
        else:
            self.output_proj = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
                                             nn.Dropout(drop_prob))

        if self.JK == "attention":
            self.attention_lstm = nn.LSTM(self.hidden_size, self.num_layer, 1, batch_first=True, bidirectional=True,
                                          dropout=0.)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # embedding start from 1
        self.init_proj = init_emb

        if self.use_rd:
            self.rd_projection = nn.Linear(1, self.hidden_size)

        if self.virtual_node:
            # set the initial virtual node embedding to 0.
            self.virtualnode_embedding = torch.nn.Embedding(1, self.hidden_size)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            # List of MLPs to transform virtual node at every layer
            self.mlp_virtualnode_list = torch.nn.ModuleList()
            for layer in range(num_layer - 1):
                self.mlp_virtualnode_list.append(torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size, self.hidden_size),
                    torch.nn.BatchNorm1d(self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size, self.hidden_size),
                    torch.nn.BatchNorm1d(self.hidden_size),
                    torch.nn.ReLU()))

        if not wo_peripheral_edge:
            edge_feature_dims = [num_hop1_edge + 2, max_edge_count + 1]
            self.peripheral_edge_embedding = FeatureConcatEncoder(edge_feature_dims, self.hidden_size, padding=0)
            self.pew = nn.Parameter(torch.rand(1), requires_grad=True)
        if not wo_peripheral_configuration:
            configuration_feature_dims = [max_distance_count + 1 for _ in range(max_hop_num + 1)]
            self.peripheral_configuration_embedding = FeatureConcatEncoder(configuration_feature_dims, self.hidden_size,
                                                                           padding=0)
            self.pcw = nn.Parameter(torch.rand(1), requires_grad=True)

        # gnn layer list
        self.gnns = nn.ModuleList(gnn_layer)

        # norm list
        if norm_type == "Batch":
            self.norms = clones(BatchNorm(self.hidden_size), num_layer)
        elif norm_type == "Layer":
            self.norms = clones(LayerNorm(self.hidden_size), num_layer)
        elif norm_type == "Instance":
            self.norms = clones(InstanceNorm(self.hidden_size), num_layer)
        elif norm_type == "GraphSize":
            self.norms = clones(GraphSizeNorm(), num_layer)
        elif norm_type == "Pair":
            self.norms = clones(PairNorm(), num_layer)
        else:
            raise ValueError("Not supported norm method")

        self.reset_parameters()

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def reset_parameters(self):
        self.init_proj.reset_parameters()
        if self.JK == "attention":
            self.attention_lstm.reset_parameters()
        self.output_proj.apply(self.weights_init)
        if self.use_rd:
            self.rd_projection.reset_parameters()
        if self.virtual_node:
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            self.mlp_virtualnode_list.apply(self.weights_init)
        if not self.wo_peripheral_edge:
            self.peripheral_edge_embedding.reset_parameters()
            nn.init.normal_(self.pew)
        if not self.wo_peripheral_configuration:
            self.peripheral_configuration_embedding.reset_parameters()
            nn.init.normal_(self.pcw)

        for g in self.gnns:
            g.reset_parameters()

    def forward(self, data):

        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch
        if "peripheral_edge_attr" in data:
            peripheral_edge_attr = data.peripheral_edge_attr
        else:
            peripheral_edge_attr = None

        if "peripheral_configuration_attr" in data:
            peripheral_configuration_attr = data.peripheral_configuration_attr
        else:
            peripheral_configuration_attr = None
        if "pe_attr" in data:
            pe_attr = data.pe_attr
        else:
            pe_attr = None

        if "rd" in data:
            rd = data.rd
        else:
            rd = None

        # initial projection
        x = self.init_proj(data).squeeze()

        num_nodes = x.size(0)

        if self.use_rd and rd is not None:
            rd_proj = self.rd_projection(rd).squeeze()
            x = rd_proj + x

        peripheral_attr = torch.zeros([num_nodes, self.K, self.hidden_size], device=x.device, dtype=x.dtype)
        if (not self.wo_peripheral_edge) and peripheral_edge_attr is not None:
            peripheral_edge_emb = self.peripheral_edge_embedding(peripheral_edge_attr)  # N * K * E' * H
            peripheral_attr += torch.tanh(self.pew) * peripheral_edge_emb.sum(-2)  # N * K * H

        if (not self.wo_peripheral_configuration) and peripheral_configuration_attr is not None:
            peripheral_attr += torch.tanh(self.pcw) * self.peripheral_configuration_embedding(
                peripheral_configuration_attr)  # N * K * H

        if self.virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
            )

        # forward in gnn layer
        h_list = [x]
        last_h = x
        for l in range(self.num_layer):
            if self.virtual_node:
                h_list[l] = h_list[l] + virtualnode_embedding[batch]
            x_list = []
            end = -1 if l + 1 <= self.K else l - self.K
            for j in range(l, end, -1):
                x_list.append(h_list[j].unsqueeze(1))
            x = torch.cat(x_list, dim=1)
            k = l + 1 if l + 1 <= self.K else self.K
            if peripheral_attr is not None:
                pak = peripheral_attr[:, :k]
            else:
                pak = None

            if pe_attr is not None:
                pek = pe_attr[:, :k - 1]
            else:
                pek = None

            h = self.gnns[l](x, edge_index, edge_attr[:, :k], pek, pak)
            h = self.norms[l](h)
            # if not the last gnn layer, add dropout layer
            if l != self.num_layer:
                h = self.dropout(h)
            if self.residual:
                h = h + last_h
                last_h = h

            h_list.append(h)

            if self.virtual_node:
                # update the virtual nodes
                if l < self.num_layer - 1:
                    virtualnode_embedding_temp = global_add_pool(
                        h_list[l], batch
                    ) + virtualnode_embedding
                    # transform virtual nodes using MLP

                    if self.residual:
                        virtualnode_embedding = virtualnode_embedding + self.dropout(
                            self.mlp_virtualnode_list[l](virtualnode_embedding_temp))
                    else:
                        virtualnode_embedding = self.dropout(self.mlp_virtualnode_list[l](virtualnode_embedding_temp))

        # JK connection
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(-1) for h in h_list]
            node_representation = F.max_pool1d(torch.cat(h_list, dim=-1), kernel_size=self.num_layer + 1).squeeze()
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)
        elif self.JK == "attention":
            h_list = [h.unsqueeze(0) for h in h_list]
            h_list = torch.cat(h_list, dim=0).transpose(0, 1)  # N *num_layer * H
            self.attention_lstm.flatten_parameters()
            attention_score, _ = self.attention_lstm(h_list)  # N * num_layer * 2*num_layer
            attention_score = torch.softmax(torch.sum(attention_score, dim=-1), dim=1).unsqueeze(
                -1)  # N * num_layer  * 1
            node_representation = torch.sum(h_list * attention_score, dim=1)

        return self.output_proj(node_representation)
