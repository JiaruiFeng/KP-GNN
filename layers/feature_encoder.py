"""
Feature embedding layer

"""

import torch
from ogb.utils.features import get_bond_feature_dims

full_bond_feature_dims = get_bond_feature_dims()


class FeatureSumEncoder(torch.nn.Module):
    """General Feature encoder with summation for different feature
    Args:
        hidden_size: hidden dimension of embedding

    """

    def __init__(self, feature_dims, hidden_size, padding=False):
        super(FeatureSumEncoder, self).__init__()

        self.embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(feature_dims):
            if padding:
                emb = torch.nn.Embedding(dim, hidden_size, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, hidden_size)
            self.embedding_list.append(emb)

    def reset_parameters(self):
        for emb in self.embedding_list:
            emb.reset_parameters()

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[-1]):
            x_embedding += self.embedding_list[i](x[..., i])
        return x_embedding


class FeatureConcatEncoder(torch.nn.Module):
    """General Feature encoder with concatenation for different feature
    Args:
        hidden_size: hidden dimension of embedding

    """

    def __init__(self, feature_dims, hidden_size, padding=False):
        super(FeatureConcatEncoder, self).__init__()

        self.embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(feature_dims):
            if padding:
                emb = torch.nn.Embedding(dim, hidden_size, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, hidden_size)
            self.embedding_list.append(emb)
        self.proj = torch.nn.Linear(len(feature_dims) * hidden_size, hidden_size)

    def reset_parameters(self):
        for emb in self.embedding_list:
            emb.reset_parameters()
        self.proj.reset_parameters()

    def forward(self, x):
        x_embeddings = []
        for i in range(x.shape[-1]):
            x_embeddings.append(self.embedding_list[i](x[..., i]))
        x_embeddings = torch.cat(x_embeddings, dim=-1)
        return self.proj(x_embeddings)


class BondEncoder(torch.nn.Module):
    """Bond encoder. Adapted from https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/mol_encoder.py
    Args:
        hidden_size: hidden dimension of embedding
    """

    def __init__(self, hidden_size):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            # add 2 for padding idx and self-loop
            emb = torch.nn.Embedding(dim + 2, hidden_size, padding_idx=0)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def reset_parameters(self):
        for emb in self.bond_embedding_list:
            torch.nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[-1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[..., i])

        return bond_embedding
