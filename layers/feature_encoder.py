"""
Input feature embedding layer
"""

import torch
import torch.nn as nn


class FeatureSumEncoder(nn.Module):
    """General Feature encoder with summation for different feature
    Args:
        feature_dims (list): a list of dim of input feature
        hidden_size (int): hidden dimension of embedding
    """

    def __init__(self, feature_dims, hidden_size, padding=False):
        super(FeatureSumEncoder, self).__init__()
        self.embedding_list = nn.ModuleList()
        for i, dim in enumerate(feature_dims):
            if padding:
                emb = nn.Embedding(dim, hidden_size, padding_idx=0)
            else:
                emb = nn.Embedding(dim, hidden_size)
            self.embedding_list.append(emb)

    def reset_parameters(self):
        for emb in self.embedding_list:
            emb.reset_parameters()

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[-1]):
            x_embedding += self.embedding_list[i](x[..., i])
        return x_embedding


class FeatureConcatEncoder(nn.Module):
    """General Feature encoder with concatenation for different feature
    Args:
        feature_dims (list): a list of dim of input feature
        hidden_size (int): hidden dimension of embedding
    """

    def __init__(self, feature_dims, hidden_size, padding=False):
        super(FeatureConcatEncoder, self).__init__()

        self.embedding_list = nn.ModuleList()

        for i, dim in enumerate(feature_dims):
            if padding:
                emb = nn.Embedding(dim, hidden_size, padding_idx=0)
            else:
                emb = nn.Embedding(dim, hidden_size)
            self.embedding_list.append(emb)
        self.proj = nn.Linear(len(feature_dims) * hidden_size, hidden_size)

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

