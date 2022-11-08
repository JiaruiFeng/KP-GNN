"""
different encoder for different dataset
"""

import torch
import torch.nn as nn
from ogb.utils.features import get_atom_feature_dims
full_atom_feature_dims = get_atom_feature_dims()



class EmbeddingEncoder(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(EmbeddingEncoder, self).__init__()
        self.init_proj=nn.Embedding(input_size,hidden_size)

    def reset_parameters(self):
        self.init_proj.reset_parameters()

    def forward(self,data):
        return self.init_proj(data.x)

class LinearEncoder(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(LinearEncoder, self).__init__()
        self.init_proj=nn.Linear(input_size,hidden_size)

    def reset_parameters(self):
        self.init_proj.reset_parameters()

    def forward(self,data):
        return self.init_proj(data.x)


class QM9InputEncoder(nn.Module):
    def __init__(self,hidden_size,use_pos=False):
        super(QM9InputEncoder, self).__init__()
        self.use_pos=use_pos
        if use_pos:
            input_size=22
        else:
            input_size=19
        self.init_proj=nn.Linear(input_size,hidden_size)
        self.z_embedding = nn.Embedding(1000, 8)


    def reset_parameters(self):
        self.init_proj.reset_parameters()
        self.z_embedding.reset_parameters()

    def forward(self,data):
        x=data.x
        z=data.z
        if "pos" in data:
            pos=data.pos
        else:
            pos=None

        z_emb = 0
        if z is not None:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)

        # concatenate with continuous node features
        x = torch.cat([z_emb, x], -1)

        if self.use_pos:
            x = torch.cat([x, pos], 1)

        x=self.init_proj(x)
        return x


class AtomEncoder(torch.nn.Module):
    """Atom encoder. Adapted from https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/mol_encoder.py
    Args:
        hidden_size: hidden dimension of embedding

    """
    def __init__(self, hidden_size):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, hidden_size)
            self.atom_embedding_list.append(emb)

    def reset_parameters(self):
        for emb in self.atom_embedding_list:
            emb.reset_parameters()

    def forward(self, data):
        x=data.x
        x_embedding = 0
        for i in range(x.shape[-1]):
            x_embedding += self.atom_embedding_list[i](x[..., i])

        return x_embedding
