"""
Molecule embedding layer

"""

import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

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
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def reset_parameters(self):
        for emb in self.atom_embedding_list:
            torch.nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[-1]):
            x_embedding += self.atom_embedding_list[i](x[..., i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    """Bond encoder. Adapted from https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/mol_encoder.py
    Args:
        hidden_size: hidden dimension of embedding
    """

    def __init__(self, hidden_size):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            #add 2 for padding idx and self-loop
            emb = torch.nn.Embedding(dim+2, hidden_size,padding_idx=0)
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