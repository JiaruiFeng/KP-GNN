"""
Utils for defining model layers
"""
import torch


def degree(index,num_nodes,index_mask):
    """Compute degree in multi-hop setting
    Args:
        index(torch.tensor): index record the node at the end of edge
        num_nodes(int): number of nodes in the graph
        index_mask(torch.tensor): mask for each hop
    """
    #index E
    #index_mask E*K
    num_hop=index_mask.size(-1)
    index=index.unsqueeze(-1) #  E * 1
    index=index.tile([1,num_hop]) #  E * K
    out = torch.zeros((num_nodes,num_hop ), device=index.device) # N * K
    one = (index_mask>0).to(out.dtype) # E * K
    return out.scatter_add_(0, index, one)


