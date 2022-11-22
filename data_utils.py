"""
utils for processing data used for training and evaluation
"""
import itertools
from copy import deepcopy as c

import networkx as nx
import numpy as np
import scipy.sparse as ssp
import torch
from scipy import linalg
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_scipy_sparse_matrix


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def extract_multi_hop_neighbors(data, K, max_edge_attr_num, max_hop_num,
                                max_edge_type, max_edge_count, max_distance_count, kernel):
    """generate multi-hop neighbors for input PyG graph using shortest path distance kernel
    Args:
        data(torch_geometric.data.Data): PyG graph data instance
        K(int): number of hop
        max_edge_attr_num(int): maximum number of encoding used for hopk edge
        max_hop_num(int): maximum number of hop to consider in computing node configuration of peripheral subgraph
        max_edge_type(int): maximum number of edge type to consider
        max_edge_count(int): maximum number of count for each type of edge
        max_distance_count(int): maximum number of count for each distance
        kernel (str): kernel used to extract neighbors
    """
    assert (isinstance(data, Data))
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes

    # graph with no edge
    if edge_index.size(1) == 0:
        edge_matrix_size = [num_nodes, K, max_edge_type, 2]
        configuration_matrix_size = [num_nodes, K, max_hop_num]
        peripheral_edge_matrix = torch.zeros(edge_matrix_size, dtype=torch.long)
        peripheral_configuration_matrix = torch.zeros(configuration_matrix_size, dtype=torch.long)
        data.peripheral_edge_attr = peripheral_edge_matrix
        data.peripheral_configuration = peripheral_configuration_matrix
        return data

    if "edge_attr" in data:
        edge_attr = data.edge_attr
    else:
        # skip 0, 1 as it is the mask and self-loop defined in the model
        edge_attr = (torch.ones([edge_index.size(-1)]) * 2).long()  # E

    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    edge_attr_adj = torch.from_numpy(to_scipy_sparse_matrix(edge_index, edge_attr, num_nodes).toarray()).long()
    # compute each order of adj
    adj_list = adj_K_order(adj, K)

    if kernel == "gd":
        # create K-hop edge with graph diffusion kernel
        final_adj = 0
        for adj_ in adj_list:
            final_adj += adj_
        final_adj[final_adj > 1] = 1
    else:
        # process adj list to generate shortest path distance matrix with path number
        exist_adj = c(adj_list[0])
        for i in range(1, len(adj_list)):
            adj_ = c(adj_list[i])
            # mask all the edge that already exist in previous hops
            adj_[exist_adj > 0] = 0
            exist_adj = exist_adj + adj_
            exist_adj[exist_adj > 1] = 1
            adj_list[i] = adj_
        # create K-hop edge with sortest path distance kernel
        final_adj = exist_adj

    g = nx.from_numpy_matrix(final_adj.numpy(), create_using=nx.DiGraph)
    edge_list = g.edges
    edge_index = torch.from_numpy(np.array(edge_list).T).long()

    hop1_edge_attr = edge_attr_adj[edge_index[0, :], edge_index[1, :]]
    edge_attr_list = [hop1_edge_attr.unsqueeze(-1)]
    pe_attr_list = []
    for i in range(1, len(adj_list)):
        adj_ = c(adj_list[i])
        adj_[adj_ > max_edge_attr_num] = max_edge_attr_num
        # skip 1 as it is the self-loop defined in the model
        adj_[adj_ > 0] = adj_[adj_ > 0] + 1
        adj_ = adj_.long()
        hopk_edge_attr = adj_[edge_index[0, :], edge_index[1, :]].unsqueeze(-1)
        edge_attr_list.append(hopk_edge_attr)
        pe_attr_list.append(torch.diag(adj_).unsqueeze(-1))
    edge_attr = torch.cat(edge_attr_list, dim=-1)  # E * K
    if K > 1:
        pe_attr = torch.cat(pe_attr_list, dim=-1)  # N * K-1
    else:
        pe_attr = None

    peripheral_edge_attr, peripheral_configuration_attr = get_peripheral_attr(adj_list, edge_attr_adj, max_hop_num,
                                                                              max_edge_type, max_edge_count,
                                                                              max_distance_count)
    # update all the attributes
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    data.peripheral_edge_attr = peripheral_edge_attr
    data.peripheral_configuration_attr = peripheral_configuration_attr
    data.pe_attr = pe_attr
    return data


def adj_K_order(adj, K):
    """compute the K order of adjacency given scipy matrix
    adj(coo_matrix): adjacency matrix
    K(int): number of hop
    """
    adj_list = [c(adj)]
    for i in range(K - 1):
        adj_ = adj_list[-1] @ adj
        adj_list.append(adj_)
    for i, adj_ in enumerate(adj_list):
        adj_ = torch.from_numpy(adj_.toarray()).int()
        # prevent the precision overflow
        # adj_[adj_<0]=1e8
        adj_.fill_diagonal_(0)
        adj_list[i] = adj_
    return adj_list


def get_peripheral_attr(adj_list, edge_attr_adj, max_hop_num,
                        max_edge_type, max_edge_count, max_distance_count):
    """Compute peripheral information for each node in graph
    Args:
        adj_list(list): adjacency matrix list of data for each hop
        edge_attr_adj(torch.tensor):edge feature matrix
        max_hop_num(int): maximum number of hop to consider in computing node configuration of peripheral subgraph
        max_edge_type(int): maximum number of edge type to consider
        max_edge_count(int): maximum number of count for each type of edge
        max_distance_count(int): maximum number of count for each distance
    """
    K = len(adj_list)
    num_nodes = edge_attr_adj.size(0)
    if max_hop_num > 0 and max_edge_type > 0:
        peripheral_edge_matrix_list = []
        peripheral_configuration_matrix_list = []
        for i in range(K):
            adj_ = c(adj_list[i])
            peripheral_edge_matrix, peripheral_configuration_matrix = extract_peripheral_attr_v2(edge_attr_adj, adj_,
                                                                                                 max_hop_num,
                                                                                                 max_edge_type,
                                                                                                 max_edge_count,
                                                                                                 max_distance_count)
            peripheral_edge_matrix_list.append(peripheral_edge_matrix)
            peripheral_configuration_matrix_list.append(peripheral_configuration_matrix)

        peripheral_edge_attr = torch.cat(peripheral_edge_matrix_list, dim=0)
        peripheral_configuration_attr = torch.cat(peripheral_configuration_matrix_list, dim=0)
        peripheral_edge_attr = peripheral_edge_attr.transpose(0, 1)  # N * K * c * f
        peripheral_configuration_attr = peripheral_configuration_attr.transpose(0, 1)  # N * K * c * f
    else:
        peripheral_edge_attr = None
        peripheral_configuration_attr = None

    return peripheral_edge_attr, peripheral_configuration_attr


#
# def extract_peripheral_edges(adj,edge_attr_adj,max_edge_num,max_component_num):
#     """extract peripheral edge information for each node using input adj and save edge attr information given edge attr adj
#     Args:
#         adj(torch.tensor): adjacency matrix
#         edge_attr_adj(torch.tensor) edge attr adjacency matrix
#         max_edge_num(int): maximum number of edge to keep
#     """
#     num_nodes=edge_attr_adj.size(0)
#     feature_size=edge_attr_adj[0,0].size() #f
#     if len(feature_size)==0:
#         feature_size=[1]
#         edge_attr_adj=edge_attr_adj.unsqueeze(-1)
#     peripheral_edge_list=[]
#     direct_edge_adj=(torch.sum(edge_attr_adj,dim=-1)>0).int() # N * N
#     if max_component_num==0:
#         component_dim=1
#     else:
#         component_dim=max_component_num
#     matrix_size=list(itertools.chain.from_iterable([[num_nodes,component_dim,max_edge_num],feature_size]))
#     peripheral_edge_matrix=torch.zeros(matrix_size,dtype=torch.long)
#     for i in range(num_nodes):
#         row=torch.where(adj[i]>0)[0]
#         # if node have no neighbors, skip
#         if len(row)<=1:
#             continue
#
#         peripheral_index=torch.combinations(row)
#         peripheral_edge_mask=direct_edge_adj[peripheral_index[:,0],peripheral_index[:,1]] # E'
#         peripheral_edge_list=peripheral_index[peripheral_edge_mask>0] # E'
#         if peripheral_edge_list.size(0)==0:
#             continue
#
#         g=nx.from_edgelist(peripheral_edge_list.numpy())
#         if max_component_num>0:
#             S = [g.subgraph(c).copy() for c in nx.connected_components(g)]
#             for j,s in enumerate(S):
#                 if j>=max_component_num:
#                     break
#                 s_edge_index=torch.from_numpy(np.array(s.edges).T).long()
#                 s_edge_attr=edge_attr_adj[s_edge_index[0,:],s_edge_index[1,:]]
#                 edge_num=s_edge_attr.size(0)
#                 if edge_num>max_edge_num:
#                     edge_num=max_edge_num
#                 peripheral_edge_matrix[i,j,:edge_num]=s_edge_attr[:edge_num]
#         else:
#             s=g
#             s_edge_index = torch.from_numpy(np.array(s.edges).T).long()
#             s_edge_attr = edge_attr_adj[s_edge_index[0, :], s_edge_index[1, :]]
#             edge_num = s_edge_attr.size(0)
#             if edge_num > max_edge_num:
#                 edge_num = max_edge_num
#             peripheral_edge_matrix[i, 0, :edge_num] = s_edge_attr[:edge_num]
#
#     return peripheral_edge_matrix

def feature_hashing(x, max_feature_size):
    """return unified index encoding given input feature tensor and maximum index for each feature dimension.
    Args:
        x(torch.tensor): input tensor with size of [...,f]
        maximum_feature_size: maximum feature size for each dimension [f]
    """

    assert x.size(-1) == max_feature_size.size(-1) - 1
    assert x.dtype in [torch.int, torch.long]
    index_increment = max_feature_size.cumprod(-1)[:-1]
    index_increment = index_increment.unsqueeze(0)
    x = x * index_increment
    return x.sum(-1)


def extract_peripheral_attr_v2(adj, k_adj, max_hop_num, max_edge_type, max_edge_count, max_distance_count):
    """extract peripheral attr information for each node using adj at this hop and original adj
    Args:
        adj(torch.tensor): adjacency matrix of original graph N*N, different number means different edge type
        k_adj(torch.tensor): adjacency matrix at the hop we want to extract peripheral information N*N
        max_hop_num(int): maximum number of hop to consider in computing node configuration of peripheral subgraph
        max_edge_type(int): maximum number of edge type to consider
        max_edge_count(int): maximum number of count for each type of edge
        max_distance_count(int): maximum number of count for each distance
    """
    num_nodes = adj.size(0)

    # component_dim=max_component_num
    # record peripheral edge information
    edge_matrix_size = [num_nodes, max_edge_type, 2]
    peripheral_edge_matrix = torch.zeros(edge_matrix_size, dtype=torch.long)
    # record node configuration
    configuration_matrix_size = [num_nodes, max_hop_num + 1]
    peripheral_configuration_matrix = torch.zeros(configuration_matrix_size, dtype=torch.long)
    for i in range(num_nodes):
        row = torch.where(k_adj[i] > 0)[0]
        # subgrapb with less than 2 nodes, no edges, thus skip
        num_sub_nodes = row.size(-1)
        if num_sub_nodes < 2:
            continue
        peripheral_subgraph = adj[row][:, row]
        s = nx.from_numpy_matrix(peripheral_subgraph.numpy(), create_using=nx.DiGraph)
        s_edge_list = list(nx.get_edge_attributes(s, "weight").values())
        if len(s_edge_list) == 0:
            continue
        s_edge_list = torch.tensor(s_edge_list).long()
        edge_count = torch.bincount(s_edge_list, minlength=max_edge_type + 2)
        # remove 0 and 1
        edge_count = edge_count[2:]
        sort_count, sort_type = torch.sort(edge_count, descending=True)
        sort_count = sort_count[:max_edge_type]
        sort_type = sort_type[:max_edge_type]
        sort_count[sort_count > max_edge_count] = max_edge_count
        peripheral_edge_matrix[i, :, 0] = sort_type
        peripheral_edge_matrix[i, :, 1] = sort_count
        shortest_path_matrix = nx_compute_shortest_path_length(s, max_length=max_hop_num)
        num_sub_p_edges = 0
        for j in range(num_sub_nodes):
            for h in range(1, max_hop_num + 1):
                h_nodes = torch.where(shortest_path_matrix[j] == h)[0]
                if h_nodes.size(-1) < 2:
                    continue
                else:
                    pp_subgraph = peripheral_subgraph[h_nodes][:, h_nodes]
                    num_sub_p_edges += torch.sum(pp_subgraph)

        configuration_feature = torch.bincount(shortest_path_matrix.view(-1), minlength=max_hop_num + 1)
        # configuration_feature=configuration_feature[1:]
        configuration_feature[0] = num_sub_p_edges
        configuration_feature[configuration_feature > max_distance_count] = max_distance_count
        peripheral_configuration_matrix[i, :] = configuration_feature
    return peripheral_edge_matrix.unsqueeze(0), peripheral_configuration_matrix.unsqueeze(0)


def nx_compute_shortest_path_length(G, max_length):
    """Compute all pair shortest path length in the graph
    Args:
        G(networkx): input graph
        max_length(int): max length when computing shortest path

    """
    num_node = G.number_of_nodes()
    shortest_path_length_matrix = torch.zeros([num_node, num_node]).int()
    all_shortest_path_lengths = nx.all_pairs_shortest_path_length(G, max_length)
    for shortest_path_lengths in all_shortest_path_lengths:
        index, path_lengths = shortest_path_lengths
        for end_node, path_length in path_lengths.items():
            if end_node == index:
                continue
            else:
                shortest_path_length_matrix[index, end_node] = path_length
    return shortest_path_length_matrix


def to_dense_edge_feature(edge_feature, edge_index, num_nodes):
    """
    convert edge feature to dense adj
    """
    edge_feature = edge_feature.squeeze()
    K = list(edge_feature.size()[1:])
    adj = torch.zeros(list(itertools.chain.from_iterable([[num_nodes], [num_nodes], K])), dtype=edge_feature.dtype)
    for i in range(edge_index.size(-1)):
        v = edge_index[0, i]
        u = edge_index[1, i]
        adj[v, u] = edge_feature[i]

    return adj


def PyG_collate(examples):
    """PyG collcate function
    Args:
        examples(list): batch of samples
    """
    data = Batch.from_data_list(examples)
    return data


def PyG_collate_new(examples):
    data = Batch.from_data_list(examples)
    num_data_nodes = data.num_data_nodes
    node_to_batch = list(itertools.chain.from_iterable(
        [[i for _ in range(num_node.item())] for i, num_node in enumerate(num_data_nodes)]))
    data.batch = torch.tensor(node_to_batch)
    return data


def resistance_distance(data):
    """resistance distance.See "Link prediction in complex networks: A survey".
    Adapted from NestedGNN:https://github.com/muhanzhang/NestedGNN
    Args:
        data(PyG.Data):pyg data object
    """

    edge_index = data.edge_index
    num_nodes = data.num_nodes
    adj = to_scipy_sparse_matrix(
        edge_index, num_nodes=num_nodes
    ).tocsr()
    laplacian = ssp.csgraph.laplacian(adj).toarray()
    try:
        L_inv = linalg.pinv(laplacian)
    except:
        laplacian += 0.01 * np.eye(*laplacian.shape)
    lxx = L_inv[0, 0]
    lyy = L_inv[list(range(len(L_inv))), list(range(len(L_inv)))]
    lxy = L_inv[0, :]
    lyx = L_inv[:, 0]
    rd_to_x = torch.FloatTensor((lxx + lyy - lxy - lyx)).unsqueeze(1)
    data.rd = rd_to_x
    return data


def post_transform(wo_path_encoding, wo_edge_feature):
    if wo_path_encoding and wo_edge_feature:
        def transform(g):
            edge_attr = g.edge_attr
            edge_attr[edge_attr > 2] = 2
            g.edge_attr = edge_attr
            if "pe_attr" in g:
                pe_attr = g.pe_attr
                pe_attr[pe_attr > 0] = 0
                g.pe_attr = pe_attr
            return g
    elif wo_edge_feature:
        def transform(g):
            edge_attr = g.edge_attr
            t = edge_attr[:, 0]
            t[t > 2] = 2
            edge_attr[:, 0] = t
            g.edge_attr = edge_attr
            return g

    elif wo_path_encoding:
        def transform(g):
            edge_attr = g.edge_attr
            t = edge_attr[:, 1:]
            t[t > 2] = 2
            edge_attr[:, 1:] = t
            g.edge_attr = edge_attr
            if "pe_attr" in g:
                pe_attr = g.pe_attr
                pe_attr[pe_attr > 0] = 0
                g.pe_attr = pe_attr
            return g
    else:
        def transform(g):
            return g

    return transform
