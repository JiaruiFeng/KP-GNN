"""
utils for processing data used for training and evaluation
"""
from copy import deepcopy as c
from torch_geometric.data import Data,Batch
from torch_geometric.utils import to_scipy_sparse_matrix
import networkx as nx
import torch
import numpy as np
import itertools
import scipy.sparse as ssp
from scipy import linalg


def maybe_num_nodes(index, num_nodes=None):
	return index.max().item() + 1 if num_nodes is None else num_nodes



def multi_hop_neighbors_with_gd_kernel(data,K,max_edge_attr_num,max_peripheral_edge_num,max_component_num=5,use_edge_feature=False):
    """generate multi-hop neighbors for input PyG graph using graph diffusion kernel
    Args:
        data(torch_geometric.data.Data): PyG graph data instance
        K(int): number of hop
        max_edge_attr_num(int): maximum number of encoding used for hopk edge
        max_peripheral_edge_num(int): maximum number of peripheral edge to stroe
        use_edge_feature(bool): whether to use edge feature
    """
    assert(isinstance(data, Data))
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes
    if edge_index.size(1)==0:
        if "bond_feature" in data:
            feature_size=data.bond_feature.size(-1)
        elif "edge_attr" in data and use_edge_feature:
            feature_size=data.edge_attr.size(-1)
        else:
            feature_size=1
        matrix_size=list(itertools.chain.from_iterable([[num_nodes,max_component_num,max_peripheral_edge_num],[feature_size]]))
        peripheral_edge_matrix=torch.zeros(matrix_size,dtype=torch.long)
        data.peripheral_attr =peripheral_edge_matrix
        return data
    old_edge_index=c(edge_index)

    if "edge_attr" in data and use_edge_feature:
        edge_attr=data.edge_attr#E
    else:
        # skip 0, 1 as it is the mask and self-loop defined in the model
        edge_attr=(torch.ones([edge_index.size(-1)])*2).long() # E


    adj=to_scipy_sparse_matrix(edge_index,num_nodes=num_nodes)

    edge_attr_adj=torch.from_numpy(to_scipy_sparse_matrix(edge_index,edge_attr,num_nodes).toarray()).long()
    #compute each order of adj
    adj_list=adj_K_order(adj,K)

    #create K-hop edge with graph diffusion kernel
    final_adj=0
    for adj_ in adj_list:
        final_adj+=adj_
    final_adj[final_adj>1]=1
    g=nx.from_numpy_matrix(final_adj.numpy(),create_using=nx.DiGraph)
    edge_list = g.edges
    edge_index = torch.from_numpy(np.array(edge_list).T).long()

    #generate K-hop edge attr
    hop1_edge_attr = edge_attr_adj[edge_index[0, :], edge_index[1, :]]
    edge_attr_list=[hop1_edge_attr.unsqueeze(-1)]
    for i in range(1,len(adj_list)):
        adj_=c(adj_list[i])
        #skip 1 as it is the self-loop defined in the model
        adj_[adj_>max_edge_attr_num]=max_edge_attr_num
        adj_[adj_>0]=adj_[adj_>0]+1
        adj_=adj_.long()
        hopk_edge_attr = adj_[edge_index[0, :], edge_index[1, :]].unsqueeze(-1)
        edge_attr_list.append(hopk_edge_attr)
    edge_attr=torch.cat(edge_attr_list,dim=-1) #E * K

    #revise bond feature for molecule dataset
    if "bond_feature" in data:
        edge_attr_adj=to_dense_edge_feature(data.bond_feature,old_edge_index,num_nodes)
        bond_feature=edge_attr_adj[edge_index[0, :], edge_index[1, :]]
        data.bond_feature=bond_feature


    peripheral_attr=get_peripheral_attr(adj_list,edge_attr_adj,max_peripheral_edge_num,max_component_num)

    #update all the attributes

    data.edge_index=edge_index
    data.edge_attr=edge_attr
    data.peripheral_attr=peripheral_attr
    return data

def multi_hop_neighbors_with_spd_kernel(data,K,max_edge_attr_num,max_peripheral_edge_num,max_component_num=5,use_edge_feature=False):
    """generate multi-hop neighbors for input PyG graph using shortest path distance kernel
    Args:
        data(torch_geometric.data.Data): PyG graph data instance
        K(int): number of hop
        max_edge_attr_num(int): maximum number of encoding used for hopk edge
        max_peripheral_edge_num(int): maximum number of peripheral edge to stroe
        use_edge_feature(bool): whether to use edge feature
    """
    assert(isinstance(data, Data))
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes
    if edge_index.size(1)==0:
        if "bond_feature" in data:
            feature_size=data.bond_feature.size(-1)
        elif "edge_attr" in data and use_edge_feature:
            feature_size=data.edge_attr.size(-1)
        else:
            feature_size=1
        matrix_size=list(itertools.chain.from_iterable([[num_nodes,max_component_num,max_peripheral_edge_num],[feature_size]]))
        peripheral_edge_matrix=torch.zeros(matrix_size,dtype=torch.long)
        data.peripheral_attr =peripheral_edge_matrix
        return data
    old_edge_index=c(edge_index)
    if "edge_attr" in data and use_edge_feature:
        edge_attr=data.edge_attr
    else:
        # skip 0, 1 as it is the mask and self-loop defined in the model
        edge_attr=(torch.ones([edge_index.size(-1)])*2).long() # E


    adj=to_scipy_sparse_matrix(edge_index,num_nodes=num_nodes)

    edge_attr_adj=torch.from_numpy(to_scipy_sparse_matrix(edge_index,edge_attr,num_nodes).toarray()).long()
    #compute each order of adj
    adj_list=adj_K_order(adj,K)
    #process adj list to generate shortest path distance matrix with path number
    exist_adj=c(adj_list[0])
    for i in range(1,len(adj_list)):
        adj_=c(adj_list[i])
        #mask all the edge that already exist in previous hops
        adj_[exist_adj>0]=0
        exist_adj=exist_adj+adj_
        exist_adj[exist_adj>1]=1
        adj_list[i]=adj_

    #create K-hop edge with sortest path distance kernel
    final_adj=exist_adj
    g=nx.from_numpy_matrix(final_adj.numpy(),create_using=nx.DiGraph)
    edge_list = g.edges
    edge_index = torch.from_numpy(np.array(edge_list).T).long()

    hop1_edge_attr = edge_attr_adj[edge_index[0, :], edge_index[1, :]]
    edge_attr_list=[hop1_edge_attr.unsqueeze(-1)]
    for i in range(1,len(adj_list)):
        adj_=c(adj_list[i])
        #skip 1 as it is the self-loop defined in the model
        adj_[adj_>max_edge_attr_num]=max_edge_attr_num
        adj_[adj_>0]=adj_[adj_>0]+1
        adj_=adj_.long()
        hopk_edge_attr = adj_[edge_index[0, :], edge_index[1, :]].unsqueeze(-1)
        edge_attr_list.append(hopk_edge_attr)
    edge_attr=torch.cat(edge_attr_list,dim=-1) #E * K


    #surround edge
    #revise bond feature for molecule dataset
    if "bond_feature" in data:
        edge_attr_adj=to_dense_edge_feature(data.bond_feature,old_edge_index,num_nodes)
        bond_feature=edge_attr_adj[edge_index[0, :], edge_index[1, :]]
        data.bond_feature=bond_feature

    peripheral_attr=get_peripheral_attr(adj_list,edge_attr_adj,max_peripheral_edge_num,max_component_num)
    # update all the attributes
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    data.peripheral_attr = peripheral_attr

    return data

def adj_K_order(adj,K):
    """compute the K order of adjacency given scipy matrix
    adj(coo_matrix): adjacency matrix
    K(int): number of hop
    """
    adj_list=[c(adj)]
    for i in range(K-1):
        adj_=adj_list[-1]@adj
        adj_list.append(adj_)
    for i,adj_ in enumerate(adj_list):
        adj_=torch.from_numpy(adj_.toarray()).int()
        #prevent the precision overflow
        #adj_[adj_<0]=1e8
        adj_.fill_diagonal_(0)
        adj_list[i]=adj_
    return adj_list

def get_peripheral_attr(adj_list,edge_attr_adj,max_peripheral_edge_num,max_component_num=None):
    """Compute peripheral information for each node in graph
    Args:
        adj_list(list): adjacency matrix list of data for each hop
        edge_attr_adj(torch.tensor):edge feature matrix
        max_peripheral_edge_num(int): maximum number of peripheral edge to store
        max_component_num(int): maximum number of components
    """
    K=len(adj_list)
    num_nodes=edge_attr_adj.size(0)
    # surround edge

    feature_size = list(edge_attr_adj[0, 0].size())
    if len(feature_size) == 0:
        feature_size = [1]
        edge_attr_adj = edge_attr_adj.unsqueeze(-1)
    if max_peripheral_edge_num > 0:
        peripheral_edge_matrix_list = []
        for i in range(K):
            adj_ = c(adj_list[i])
            peripheral_edge_matrix = extract_peripheral_edges(adj_, edge_attr_adj, max_peripheral_edge_num,max_component_num) # N * c * E' * f
            peripheral_edge_matrix_list.append(peripheral_edge_matrix)
        if max_component_num == 0:
            component_dim = 1
        else:
            component_dim = max_component_num
        peripheral_attr = torch.zeros(
            list(itertools.chain.from_iterable([[K, num_nodes,component_dim, max_peripheral_edge_num], feature_size])))
        for i, peripheral_edge_matrix in enumerate(peripheral_edge_matrix_list):
            peripheral_attr[i] = peripheral_edge_matrix
        peripheral_attr = peripheral_attr.transpose(0, 1)  # N * K * c * E' * f
        peripheral_attr = peripheral_attr.long().squeeze(-1) # if there is only one feature, squeeze feature dimension
    else:
        peripheral_attr = None

    return peripheral_attr


def extract_peripheral_edges(adj,edge_attr_adj,max_edge_num,max_component_num):
    """extract peripheral edge information for each node using input adj and save edge attr information given edge attr adj
    Args:
        adj(torch.tensor): adjacency matrix
        edge_attr_adj(torch.tensor) edge attr adjacency matrix
        max_edge_num(int): maximum number of edge to keep
    """
    num_nodes=edge_attr_adj.size(0)
    feature_size=edge_attr_adj[0,0].size() #f
    if len(feature_size)==0:
        feature_size=[1]
        edge_attr_adj=edge_attr_adj.unsqueeze(-1)
    peripheral_edge_list=[]
    direct_edge_adj=(torch.sum(edge_attr_adj,dim=-1)>0).int() # N * N
    if max_component_num==0:
        component_dim=1
    else:
        component_dim=max_component_num
    matrix_size=list(itertools.chain.from_iterable([[num_nodes,component_dim,max_edge_num],feature_size]))
    peripheral_edge_matrix=torch.zeros(matrix_size,dtype=torch.long)
    for i in range(num_nodes):
        row=torch.where(adj[i]>0)[0]
        # if node have no neighbors, skip
        if len(row)<=1:
            continue

        peripheral_index=torch.combinations(row)
        peripheral_edge_mask=direct_edge_adj[peripheral_index[:,0],peripheral_index[:,1]] # E'
        peripheral_edge_list=peripheral_index[peripheral_edge_mask>0] # E'
        if peripheral_edge_list.size(0)==0:
            continue

        g=nx.from_edgelist(peripheral_edge_list.numpy())
        if max_component_num>0:
            S = [g.subgraph(c).copy() for c in nx.connected_components(g)]
            for j,s in enumerate(S):
                if j>=max_component_num:
                    break
                s_edge_index=torch.from_numpy(np.array(s.edges).T).long()
                s_edge_attr=edge_attr_adj[s_edge_index[0,:],s_edge_index[1,:]]
                edge_num=s_edge_attr.size(0)
                if edge_num>max_edge_num:
                    edge_num=max_edge_num
                peripheral_edge_matrix[i,j,:edge_num]=s_edge_attr[:edge_num]
        else:
            s=g
            s_edge_index = torch.from_numpy(np.array(s.edges).T).long()
            s_edge_attr = edge_attr_adj[s_edge_index[0, :], s_edge_index[1, :]]
            edge_num = s_edge_attr.size(0)
            if edge_num > max_edge_num:
                edge_num = max_edge_num
            peripheral_edge_matrix[i, 0, :edge_num] = s_edge_attr[:edge_num]

    return peripheral_edge_matrix



def to_dense_edge_feature(edge_feature,edge_index,num_nodes):
    """
    convert edge feature to dense adj
    """
    edge_feature=edge_feature.squeeze()
    K=list(edge_feature.size()[1:])
    adj=torch.zeros(list(itertools.chain.from_iterable([[num_nodes],[num_nodes],K])),dtype=edge_feature.dtype)
    for i in range(edge_index.size(-1)):
        v=edge_index[0,i]
        u=edge_index[1,i]
        adj[v,u]=edge_feature[i]

    return adj


def PyG_collate(examples):
    """PyG collcate function
    Args:
        examples(list): batch of samples
    """
    data=Batch.from_data_list(examples)
    return data


def PyG_collate_new(examples):
    data=Batch.from_data_list(examples)
    num_data_nodes=data.num_data_nodes
    node_to_batch=list(itertools.chain.from_iterable([[i for _ in range(num_node.item())] for i,num_node in enumerate(num_data_nodes)]))
    data.batch=torch.tensor(node_to_batch)
    return data


def resistance_distance(data):
    """resistance distance.See "Link prediction in complex networks: A survey".
    Adapted from NestedGNN:https://github.com/muhanzhang/NestedGNN
    Args:
        data(PyG.Data):pyg data object
    """

    edge_index=data.edge_index
    num_nodes=data.num_nodes
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

