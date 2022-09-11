"""
script to run simulation experiment on regular graph
Adapted from Nested GNN:https://github.com/muhanzhang/NestedGNN
"""
import argparse
import time, os, sys
from shutil import copy
import matplotlib.pyplot as plt
import logging
from math import ceil
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from layers.KPGIN import KPGINConv
from data_utils import multi_hop_neighbors_with_spd_kernel,multi_hop_neighbors_with_gd_kernel
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
import train_utils
from json import dumps
import torch.nn as nn
from layers.combine import *
from torch_geometric.nn import MessagePassing,global_add_pool
from copy import deepcopy as c


logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True
import pdb


class KPGINConv(MessagePassing):
    """
    KP-GNN with GIN kernel
    Args:
        input_size(int): the size of input feature
        output_size(int): the size of output feature
        K(int): number of hop to consider in Convolution layer
        eps(float): initial epsilon
        train_eps(bool):whether the epsilon is trainable
        num_hop1_edge(int): number of edge type at 1 hop, need to be equal or larger than 3. default is 3.
                            Where index 0 represent mask(no edge), index 1 represent self-loop, index 2 represent edge.
                            larger than 2 means edge features if have.
        num_hopk_edge(int): number of edge type higher than 1 hop, need to be equal or larger than 3. default is 3.
                    Where index 0 represent mask(no edge), index 1 represent self-loop, index 2 represent edge.
                    larger than 2 means edge features if have.
        combine(str): combination method for information in different hop. select from(geometric, attention)
    """
    def __init__(self,input_size,output_size,K,eps=0.,train_eps=False,num_hop1_edge=3,num_hopk_edge=3,combine="geometric"):
        super(KPGINConv, self).__init__(node_dim=0)
        self.aggr="add"
        self.K=K
        self.output_size=output_size
        self.input_dk=output_size
        self.output_dk=output_size
        # multi-layer perceptron
        self.initial_proj=nn.Linear(input_size,self.output_dk*self.K)
        self.hop_proj1=torch.nn.Parameter(torch.Tensor(self.K,self.input_dk,self.output_dk))
        self.hop_bias1=torch.nn.Parameter(torch.Tensor(self.K,self.output_dk))
        self.hop_proj2=torch.nn.Parameter(torch.Tensor(self.K,self.output_dk,self.output_dk))
        self.hop_bias2=torch.nn.Parameter(torch.Tensor(self.K,self.output_dk))

        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))


        # edge embedding for 1-hop and k-hop
        # Notice that in hops larger than one, there is no actually edge feature, therefore need addtional embedding layer to encode
        # self defined features like path encoding

        self.hop1_edge_emb = torch.nn.Embedding(num_hop1_edge, self.input_dk,padding_idx=0)

        #If K larger than 1, define additional embedding and combine function
        if self.K>1:
            self.hopk_edge_emb = torch.nn.Embedding(num_hopk_edge, self.input_dk,padding_idx=0)
        else:
            self.hopk_edge_emb=None
        self.combine_proj=nn.Linear(self.output_dk*K, self.output_size)
        self.reset_parameters()

    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def reset_parameters(self):
        self.hop1_edge_emb.reset_parameters()
        nn.init.xavier_uniform_(self.hop_proj1.data)
        nn.init.xavier_uniform_(self.hop_proj2.data)
        nn.init.zeros_(self.hop_bias1.data)
        nn.init.zeros_(self.hop_bias2.data)
        if self.hopk_edge_emb is not None:
            nn.init.xavier_uniform_(self.hopk_edge_emb.weight.data)
        self.combine_proj.apply(self.weights_init)
        self.initial_proj.apply(self.weights_init)
        nn.init.zeros_(self.eps)

    def forward(self,data):
        x=data.x
        edge_index=data.edge_index
        edge_attr=data.edge_attr
        peripheral_attr=data.peripheral_attr
        if x is None:
            x = torch.ones([data.num_nodes, 1]).to(edge_index.device)
        x=self.initial_proj(x)

        x=x.view(-1,self.K,self.input_dk) # N * K * dk

        #embedding of edge
        e1_emb = self.hop1_edge_emb(edge_attr[:,:1]) # E * 1 * dk
        if self.K>1:
            ek_emb = self.hopk_edge_emb(edge_attr[:,1:]) # E * K-1 * dk
            e_emb = torch.cat([e1_emb,ek_emb],dim=-2) # E * K * dk
        else:
            e_emb=e1_emb

        x_n=self.propagate(edge_index, x=x, edge_attr=e_emb, mask=edge_attr) # N * K * dk

        #add peripheral subgraph information
        if peripheral_attr is not None:
            se_emb = self.hop1_edge_emb(peripheral_attr) # N * K * c * E' * dk
            se_emb.masked_fill_(peripheral_attr.unsqueeze(-1)==0,0.)
            se_emb=torch.sum(se_emb,dim=-2) # N * K * c * dk
            total=torch.sum((peripheral_attr>0).int(),dim=-1).unsqueeze(-1) # N * K * c * 1
            total[total==0]=1
            se_emb=se_emb/total
            se_emb=torch.sum(se_emb,dim=-2) # N * K * dk
            x_n=x_n+se_emb

        x=x_n + (1 + self.eps) * x
        x=x.permute(1,0,2)
        x=F.relu(torch.matmul(x,self.hop_proj1)+self.hop_bias1.unsqueeze(1))
        x=F.relu(torch.matmul(x,self.hop_proj2)+self.hop_bias2.unsqueeze(1))
        x=x.permute(1,0,2).contiguous()
        #combine
        x=self.combine_proj(x.view(-1,self.K*self.output_dk))

        if args.graph:
            x=global_add_pool(x, data.batch)
        return x


    def message(self, x_j,edge_attr,mask):
        x_j=x_j+edge_attr # E * K * dk
        mask=mask.unsqueeze(-1) # E * K * 1
        return x_j.masked_fill_(mask==0, 0.)


    def update(self,aggr_out):
        return aggr_out



def simulate(args, device):
    results = {}
    for n in args.n:
        print('n = {}'.format(n))
        graphs = generate_many_k_regular_graphs(args.R, n, args.N)
        for k in range(1, args.K + 1):
            G=c(graphs)
            G = [multi_hop_neighbors_with_spd_kernel(g,k,1,0,0) for g in G]
            #print(G[0].edge_attr)
            loader = DataLoader(G, batch_size=1)
            hidden_size=32
            model=KPGINConv(args.input_size,hidden_size,k)
            model.to(device)
            output = run_simulation(model, loader, device)  # output shape [G.number_of_nodes(), feat_dim]
            collision_rate = compute_simulation_collisions(output, ratio=True)
            results[(n, k)] = collision_rate
            torch.cuda.empty_cache()
            print('k = {}: {}'.format(k, collision_rate))
        print('#' * 30)
    return results


def generate_many_k_regular_graphs(k, n, N, seed=0):
    graphs = [generate_k_regular(k, n, s) for s in range(seed, seed + N)]
    graphs = [from_networkx(g) for g in graphs]
    return graphs


def generate_k_regular(k, n, seed=0):
    G = nx.random_regular_graph(d=k, n=n, seed=seed)
    return G


def run_simulation(model, loader, device):
    model.eval()
    with torch.no_grad():
        output = []
        for data in loader:
            data = data.to(device)
            output.append(model(data))
        output = torch.cat(output, 0)
    return output


def save_simulation_result(results, res_dir, pic_format='pdf'):
    n_l, k_l, r_l = [], [], []
    for (n, h), r in results.items():
        n_l.append(n)
        k_l.append(h)
        r_l.append(r)
    main = plt.scatter(n_l, k_l, c=r_l, cmap="Greys", edgecolors='k', linewidths=0.2)
    plt.colorbar(main, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    n_min, n_max = min(n_l), max(n_l)
    #lbound = plt.plot([n_min, n_max],
    #                  [np.log(n_min) / np.log(2) / 2, np.log(n_max) / np.log(2) / 2],
    #                  'r--', label='0.5 log(n) / log(r-1)')
    #ubound = plt.plot([n_min, n_max],
    #                  [np.log(n_min) / np.log(2), np.log(n_max) / np.log(2)],
    #                  'b--', label='log(n) / log(r-1)')
    plt.xscale('log')
    plt.xlabel('number of nodes (n)')
    plt.ylabel('maximum number of hop to consider (K)')
    plt.legend(loc='upper left')
    plt.savefig('{}/simulation_results.{}'.format(res_dir, pic_format), dpi=300)


def compute_simulation_collisions(outputs, ratio=True):
    epsilon = 1e-10
    N = outputs.size(0)
    with torch.no_grad():
        a = outputs.unsqueeze(-1)
        b = outputs.t().unsqueeze(0)
        diff = a - b
        diff = (diff ** 2).sum(dim=1)
        n_collision = int(((diff < epsilon).sum().item() - N) / 2)
        r = n_collision / (N * (N - 1) / 2)
    if ratio:
        return r
    else:
        return n_collision


parser = argparse.ArgumentParser(description='Node configuration Simulation Experiment')
parser.add_argument('--R', type=int, default=3,
                    help='node degree (r) or synthetic r-regular graph')
parser.add_argument('--n', nargs='*',
                    help='a list of number of nodes in each connected k-regular subgraph')
parser.add_argument('--N', type=int, default=100,
                    help='number of graphs in simultation')
parser.add_argument('--K', type=int, default=10,
                    help='largest number of hop to consider')
parser.add_argument('--graph', action='store_true', default=False,
                    help='if True, compute whole-graph collision rate; otherwise node')
parser.add_argument('--layers', type=int, default=1, help='# message passing layers')
parser.add_argument('--save_appendix', default='',
                    help='what to append to save-names when saving results')
parser.add_argument('--input_size',type=int,default=1,
                    help="input size of the model")
parser.add_argument('--hidden_size',type=int,default=32,
                    help="hidden size of the model")
args = parser.parse_args()
args.n = [int(n) for n in args.n]


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
else:
    args.save_appendix = args.save_appendix+'_' + time.strftime("%Y%m%d%H%M%S")
args.res_dir = 'save/simulation{}'.format(args.save_appendix)

print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)
log = train_utils.get_logger(args.res_dir, '')
# Backup python files.
copy('run_simulation.py', args.res_dir)

# output argument to log file
log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

# Plot visualization figure
results = simulate(args, device)
save_simulation_result(results, args.res_dir)




