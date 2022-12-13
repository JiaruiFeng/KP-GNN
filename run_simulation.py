"""
script to run simulation experiment on regular graph
Adapted from Nested GNN:https://github.com/muhanzhang/NestedGNN
"""
import argparse
import logging
import math
import os
import time
from copy import deepcopy as c
from json import dumps

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import from_networkx

import train_utils
from data_utils import extract_multi_hop_neighbors
from layers.combine import *

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True


class KGINConv(MessagePassing):
    """
    K hop-GNN with GIN kernel
    Args:
        hidden_size(int)
    """

    def __init__(self, hidden_size, K, eps=0., train_eps=False):
        super(KGINConv, self).__init__(node_dim=0)
        self.aggr = "add"
        self.K = K
        self.hidden_size = hidden_size
        # multi-layer perceptron
        self.proj = nn.Linear(1, K * hidden_size)
        self.hop_proj1 = torch.nn.Parameter(torch.Tensor(self.K, hidden_size, hidden_size))
        self.hop_bias1 = torch.nn.Parameter(torch.Tensor(self.K, hidden_size))
        self.hop_proj2 = torch.nn.Parameter(torch.Tensor(self.K, hidden_size, hidden_size))
        self.hop_bias2 = torch.nn.Parameter(torch.Tensor(self.K, hidden_size))

        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.combine_proj = nn.Linear(hidden_size * K, hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.hop_proj1)
        nn.init.kaiming_uniform_(self.hop_proj2)
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.hop_proj1)
        bound1 = 1 / math.sqrt(fan_in1) if fan_in1 > 0 else 0
        nn.init.uniform_(self.hop_bias1, -bound1, bound1)
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.hop_proj2)
        bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
        nn.init.uniform_(self.hop_bias2, -bound2, bound2)
        if isinstance(self.combine_proj, nn.Linear):
            self.combine_proj.reset_parameters()
        nn.init.zeros_(self.eps)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.proj(x)
        x = x.view(-1, self.K, self.hidden_size)  # N * K * h
        x_n = self.propagate(edge_index, x=x, mask=edge_attr)  # N * K * dk

        x = x_n + (1 + self.eps) * x
        x = x.permute(1, 0, 2)
        x = F.relu(torch.matmul(x, self.hop_proj1) + self.hop_bias1.unsqueeze(1))
        x = F.relu(torch.matmul(x, self.hop_proj2) + self.hop_bias2.unsqueeze(1))
        x = x.permute(1, 0, 2).contiguous()
        x = x.view(-1, self.K * self.hidden_size)
        # combine
        x = self.combine_proj(x)
        if args.graph:
            x = global_add_pool(x, batch)
        return x

    def message(self, x_j, mask):
        # x_j=x_j+edge_emb # E * K * dk
        mask = mask.unsqueeze(-1)  # E * K * 1
        return x_j.masked_fill_(mask == 0, 0.)

    def update(self, aggr_out):
        return aggr_out


def simulate(args, device):
    results = {}
    for n in args.n:
        print('n = {}'.format(n))
        graphs = generate_many_k_regular_graphs(args.R, n, args.N)
        for k in range(1, args.K + 1):
            G = c(graphs)
            G = [extract_multi_hop_neighbors(g, k, 10, 1, 1, 1, 1, "spd") for g in G]
            # print(G[0].edge_attr)
            loader = DataLoader(G, batch_size=1)
            hidden_size = 16
            model = KGINConv(hidden_size, k)
            model.to(device)
            output = run_simulation(model, loader, device)  # output shape [G.number_of_nodes(), feat_dim]
            output = output.cpu()
            collision_rate = compute_simulation_collisions(output, ratio=True)
            results[(n, k)] = collision_rate
            torch.cuda.empty_cache()
            print('k = {}: {}'.format(k, collision_rate))
        print('#' * 30)
    return results


def generate_many_k_regular_graphs(k, n, N, seed=0):
    graphs = [generate_k_regular(k, n, s) for s in range(seed, seed + N)]
    graphs = [from_networkx(g) for g in graphs]
    for g in graphs:
        g.x = torch.ones([g.num_nodes, 1])
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
            output.append(model(data.x, data.edge_index, data.edge_attr, data.batch))
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
    lbound = plt.plot([n_min, n_max],
                      [np.log(2 * n_min) / np.log(2) / 2, np.log(2 * n_max) / np.log(2) / 2],
                      'r--', label='0.5 log(2n) / log(r-1)')
    # ubound = plt.plot([n_min, n_max],
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
                    help='Node degree (r) or synthetic r-regular graph')
parser.add_argument('--n', nargs='*',
                    help='A list of number of nodes in each connected k-regular subgraph')
parser.add_argument('--N', type=int, default=100,
                    help='Number of graphs in simultation')
parser.add_argument('--K', type=int, default=6,
                    help='Largest number of hop to consider')
parser.add_argument('--graph', action='store_true', default=False,
                    help='If True, compute whole-graph collision rate; otherwise node')
parser.add_argument('--save_appendix', default='',
                    help='What to append to save-names when saving results')
args = parser.parse_args()
args.n = [int(n) for n in args.n]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
else:
    args.save_appendix = args.save_appendix + '_' + time.strftime("%Y%m%d%H%M%S")
args.res_dir = 'save/simulation{}'.format(args.save_appendix)

if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)
log = train_utils.get_logger(args.res_dir, '')
log.info('Results will be saved in ' + args.res_dir)
# output argument to log file
log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

# Plot visualization figure
results = simulate(args, device)
save_simulation_result(results, args.res_dir)
