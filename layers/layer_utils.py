"""
Utils for defining model layers
"""
import torch
from layers.KPGCN import *
from layers.KPGIN import *
from layers.KPGINplus import *
from layers.KPGraphSAGE import *

def make_gnn_layer(args):

    """function to construct gnn layer
    Args:
        args(argparser): arguments list
    """
    model_name=args.model_name
    if model_name=="KPGCN":
        gnn_layer=KPGCNConv(args.hidden_size,args.hidden_size,args.K,args.num_hop1_edge,args.max_pe_num,args.combine)
    elif model_name=="KPGIN":
        gnn_layer=KPGINConv(args.hidden_size,args.hidden_size,args.K,args.eps,args.train_eps,args.num_hop1_edge,args.max_pe_num,args.combine)
    elif model_name=="KPGraphSAGE":
        gnn_layer=KPGraphSAGEConv(args.hidden_size,args.hidden_size,args.K,args.aggr,args.num_hop1_edge,args.max_pe_num,args.combine)
    elif model_name=="KPGINPlus":
        gnn_layer=[KPGINPlusConv(args.hidden_size,args.hidden_size,l,args.num_hop1_edge,args.max_pe_num,args.combine)
                   if l<=args.K else KPGINPlusConv(args.hidden_size,args.hidden_size,args.K,args.num_hop1_edge,args.max_pe_num,args.combine)
                   for l in range(1,args.num_layer+1)]
    else:
        raise ValueError("Not supported GNN type")

    return gnn_layer

