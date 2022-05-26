"""
Model utils file
"""

from models.GNNs import *
from layers.KPGCN import *
from layers.KPGIN import *
from layers.KPGraphSAGE import *
from layers.KPGINplus import *


def make_gnn_layer(args):

    """function to construct gnn layer
    Args:
        args(argparser): arguments list
    """
    model_name=args.model_name
    if model_name=="KPGCN":
        gnn_layer=KPGCNConv(args.hidden_size,args.hidden_size,args.K,args.num_hop1_edge,args.num_hopk_edge,args.combine)
    elif model_name=="KPGIN":
        gnn_layer=KPGINConv(args.hidden_size,args.hidden_size,args.K,args.eps,args.train_eps,args.num_hop1_edge,args.num_hopk_edge,args.combine)
    elif model_name=="KPGraphSAGE":
        gnn_layer=KPGraphSAGEConv(args.hidden_size,args.hidden_size,args.K,args.aggr,args.num_hop1_edge,args.num_hopk_edge,args.combine)
    elif model_name=="KPGINPlus":
        gnn_layer=[KPGINPlusConv(args.hidden_size,args.hidden_size,l,args.eps,args.train_eps,args.num_hop1_edge,args.num_hopk_edge)
                   if l<=args.K else KPGINPlusConv(args.hidden_size,args.hidden_size,args.K,args.eps,args.train_eps,args.num_hop1_edge,args.num_hopk_edge)
                   for l in range(1,args.num_layer+1)]
    else:
        raise ValueError("Not supported GNN type")



    return gnn_layer


def make_OGBMol_gnn_layer(args):

    """function to construct gnn layer for OGBMol
    Args:
        args(argparser): arguments list
    """
    model_name=args.model_name
    if model_name=="KPGIN":
        gnn_layer=KPGINConvOGBMol(args.hidden_size,args.hidden_size,args.K,args.eps,args.train_eps,args.num_hopk_edge,args.combine)
    elif model_name=="KPGINPlus":
        gnn_layer=[KPGINPlusConvOGBMol(args.hidden_size,args.hidden_size,l,args.eps,args.train_eps,args.num_hopk_edge)
         if l <= args.K else KPGINPlusConvOGBMol(args.hidden_size,args.hidden_size,args.K,args.eps,args.train_eps,args.num_hopk_edge)
         for l in range(1, args.num_layer + 1)]

    else:
        raise ValueError("Not supported GNN type")

    return gnn_layer


def make_QM9_gnn_layer(args):

    """function to construct gnn layer for QM9
    Args:
        args(argparser): arguments list
    """
    model_name=args.model_name
    if model_name=="KPGIN":
        gnn_layer=KPGINConvQM9(args.hidden_size,args.hidden_size,args.K,args.eps,args.train_eps,args.num_hopk_edge,args.combine)
    elif model_name=="KSGNNPlus":
        gnn_layer=[KPGINPlusConvQM9(args.hidden_size,args.hidden_size,l,args.eps,args.train_eps,args.num_hopk_edge)
         if l <= args.K else KPGINPlusConvOGBMol(args.hidden_size,args.hidden_size,args.K,args.eps,args.train_eps,args.num_hopk_edge)
         for l in range(1, args.num_layer + 1)]

    else:
        raise ValueError("Not supported GNN type")

    return gnn_layer

def make_GNN(args):
    if args.dataset_name.startswith('ogbg-mol'):
        if args.model_name=="KPGINPlus":
            return KPGINPlusOGBMol
        else:
            return GNNOGBMol
    elif args.dataset_name=="QM9":
        if args.model_name=="KPGINPlus":
            return KPGINPlusQM9
        else:
            return GNNQM9
    else:
        if args.model_name == "KPGINPlus":
            return KPGINPlus
        else:
            return GNN
