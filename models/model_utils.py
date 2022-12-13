"""
Model utils file
"""

from models.GNNs import *


def make_GNN(args):
    if args.model_name == "KPGINPlus":
        return GNNPlus
    elif args.model_name == "KPGINPrime":
        return GNNPrime
    else:
        return GNN
