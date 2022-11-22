"""
Model utils file
"""

from models.GNNs import *


def make_GNN(args):
    if args.model_name == "KPGINPlus":
        return GNNPlus
    else:
        return GNN
