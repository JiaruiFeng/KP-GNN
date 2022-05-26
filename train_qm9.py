"""
script to train on QM9 task
"""

import torch
import torch.nn.functional as F
import numpy as np
import train_utils
import random
import argparse
from data_utils import multi_hop_neighbors_with_gd_kernel,multi_hop_neighbors_with_spd_kernel,PyG_collate,resistance_distance
from models.model_utils import make_QM9_gnn_layer,make_GNN
from models.GraphRegression import GraphRegression
from torch_geometric.nn import DataParallel
from json import dumps
from datasets.QM9Dataset import QM9,conversion
import torch_geometric.transforms as T
import shutil
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau



def get_model(args):
    layer=make_QM9_gnn_layer(args)
    GNNModel=make_GNN(args)
    gnn=GNNModel(
                  num_layer=args.num_layer,
                  gnn_layer=layer,
                  input_size=args.input_size,
                  JK=args.JK,
                  norm_type=args.norm_type,
                  virtual_node=args.virtual_node,
                  residual=args.residual,
                  use_rd=args.use_rd,
                  use_pos=args.use_pos,
                  drop_prob=args.drop_prob)

    model=GraphRegression(embedding_model=gnn,
                          pooling_method=args.pooling_method)

    model.reset_parameters()
    if args.parallel:
        model = DataParallel(model, args.gpu_ids)


    return model

class MyTransform(object):
    def __init__(self, target,pre_convert=False):
        self.target=target
        self.pre_convert = pre_convert

    def __call__(self, data):
        data.y = data.y[:, int(self.target)]  # Specify target: 0 = mu
        if self.pre_convert:  # convert back to original units
            data.y = data.y / conversion[int(self.target)]
        return data

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes > 6  # Remove graphs with less than 6 nodes.

def bond_feature_reset(g):
    bond_feature=g.edge_attr
    #0 for mask, 1 for self-loop
    bond_feature=bond_feature+2
    g.bond_feature=bond_feature
    g.edge_attr=None
    return g



def train(model,train_loader,optimizer,device):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        num_graphs = data.num_graphs
        optimizer.zero_grad()
        y = data.y

        loss = F.mse_loss(model(data), y)

        loss.backward()
        loss_all += loss * num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(model,loader,task,std,device):
    model.eval()
    error = 0

    for data in loader:
        if type(data) == dict:
            data = {key: data_.to(device) for key, data_ in data.items()}
        else:
            data = data.to(device)
        y = data.y
        error += ((model(data) * std[task].cuda()) -
                  (y * std[task].cuda())).abs().sum().item()  # MAE
    return error / len(loader.dataset)




def main():
    parser = argparse.ArgumentParser(f'arguments for training and testing')
    parser.add_argument('--save_dir',type=str,default='./save',help='Base directory for saving information.')
    parser.add_argument('--seed',type=int,default=234,help='Random seed for reproducibility.')
    parser.add_argument('--dataset_name',type=str,default="QM9",help='name of dataset')
    parser.add_argument('--task',type=int,default=0,choices=list(range(19)),help='number of task')
    parser.add_argument('--drop_prob',type=float,default=0.0,help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--batch_size',type=int,default=128,help='Batch size per GPU. Scales automatically when \
                            multiple GPUs are available.')
    parser.add_argument('--num_workers',type=int,default=0,help='number of worker.')
    parser.add_argument('--load_path',type=str,default=None,help='Path to load as a model checkpoint.')
    parser.add_argument('--lr',type=float,default=0.001,help='Learning rate.')
    parser.add_argument('--min_lr',type=float,default=1e-6,help='Minimum learning rate.')
    parser.add_argument('--l2_wd',type=float,default=3e-9,help='L2 weight decay.')
    parser.add_argument("--kernel",type=str,default="spd",choices=("gd","spd"),help="the kernel used for K-hop computation")
    parser.add_argument('--num_epochs',type=int,default=200,help='Number of epochs.')
    parser.add_argument('--max_grad_norm',type=float,default=5.0,help='Maximum gradient norm for gradient clipping.')
    parser.add_argument("--hidden_size",type=int,default=128,help="hidden size of the model")
    parser.add_argument("--model_name",type=str,default="KPGINPlus",choices=("KPGIN","KPGINPlus"),help="Base GNN model")
    parser.add_argument("--K",type=int,default=8,help="number of hop to consider")
    parser.add_argument("--max_edge_attr_num",type=int,default=10,help="max length in edge attr of hop larger than 1. "
                                                                      "Must be equal to or greater than 1")
    parser.add_argument("--max_peripheral_edge_num",type=int,default=6,help="max number of edge to keep for surround edge,"
                                                                           " 0 means no peripheral edge information")
    parser.add_argument("--max_component_num",type=int,default=3,help="maximum component number in peripheral subgraph information")
    parser.add_argument("--use_edge_feature",type=bool,default=False,help="Whether to use edge feature")
    parser.add_argument("--num_hop1_edge",type=int,default=3,help="number of edge type in hop 1")
    parser.add_argument("--num_layer",type=int,default=8,help="number of layer for feature encoder")
    parser.add_argument("--JK",type=str,default="concat",choices=("sum","max","mean","attention","last","concat"),help="Jumping knowledge method")
    parser.add_argument("--residual",type=bool,default=False,help="Whether to use residual connection between each layer")
    parser.add_argument("--eps",type=float,default=0.,help="Initital epsilon in GIN")
    parser.add_argument("--train_eps",type=bool,default=True,help="Whether the epsilon is trainable")
    parser.add_argument("--negative_slope",type=float,default=0.2,help="slope in LeakyRelu")
    parser.add_argument("--combine",type=str,default="attention",choices=("attention","geometric"),help="Jumping knowledge method")
    parser.add_argument("--pooling_method",type=str,default="attention",choices=("mean","sum","attention"),help="pooling method in graph classification")
    parser.add_argument('--norm_type',type=str,default="Batch",choices=("Batch","Layer","Instance","GraphSize","Pair"),
                        help="normalization method in model")
    parser.add_argument('--aggr',type=str,default="add",help='aggregation method in GNN layer, only works in GraphSAGE')
    parser.add_argument('--factor',type=float,default=0.7,help='factor in the ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--patience',type=int,default=5,help='patience in the ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--reprocess',type=bool,default=False,help='Whether to reprocess the dataset')
    parser.add_argument('--normalize_x', action='store_true', default=False,
                        help='if True, normalize non-binary node features')
    parser.add_argument('--virtual_node',type=bool, default=True,help="virtual node")
    parser.add_argument('--use_pos', action='store_true', default=False,
                        help='use node position (3D) as continuous node features')
    parser.add_argument('--use_rd', action='store_true', default=True,
                        help='use resistance distance as additional node labels')
    parser.add_argument('--filter', action='store_true', default=False,
                        help='whether to filter graphs with less than 7 nodes')
    parser.add_argument('--convert', type=str, default='post',
                        help='if "post", convert units after optimization; if "pre", \
                        convert units before optimization')


    args=parser.parse_args()
    args.num_hopk_edge=args.max_edge_attr_num+2
    if args.max_peripheral_edge_num>0:
        args.name=args.model_name+"_"+str(args.K)+"_"+"ps"+"_"+args.kernel+"_"+str(args.num_layer)+"_"+str(args.hidden_size)+"_"+str(args.max_edge_attr_num)
    else:
        args.name=args.model_name+"_"+str(args.K)+"_"+"nops"+"_"+args.kernel+"_"+str(args.num_layer)+"_"+str(args.hidden_size)+"_"+str(args.max_edge_attr_num)
    # Set up logging and devices
    args.save_dir = train_utils.get_save_dir(args.save_dir, args.name, type=args.dataset_name)
    log = train_utils.get_logger(args.save_dir, args.name)
    device, args.gpu_ids = train_utils.get_available_devices()
    if len(args.gpu_ids)>1:
        args.parallel=True
    else:
        args.parallel=False
    args.batch_size *= max(1, len(args.gpu_ids))
    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    if args.kernel=="gd":
        def pre_transform(g):
            return multi_hop_neighbors_with_gd_kernel(g,args.K,args.max_edge_attr_num,args.max_peripheral_edge_num,
                                                      args.max_component_num,args.use_edge_feature)
    elif args.kernel=="spd":
        def pre_transform(g):
            return multi_hop_neighbors_with_spd_kernel(g,args.K,args.max_edge_attr_num,args.max_peripheral_edge_num,
                                                       args.max_component_num,args.use_edge_feature)
    else:
        def pre_transform(g):
            return g

    path="data/qm9_"
    path=path+str(args.K)+"_"+str(args.max_peripheral_edge_num)+"_"+str(args.max_component_num)+"_"+str(args.max_edge_attr_num)+"_"+args.kernel
    if os.path.exists(path+'/processed') and args.reprocess:
        shutil.rmtree( path+'/processed')

    pre_filter = None
    if args.filter:
        pre_filter = MyFilter()
        path += '_filtered'


    if args.use_rd:
        rd_feature=resistance_distance
    else:
        def rd_feature(g):
            return g

    dataset = QM9(path,pre_transform=T.Compose([bond_feature_reset,pre_transform,rd_feature]),transform=T.Compose([MyTransform(args.task,args.convert=='pre')]),pre_filter=pre_filter)
    dataset = dataset.shuffle()

    # Normalize targets to mean = 0 and std = 1.
    tenpercent = int(len(dataset) * 0.1)
    mean = dataset.data.y[tenpercent:].mean(dim=0)
    std = dataset.data.y[tenpercent:].std(dim=0)
    dataset.data.y = (dataset.data.y - mean) / std

    train_dataset = dataset[2 * tenpercent:]

    cont_feat_start_dim = 5
    if args.normalize_x:
        x_mean = train_dataset.data.x[:, cont_feat_start_dim:].mean(dim=0)
        x_std = train_dataset.data.x[:, cont_feat_start_dim:].std(dim=0)
        x_norm = (train_dataset.data.x[:, cont_feat_start_dim:] - x_mean) / x_std
        dataset.data.x = torch.cat([dataset.data.x[:, :cont_feat_start_dim], x_norm], 1)

    test_dataset = dataset[:tenpercent]
    val_dataset = dataset[tenpercent:2 * tenpercent]
    train_dataset = dataset[2 * tenpercent:]

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,collate_fn=PyG_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,collate_fn=PyG_collate)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=PyG_collate)

    input_size=19
    if args.use_pos:
        input_size+=3
    args.input_size=input_size
    args.output_size=1

    #output argument to log file
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    model = get_model(args)
    model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    log.info(f'The total parameters of model :{[pytorch_total_params]}')

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",factor=args.factor, patience=args.patience)

    best_val_error=1E6
    test_error=1E6
    pbar = tqdm(range(1, args.num_epochs+1))
    for epoch in pbar:
        pbar.set_description('Epoch: {:03d}'.format(epoch))
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(model,train_loader,optimizer,device)
        val_error = test(model,val_loader,args.task,std,device)
        scheduler.step(val_error)

        if val_error <= best_val_error:
            test_error = test(model,test_loader,args.task,std,device)
            best_val_error = val_error
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f'best_model.pth'))

        info = (
            'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, ' +
            'Test MAE: {:.7f}, Test MAE norm: {:.7f}, Test MAE convert: {:.7f}'
        ).format(
             epoch, lr, loss, val_error,
             test_error,
             test_error / std[args.task].cuda(),
             test_error / conversion[int(args.task)].cuda() if args.convert == 'post' else 0
        )

        log.info(info)

        if optimizer.param_groups[0]['lr'] < args.min_lr:
            log.info("\n!! LR EQUAL TO MIN LR SET.")
            break


if __name__=="__main__":
    main()
