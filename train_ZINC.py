"""
script to train on ZINC task
"""
import train_utils
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from json import dumps
from models.model_utils import make_gnn_layer,make_GNN
from models.GraphRegression import GraphRegression
from tqdm import tqdm
import torch.nn.functional as F
from functools import partial
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import DataParallel
import argparse
from data_utils import multi_hop_neighbors_with_gd_kernel,multi_hop_neighbors_with_spd_kernel,PyG_collate,resistance_distance
import torch_geometric.transforms as T
from datasets.ZINC_dataset import ZINC
import os
import shutil

def train(model,device,loader,optimizer,epoch):
    total_loss=0
    with torch.enable_grad(), \
         tqdm(total=len(loader.dataset)) as progress_bar:
        for data in loader:
            y=data.y
            y = y.to(device)
            data = data.to(device)
            batch_size = data.num_graphs
            optimizer.zero_grad()
            # forward
            score = model(data )
            loss = F.l1_loss(score, y)
            mae = MAE(score, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.0)
            optimizer.step()
            model.zero_grad()

            progress_bar.update(batch_size)
            loss_val = loss.item()
            lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(epoch=epoch,
                                     loss=loss_val,
                                     mae=mae,
                                     lr=lr)
            total_loss+=loss.item()*batch_size
    return total_loss/ len(loader.dataset)


def test(model,device,loader):
    total_mae=0
    model.eval()

    with torch.no_grad(), \
         tqdm(total=len(loader.dataset)) as progress_bar:
        for data in loader:
            y=data.y
            y = y.to(device)
            data = data.to(device)
            batch_size = data.num_graphs
            # forward
            score = model(data)
            mae=MAE(score,y)
            mae=mae*batch_size
            total_mae=total_mae+mae

            progress_bar.update(batch_size)
            progress_bar.set_postfix(MAE=mae)

    mae_avg=total_mae/len(loader.dataset)
    model.train()
    results_list = [
        ('Loss', mae_avg),
        ('MAE', mae_avg)
    ]
    results = OrderedDict(results_list)
    return results


def get_model(args):
    layer=make_gnn_layer(args)
    init_emb=nn.Embedding(args.input_size,args.hidden_size)
    GNNModel=make_GNN(args)
    gnn=GNNModel(
                  num_layer=args.num_layer,
                  gnn_layer=layer,
                  JK=args.JK,
                  norm_type=args.norm_type,
                  init_emb=init_emb,
                  residual=args.residual,
                  virtual_node=args.virtual_node,
                  use_rd=args.use_rd,
                  drop_prob=args.drop_prob)


    model=GraphRegression(embedding_model=gnn,
                          pooling_method=args.pooling_method)
    model.reset_parameters()
    #If use multiple gpu, torch geometric model must use DataParallel class
    if args.parallel:
        model = DataParallel(model, args.gpu_ids)



    return model

def convert_edge_labels(data):
    if data.edge_attr is not None:
        data.edge_attr=data.edge_attr+1
    return data

def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE

def main():
    parser = argparse.ArgumentParser(f'arguments for training and testing')
    parser.add_argument('--save_dir',type=str,default='./save',help='Base directory for saving information.')
    parser.add_argument('--seed',type=int,default=234,help='Random seed for reproducibility.')
    parser.add_argument('--dataset_name',type=str,default="ZINC",help='name of dataset')
    parser.add_argument('--drop_prob',type=float,default=0.1,help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--batch_size',type=int,default=32,help='Batch size per GPU. Scales automatically when \
                            multiple GPUs are available.')
    parser.add_argument('--num_workers',type=int,default=0,help='number of worker.')
    parser.add_argument('--load_path',type=str,default=None,help='Path to load as a model checkpoint.')
    parser.add_argument('--lr',type=float,default=0.001,help='Learning rate.')
    parser.add_argument('--min_lr',type=float,default=1e-6,help='Minimum learning rate.')
    parser.add_argument('--l2_wd',type=float,default=3e-7,help='L2 weight decay.')
    parser.add_argument("--kernel",type=str,default="gd",choices=("gd","spd"),help="the kernel used for K-hop computation")
    parser.add_argument('--num_epochs',type=int,default=500,help='Number of epochs.')
    parser.add_argument('--max_grad_norm',type=float,default=5.0,help='Maximum gradient norm for gradient clipping.')
    parser.add_argument("--hidden_size",type=int,default=136,help="hidden size of the model")
    parser.add_argument("--model_name",type=str,default="KPGINPlus",choices=("KPGIN","KPGINPlus"),help="Base GNN model")
    parser.add_argument("--K",type=int,default=4,help="number of hop to consider")
    parser.add_argument("--max_edge_attr_num",type=int,default=1,help="max length in edge attr of hop larger than 1. "
                                                                      "Must be equal to or greater than 1")
    parser.add_argument("--max_peripheral_edge_num",type=int,default=6,help="max number of edge to keep for surround edge,"
                                                                           " 0 means no peripheral edge information")
    parser.add_argument("--max_component_num",type=int,default=3,help="maximum component number in peripheral subgraph information")
    parser.add_argument("--use_edge_feature",type=bool,default=True,help="Whether to use edge feature")
    parser.add_argument("--num_hop1_edge",type=int,default=5,help="number of edge type in hop 1")
    parser.add_argument("--num_layer",type=int,default=4,help="number of layer for feature encoder")
    parser.add_argument("--head",type=int,default=4,help="number of head")
    parser.add_argument("--JK",type=str,default="concat",choices=("sum","max","mean","attention","last","concat"),help="Jumping knowledge method")
    parser.add_argument("--residual",type=bool,default=True,help="Whether to use residual connection between each layer")
    parser.add_argument("--eps",type=float,default=0.,help="Initital epsilon in GIN")
    parser.add_argument("--train_eps",type=bool,default=True,help="Whether the epsilon is trainable")
    parser.add_argument("--negative_slope",type=float,default=0.2,help="slope in LeakyRelu")
    parser.add_argument("--combine",type=str,default="attention",choices=("attention","geometric"),help="Jumping knowledge method")
    parser.add_argument("--pooling_method",type=str,default="attention",choices=("mean","sum","attention"),help="pooling method in graph classification")
    parser.add_argument('--norm_type',type=str,default="Batch",choices=("Batch","Layer","Instance","GraphSize","Pair"),
                        help="normalization method in model")
    parser.add_argument('--aggr',type=str,default="add",help='aggregation method in GNN layer, only works in GraphSAGE')
    parser.add_argument('--factor',type=float,default=0.5,help='factor in the ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--patience',type=float,default=10,help='factor in the ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--reprocess',type=bool,default=False,help='Whether to reprocess the dataset')
    parser.add_argument('--use_rd', action='store_true', default=False,
                        help='use resistance distance as additional node labels')
    parser.add_argument('--virtual_node', type=bool, default=False,help='enable using virtual node, default true')
    args = parser.parse_args()
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


    if args.use_rd:
        rd_feature=resistance_distance
    else:
        def rd_feature(g):
            return g

    path="data/ZINC_"
    path=path+str(args.K)+"_"+str(args.max_peripheral_edge_num)+"_"+str(args.max_component_num)+"_"+str(args.max_edge_attr_num)+"_"+args.kernel

    if os.path.exists(path+"/"+args.dataset_name+ '/processed') and args.reprocess:
        shutil.rmtree( path+"/"+args.dataset_name+ '/processed')


    trainset = ZINC(path,subset=True,split="train",pre_transform=T.Compose([convert_edge_labels,rd_feature,pre_transform]))
    valset = ZINC(path,subset=True,split="val",pre_transform=T.Compose([convert_edge_labels,rd_feature,pre_transform]))
    testset = ZINC(path,subset=True,split="test",pre_transform=T.Compose([convert_edge_labels,rd_feature,pre_transform]))


    args.input_size=21

    #output argument to log file
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    train_loader=data.DataLoader(trainset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=PyG_collate)

    val_loader=data.DataLoader(valset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 collate_fn=PyG_collate)

    test_loader=data.DataLoader(testset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 collate_fn=partial(PyG_collate))
    # Get your model
    log.info('Building model...')
    model=get_model(args)
    model.to(device)
    model.train()
    #ema=train_utils.EMA(model, args.ema_decay)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    log.info(f'The total parameters of model :{[pytorch_total_params]}')


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_wd)
    scheduler = ReduceLROnPlateau(optimizer,mode="min",factor=args.factor,patience=args.patience,verbose=True)


    #cri=nn.L1Loss()
    # Train
    log.info('Training GNN model...')
    best_valid_perf = 1E6
    best_test_perf = None
    epoch = 0
    try:
        while epoch != args.num_epochs:
            epoch += 1
            log.info(f'Starting epoch {epoch}...')
            loss=train(model,device,train_loader,optimizer,epoch)
            log.info(f'Evaluating after epoch {epoch}...')
            valid_perf=test(model,device,val_loader)["MAE"]
            scheduler.step(valid_perf)

            if valid_perf < best_valid_perf:
                best_valid_perf = valid_perf
                best_test_perf = test(model,device,test_loader)["MAE"]
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, f'best_model.pth'))

            results = {'Epoch': epoch, 'Loss': loss, 'Cur Val': valid_perf,
                   'Best Val': best_valid_perf, 'Best Test': best_test_perf}
            results_str = ', '.join(f'{k}: {v:05.5f}' for k, v in results.items())
            log.info(results_str)

            if optimizer.param_groups[0]['lr'] < args.min_lr:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break


    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    results = {
               'Best Val': best_valid_perf, 'Best Test': best_test_perf}
    results_str = ', '.join(f'{k}: {v:05.5f}' for k, v in results.items())

    log.info(results_str)

    return

if __name__=="__main__":
    main()