"""
script to train on EXP classification dataset
"""
import numpy as np
import random
import torch
import torch.nn as nn
import train_utils
from json import dumps
from datasets.PlanarSATPairsDataset import PlanarSATPairsDataset
from tqdm import tqdm
from torch.optim import Adam
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import DataParallel
from data_utils import multi_hop_neighbors_with_gd_kernel,multi_hop_neighbors_with_spd_kernel,PyG_collate,resistance_distance
from models.model_utils import make_gnn_layer,make_GNN
from models.GraphClassification import GraphClassification
import shutil
import os
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

    model=GraphClassification(embedding_model=gnn,
                          pooling_method=args.pooling_method,
                              output_size=args.output_size)

    model.reset_parameters()

    if args.parallel:
        model = DataParallel(model, args.gpu_ids)


    return model


class MyFilter(object):
    def __call__(self, data):
        return True  # No Filtering


class MyPreTransform(object):
    def __call__(self, data):
        data.x = data.x[:, 0].to(torch.long)
        return data

def val(data_loader,model,device):
    model.eval()
    loss_all = 0
    with torch.no_grad(), \
         tqdm(total=len(data_loader.dataset)) as progress_bar:
        for batch_graphs in data_loader:
            batch_graphs = batch_graphs.to(device)
            batch_size=batch_graphs.num_graphs
            predict=model(batch_graphs)
            predict=F.log_softmax(predict,dim=-1)
            loss=F.nll_loss(predict,batch_graphs.y, reduction='sum').item()
            loss_all += loss
            progress_bar.update(batch_size)

    model.train()
    return loss_all / len(data_loader.dataset)

def test(data_loader,model,device):
    model.eval()
    correct = 0
    with torch.no_grad(), \
         tqdm(total=len(data_loader.dataset)) as progress_bar:
        for batch_graphs in data_loader:
            batch_graphs = batch_graphs.to(device)
            batch_size=batch_graphs.num_graphs
            nb_trials = 1   # Support majority vote, but single trial is default
            successful_trials = torch.zeros_like(batch_graphs.y)
            for i in range(nb_trials):  # Majority Vote
                pred = model(batch_graphs).max(1)[1]
                successful_trials += pred.eq(batch_graphs.y)
            successful_trials = successful_trials > (nb_trials // 2)
            correct += successful_trials.sum().item()
            progress_bar.update(batch_size)
    model.train()
    return correct / len(data_loader.dataset)

def main():
    parser = argparse.ArgumentParser(f'arguments for training and testing')
    parser.add_argument('--save_dir',type=str,default='./save',help='Base directory for saving information.')
    parser.add_argument('--seed',type=int,default=224,help='Random seed for reproducibility.')
    parser.add_argument('--dataset_name',type=str,default="EXP",choices=("EXP","CEXP"),help='name of dataset')
    parser.add_argument('--drop_prob',type=float,default=0.,help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--batch_size',type=int,default=128,help='Batch size per GPU. Scales automatically when \
                            multiple GPUs are available.')
    parser.add_argument('--num_workers',type=int,default=0,help='number of worker.')
    parser.add_argument('--load_path',type=str,default=None,help='Path to load as a model checkpoint.')
    parser.add_argument('--lr',type=float,default=0.001,help='Learning rate.')
    parser.add_argument('--min_lr',type=float,default=1e-6,help='minimum Learning rate.')
    parser.add_argument('--l2_wd',type=float,default=3e-7,help='L2 weight decay.')
    parser.add_argument("--kernel",type=str,default="spd",choices=("gd","spd"),help="the kernel used for K-hop computation")
    parser.add_argument('--num_epochs',type=int,default=40,help='Number of epochs.')
    parser.add_argument('--max_grad_norm',type=float,default=5.0,help='Maximum gradient norm for gradient clipping.')
    parser.add_argument("--hidden_size",type=int,default=60,help="hidden size of the model")
    parser.add_argument("--model_name",type=str,default="KPGINPlus",choices=("KPGCN","KPGIN","KPGraphSAGE","KPGINPlus"),help="Base GNN model")
    parser.add_argument("--K",type=int,default=3,help="number of hop to consider")
    parser.add_argument("--max_edge_attr_num",type=int,default=1,help="max length in edge attr of hop larger than 1. "
                                                                      "Must be equal to or greater than 1")
    parser.add_argument("--max_peripheral_edge_num",type=int,default=10,help="max number of edge to keep for peripheral edge,"
                                                                           " 0 means no peripheral subgraph information")
    parser.add_argument("--max_component_num",type=int,default=0,help="max number of component in peripheral subgraph information")
    parser.add_argument("--use_edge_feature",type=bool,default=False,help="Whether to use edge feature")
    parser.add_argument("--num_hop1_edge",type=int,default=3,help="number of edge type in hop 1")
    parser.add_argument("--num_layer",type=int,default=4,help="number of layer for feature encoder")
    parser.add_argument("--JK",type=str,default="last",choices=("sum","max","mean","attention","last"),help="Jumping knowledge method")
    parser.add_argument("--residual",type=bool,default=False,help="Whether to use residual connection between each layer")
    parser.add_argument("--use_rd",type=bool,default=False,help="Whether to add resistance distance feature to model")
    parser.add_argument("--virtual_node",type=bool,default=False,help="Whether add virtual node information in each layer")
    parser.add_argument("--eps",type=float,default=0.,help="Initital epsilon in GIN")
    parser.add_argument("--train_eps",type=bool,default=True,help="Whether the epsilon is trainable")
    parser.add_argument("--negative_slope",type=float,default=0.2,help="slope in LeakyRelu")
    parser.add_argument("--combine",type=str,default="attention",choices=("attention","geometric"),help="Jumping knowledge method")
    parser.add_argument("--pooling_method",type=str,default="sum",choices=("mean","sum","attention"),help="pooling method in graph classification")
    parser.add_argument('--norm_type',type=str,default="Batch",choices=("Batch","Layer","Instance","GraphSize","Pair"),
                        help="normalization method in model")
    parser.add_argument('--aggr',type=str,default="add",help='aggregation method in GNN layer, only works in GraphSAGE')
    parser.add_argument('--split',type=int,default=10,help='number of fold in cross validation')
    parser.add_argument('--factor',type=float,default=0.5,help='factor in the ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--patience',type=int,default=5,help='patience in the ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--reprocess',type=bool,default=True,help='Whether to reprocess the dataset')
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
    if args.use_rd:
        rd_feature=resistance_distance
    else:
        def rd_feature(g):
            return g

    path="data/" + args.dataset_name
    if os.path.exists(path+'/processed') and args.reprocess:
        shutil.rmtree( path+'/processed')

    dataset = PlanarSATPairsDataset(root=path,
                                    pre_transform=T.Compose([MyPreTransform(),pre_transform,rd_feature]),
                                    pre_filter=MyFilter())

    #additional parameter for EXP dataset and training
    args.input_size=2
    args.output_size=dataset.num_classes
    args.MODULO=4
    args.MOD_THRESH=1

    #output argument to log file
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    tr_accuracies = np.zeros((args.num_epochs, args.split))
    tst_accuracies = np.zeros((args.num_epochs, args.split))
    tst_exp_accuracies = np.zeros((args.num_epochs, args.split))
    tst_lrn_accuracies = np.zeros((args.num_epochs, args.split))
    acc = []
    tr_acc = []

    for i in range(args.split):
        log.info(f"---------------Training on fold {i}------------------------")
        model=get_model(args)
        model.to(device)
        model.train()
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        log.info(f'The total parameters of model :{[pytorch_total_params]}')

        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=args.factor, patience=args.patience, min_lr=args.min_lr)

        #K-fold cross validation split
        n = len(dataset) // args.split
        test_mask = torch.zeros(len(dataset))
        test_exp_mask = torch.zeros(len(dataset))
        test_lrn_mask = torch.zeros(len(dataset))

        test_mask[i * n:(i + 1) * n] = 1  # Now set the masks
        learning_indices = [x for idx, x in enumerate(range(n * i, n * (i + 1))) if x % args.MODULO <= args.MOD_THRESH]
        test_lrn_mask[learning_indices] = 1
        exp_indices = [x for idx, x in enumerate(range(n * i, n * (i + 1))) if x % args.MODULO > args.MOD_THRESH]
        test_exp_mask[exp_indices] = 1

        # Now load the datasets
        test_dataset = dataset[test_mask.bool()]
        test_exp_dataset = dataset[test_exp_mask.bool()]
        test_lrn_dataset = dataset[test_lrn_mask.bool()]
        train_dataset = dataset[(1 - test_mask).bool()]

        n = len(train_dataset) // args.split
        val_mask = torch.zeros(len(train_dataset), dtype=torch.bool)
        val_mask[i * n:(i + 1) * n] = 1
        val_dataset = train_dataset[val_mask]
        train_dataset = train_dataset[~val_mask]

        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size,collate_fn=PyG_collate)
        test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size,collate_fn=PyG_collate)
        test_exp_loader = data.DataLoader(test_exp_dataset, batch_size=args.batch_size,collate_fn=PyG_collate)  # These are the new test splits
        test_lrn_loader = data.DataLoader(test_lrn_dataset, batch_size=args.batch_size,collate_fn=PyG_collate)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=PyG_collate)
        best_val_loss, best_test_acc,best_train_acc = 100, 0,0
        for epoch in range(args.num_epochs):
            log.info(f'Starting epoch {epoch+1}...')
            with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
                for graphs in train_loader:
                    graphs = graphs.to(device)
                    batch_size = graphs.num_graphs
                    optimizer.zero_grad()
                    predict = model(graphs)
                    predict = F.log_softmax(predict, dim=-1)
                    loss = F.nll_loss(predict, graphs.y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    model.zero_grad()
                    # Log info
                    progress_bar.update(batch_size)
                    loss_val = loss.item()
                    lr = optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix(epoch=epoch + 1,
                                             loss=loss_val,
                                             lr=lr)


                log.info(f"evaluate after epoch {epoch+1}...")
                train_loss=val(train_loader,model,device)
                val_loss = val(val_loader,model,device)
                scheduler.step(val_loss)
                train_acc = test(train_loader,model,device)
                test_acc = test(test_loader,model,device)
                if best_val_loss >= val_loss:
                    best_val_loss = val_loss
                    best_train_acc=train_acc
                    best_test_acc=test_acc

                test_exp_acc = test(test_exp_loader,model,device)
                test_lrn_acc = test(test_lrn_loader,model,device)
                tr_accuracies[epoch, i] = train_acc
                tst_accuracies[epoch, i] = test_acc
                tst_exp_accuracies[epoch, i] = test_exp_acc
                tst_lrn_accuracies[epoch, i] = test_lrn_acc
                log.info('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                  'Val Loss: {:.7f}, Test Acc: {:.7f}, Exp Acc: {:.7f}, Lrn Acc: {:.7f}, Train Acc: {:.7f}'.format(
                      epoch+1, lr, train_loss, val_loss, test_acc, test_exp_acc, test_lrn_acc, train_acc))
        acc.append(best_test_acc)
        tr_acc.append(best_train_acc)
    acc = torch.tensor(acc)
    tr_acc = torch.tensor(tr_acc)
    log.info("-------------------Print final result-------------------------")
    log.info(f"Train result: Mean: {tr_acc.mean().item()}, Std :{tr_acc.std().item()}")
    log.info(f"Test result: Mean: {acc.mean().item()}, Std :{acc.std().item()}")


if __name__=="__main__":
    main()