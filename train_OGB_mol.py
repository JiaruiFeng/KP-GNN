"""
script to train on OGB molecule dataset
Adapted from Nested GNN:https://github.com/muhanzhang/NestedGNN
"""

from tqdm import tqdm
import  os
import random
import pdb
import argparse
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch_geometric.utils import to_networkx
from ogb.graphproppred import Evaluator
from json import dumps
from datasets.pyg_dataset import PygGraphPropPredDataset  # customized to support data list
from data_utils import multi_hop_neighbors_with_gd_kernel,multi_hop_neighbors_with_spd_kernel,PyG_collate,resistance_distance
from models.model_utils import make_OGBMol_gnn_layer,make_GNN
from models.GraphClassification import GraphClassification
from torch.utils.data import DataLoader
import train_utils
from torch_geometric.nn import DataParallel
import torch_geometric.transforms as T
from layers.mol_encoder import AtomEncoder

cls_criterion = torch.nn.BCEWithLogitsLoss
reg_criterion = torch.nn.MSELoss
multicls_criterion = torch.nn.CrossEntropyLoss


def get_model(args):

    layer=make_OGBMol_gnn_layer(args)
    init_emb=AtomEncoder(args.hidden_size)
    GNNModel=make_GNN(args)
    gnn=GNNModel(
                  num_layer=args.num_layer,
                  gnn_layer=layer,
                  JK=args.JK,
                  norm_type=args.norm_type,
                  virtual_node=args.virtual_node,
                  init_emb=init_emb,
                  residual=args.residual,
                  use_rd=args.use_rd,
                  drop_prob=args.drop_prob)

    model=GraphClassification(embedding_model=gnn,
                          pooling_method=args.pooling_method,
                          output_size=args.output_size)
    model.reset_parameters()

    if args.parallel:
        model = DataParallel(model, args.gpu_ids)


    return model



def train(model, device, loader, optimizer, task_type):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration", ncols=70)):
        batch = batch.to(device)
        skip_epoch = batch.x.shape[0] == 1 or batch.batch[-1] == 0

        if skip_epoch:
            pass

        if task_type == 'binary classification':
            train_criterion = cls_criterion
        elif task_type == 'multiclass classification':
            train_criterion = multicls_criterion
        else:
            train_criterion = reg_criterion

        y = batch.y

        if task_type == 'multiclass classification':
            y = y.view(-1, )
        else:
            y = y.to(torch.float32)

        is_labeled = y == y
        pred = model(batch)
        optimizer.zero_grad()

        ## ignore nan targets (unlabeled) when computing training loss.
        loss = train_criterion()(pred.to(torch.float32)[is_labeled],
                                 y[is_labeled])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.shape[0]
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval(model, device, loader, evaluator, return_loss=False,
         task_type=None, checkpoints=[None]):
    model.eval()

    Y_loss = []
    Y_pred = []
    for checkpoint in checkpoints:
        if checkpoint:
            model.load_state_dict(torch.load(checkpoint))

        y_true = []
        y_pred = []
        y_loss = []

        for step, batch in enumerate(tqdm(loader, desc="Iteration", ncols=70)):
            batch = batch.to(device)
            skip_epoch = batch.x.shape[0] == 1

            if skip_epoch:
                pass
            else:
                with torch.no_grad():
                    pred = model(batch)

                y = batch.y

                if task_type == 'multiclass classification':
                    y = y.view(-1, )
                else:
                    y = y.view(pred.shape).to(torch.float32)

                y_true.append(y.detach().cpu())
                y_pred.append(pred.detach().cpu())

            if return_loss:
                if task_type == 'binary classification':
                    train_criterion = cls_criterion
                elif task_type == 'multiclass classification':
                    train_criterion = multicls_criterion
                else:
                    train_criterion = reg_criterion
                loss = train_criterion(reduction='none')(pred.to(torch.float32),
                                                         y)
                loss[torch.isnan(loss)] = 0
                y_loss.append(loss.sum(1).cpu())

        if return_loss:
            y_loss = torch.cat(y_loss, dim=0).numpy()
            Y_loss.append(y_loss)

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        Y_pred.append(y_pred)

    if return_loss:
        y_loss = np.stack(Y_loss).mean(0)
        return y_loss

    y_pred = np.stack(Y_pred).mean(0)

    if task_type == 'multiclass classification':
        y_pred = np.argmax(y_pred, 1).reshape([-1, 1])
        y_true = y_true.reshape([-1, 1])

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    res = evaluator.eval(input_dict)
    return res


def visualize(dataset, save_path, name='vis', number=20, loss=None, sort=True):
    if loss is not None:
        assert (len(loss) == len(dataset))
        if sort:
            order = np.argsort(loss.flatten()).tolist()
        else:
            order = list(range(len(loss.flatten())))
        loader = [dataset.get(i) for i in order[-number:][::-1]]
        # loss = [loss[i] for i in order[::-1]]
        loss = [loss[i] for i in order]
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for idx, data in enumerate(loader):
        f = plt.figure(figsize=(20, 20))
        limits = plt.axis('off')
        if 'name' in data.keys:
            del data.name

        node_size = 300
        with_labels = True
        data.x = data.x[:, 0]
        G = to_networkx(data, node_attrs=['x'])
        labels = {i: G.nodes[i]['x'] for i in range(len(G))}
        if loss is not None:
            label = 'Loss = ' + str(loss[idx])
            print(label)
        else:
            label = ''

        nx.draw_networkx(G, node_size=node_size, arrows=True, with_labels=with_labels,
                         labels=labels)
        plt.title(label)
        f.savefig(os.path.join(save_path, f'{name}_{idx}.png'))
        if (idx + 1) % 5 == 0:
            pdb.set_trace()

def bond_feature_reset(g):
    bond_feature=g.edge_attr
    #0 for mask, 1 for self-loop
    bond_feature=bond_feature+2
    g.bond_feature=bond_feature
    g.edge_attr=None
    return g

def main():
    parser = argparse.ArgumentParser(description=f'arguments for training and testing')
    parser.add_argument('--dataset_name', type=str, default="ogbg-molhiv")
    parser.add_argument('--runs', type=int, default=10, help='how many repeated runs')
    parser.add_argument('--save_dir', type=str, default='./save', help='Base directory for saving information.')
    parser.add_argument('--seed', type=int, default=234, help='Random seed for reproducibility.')
    parser.add_argument('--drop_prob',type=float,default=0.5,help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--batch_size',type=int,default=32,help='Batch size per GPU. Scales automatically when \
                            multiple GPUs are available.')
    parser.add_argument('--num_workers',type=int,default=0,help='number of worker.')
    parser.add_argument('--load_path',type=str,default=None,help='Path to load as a model checkpoint.')
    parser.add_argument('--lr',type=float,default=0.0005,help='Learning rate.')
    parser.add_argument('--min_lr',type=float,default=1e-6,help='minimum Learning rate.')
    parser.add_argument('--l2_wd',type=float,default=3e-4,help='L2 weight decay.')
    parser.add_argument("--kernel",type=str,default="spd",choices=("gd","spd"),help="the kernel used for K-hop computation")
    parser.add_argument('--num_epochs',type=int,default=60,help='Number of epochs.')
    parser.add_argument('--max_grad_norm',type=float,default=5.0,help='Maximum gradient norm for gradient clipping.')
    parser.add_argument("--hidden_size",type=int,default=64,help="hidden size of the model")
    parser.add_argument("--model_name",type=str,default="KPGINPlus",choices=("KPGIN","KPGINPlus"),help="Base GNN model")
    parser.add_argument("--K",type=int,default=8,help="number of hop to consider")
    parser.add_argument('--virtual_node', type=bool, default=True,help='enable using virtual node, default true')
    parser.add_argument("--max_edge_attr_num",type=int,default=1,help="max length in edge attr of hop larger than 1. "
                                                                      "Must be equal to or greater than 1")
    parser.add_argument("--max_peripheral_edge_num",type=int,default=6,help="max number of edge to keep for surround edge,"
                                                                           " 0 means no peripheral edge information")
    parser.add_argument("--max_component_num",type=int,default=3,help="maximum component number in peripheral subgraph information")
    parser.add_argument("--use_edge_feature",type=bool,default=False,help="Whether to use edge feature")
    parser.add_argument("--use_rd",type=bool,default=True,help="Whether to use resistance distance as additional feature")
    parser.add_argument("--num_layer",type=int,default=8,help="number of layer for feature encoder")
    parser.add_argument("--JK",type=str,default="concat",choices=("sum","max","mean","attention","last","concat"),help="Jumping knowledge method")
    parser.add_argument("--residual",type=bool,default=False,help="Whether to use residual connection between each layer")
    parser.add_argument("--eps",type=float,default=0.,help="Initital epsilon in GIN")
    parser.add_argument("--train_eps",type=bool,default=False,help="Whether the epsilon is trainable")
    parser.add_argument("--negative_slope",type=float,default=0.2,help="slope in LeakyRelu")
    parser.add_argument("--combine",type=str,default="geometric",choices=("attention","geometric"),help="Jumping knowledge method")
    parser.add_argument("--pooling_method",type=str,default="mean",choices=("mean","sum","attention"),help="pooling method in graph classification")
    parser.add_argument('--norm_type',type=str,default="Batch",choices=("Batch","Layer","Instance","GraphSize","Pair"),
                        help="normalization method in model")
    parser.add_argument('--aggr',type=str,default="add",help='aggregation method in GNN layer, only works in GraphSAGE')
    parser.add_argument('--factor',type=float,default=0.5,help='factor for reducing learning rate scheduler')
    parser.add_argument('--ensemble', action='store_true', default=False,
                        help='load a series of model checkpoints and ensemble the results')
    parser.add_argument('--ensemble_lookback', type=int, default=90,
                        help='how many epochs to look back in ensemble')
    parser.add_argument('--ensemble_interval', type=int, default=10,
                        help='ensemble every x epochs')

    parser.add_argument('--visualize_all', action='store_true', default=False,
                        help='visualize all graphs in dataset sequentially')
    parser.add_argument('--visualize_test', action='store_true', default=False,
                        help='visualize test graphs by loss')
    parser.add_argument('--pre_visualize', action='store_true', default=False)
    parser.add_argument('--continue_from', type=int, default=None,
                        help="from which epoch's checkpoint to continue training")
    parser.add_argument('--run_from', type=int, default=1,
                        help="from which run (of multiple repeated experiments) to start")
    parser.add_argument('--save_appendix', type=str, default='',
                        help='appendix to save results')
    parser.add_argument('--log_steps', type=int, default=10,
                        help='save model checkpoint every x epochs')

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

    transform = None
    if args.dataset_name == 'ogbg-ppa':
        def add_zeros(data):
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)
            return data

        transform = add_zeros

    path="data/OGB_mol_"
    path=path+str(args.K)+"_"+str(args.max_peripheral_edge_num)+"_"+str(args.max_component_num)+"_"+str(args.max_edge_attr_num)+"_"+args.kernel
    dataset = PygGraphPropPredDataset(
        name=args.dataset_name, root=path, transform=transform, pre_transform=T.Compose([rd_feature,bond_feature_reset,pre_transform]),
        skip_collate=False)

    split_idx = dataset.get_idx_split()

    evaluator = Evaluator(args.dataset_name)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers,collate_fn=PyG_collate)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers,collate_fn=PyG_collate)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers,collate_fn=PyG_collate)

    if args.pre_visualize:
        visualize(dataset, args.save_dir)


    args.output_size = dataset.num_tasks if args.dataset_name.startswith('ogbg-mol') else dataset.num_classes
    args.input_size=dataset.num_node_features
    #output argument to log file
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    valid_perfs, test_perfs = [], []
    start_run = args.run_from - 1
    runs = args.runs - args.run_from + 1
    for run in range(start_run, start_run + runs):
        model=get_model(args)
        model.to(device)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        log.info(f'The total parameters of model :{[pytorch_total_params]}')

        optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.l2_wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                                    gamma=args.factor)
        #ema = train_utils.EMA(model, args.ema_decay)

        start_epoch = 1
        epochs = args.num_epochs
        if args.continue_from is not None:
            model.load_state_dict(
                torch.load(os.path.join(args.load_path,
                                        'run{}_model_checkpoint{}.pth'.format(run + 1, args.continue_from)))
            )
            optimizer.load_state_dict(
                torch.load(os.path.join(args.load_path,
                                        'run{}_optimizer_checkpoint{}.pth'.format(run + 1, args.continue_from)))
            )
            start_epoch = args.continue_from + 1
            epochs = epochs - args.continue_from

        if args.visualize_all:  # visualize all graphs
            model.load_state_dict(torch.load(os.path.join(args.res_dir, 'best_model.pth')))
            dataset = dataset[:100]
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            all_losses = eval(model, device, loader, evaluator, True,
                              dataset.task_type).flatten()
            visualize(dataset, args.res_dir, 'all_vis', loss=all_losses, sort=False)

        if args.visualize_test:
            model.load_state_dict(torch.load(os.path.join(args.res_dir, 'best_model.pth')))
            test_losses = eval(model, device, test_loader, evaluator, True,
                               dataset.task_type).flatten()
            visualize(dataset[split_idx["test"]], args.res_dir, 'test_vis', loss=test_losses)

        # Training begins.
        eval_metric = dataset.eval_metric
        best_valid_perf = -1E6 if 'classification' in dataset.task_type else 1E6
        best_test_perf = None
        for epoch in range(start_epoch, start_epoch + epochs):
            log.info(f"=====Run {run + 1}, epoch {epoch}, {args.save_appendix}")
            log.info('Training...')
            loss = train(model, device, train_loader, optimizer, dataset.task_type)

            log.info('Evaluating...')
            valid_perf = eval(model, device, valid_loader, evaluator, False,
                              dataset.task_type)[eval_metric]
            if 'classification' in dataset.task_type:
                if valid_perf > best_valid_perf:
                    best_valid_perf = valid_perf
                    best_test_perf = eval(model, device, test_loader, evaluator, False,
                                          dataset.task_type)[eval_metric]
                    torch.save(model.state_dict(),
                               os.path.join(args.save_dir, f'run{run + 1}_best_model.pth'))
            else:
                if valid_perf < best_valid_perf:
                    best_valid_perf = valid_perf
                    best_test_perf = eval(model, device, test_loader, evaluator, False,
                                          dataset.task_type)[eval_metric]
                    torch.save(model.state_dict(),
                               os.path.join(args.save_dir, f'run{run + 1}_best_model.pth'))
            scheduler.step()

            results = {'Epoch': epoch, 'Loss': loss, 'Cur Val': valid_perf,
                   'Best Val': best_valid_perf, 'Best Test': best_test_perf}
            results_str = ', '.join(f'{k}: {v:05.5f}' for k, v in results.items())
            log.info(results_str)

            if epoch % args.log_steps == 0:
                model_name = os.path.join(
                    args.save_dir, 'run{}_model_checkpoint{}.pth'.format(run + 1, epoch))
                optimizer_name = os.path.join(
                    args.save_dir, 'run{}_optimizer_checkpoint{}.pth'.format(run + 1, epoch))
                torch.save(model.state_dict(), model_name)
                torch.save(optimizer.state_dict(), optimizer_name)

        final_res = '''Run {}\nBest validation score: {}\nTest score: {}
        '''.format(run + 1, best_valid_perf, best_test_perf)
        log.info('Finished training!')
        log.info(final_res)


        if args.ensemble:
            log.info('Start ensemble testing...')
            start_epoch, end_epoch = args.num_epochs - args.ensemble_lookback, args.num_epochs
            checkpoints = [
                os.path.join(args.load_path, 'run{}_model_checkpoint{}.pth'.format(run + 1, x))
                for x in range(start_epoch, end_epoch + 1, args.ensemble_interval)
            ]
            ensemble_valid_perf = eval(model, device, valid_loader, evaluator, False,
                                       dataset.task_type, checkpoints)[eval_metric]
            ensemble_test_perf = eval(model, device, test_loader, evaluator, False,
                                      dataset.task_type, checkpoints)[eval_metric]
            ensemble_res = '''Run {}\nEnsemble validation score: {}\nEnsemble test score: {}
            '''.format(run + 1, ensemble_valid_perf, ensemble_test_perf)
            log.info(ensemble_res)


        if args.ensemble:
            valid_perfs.append(ensemble_valid_perf)
            test_perfs.append(ensemble_test_perf)
        else:
            valid_perfs.append(best_valid_perf)
            test_perfs.append(best_test_perf)

    valid_perfs = torch.tensor(valid_perfs)
    test_perfs = torch.tensor(test_perfs)
    log.info('===========================')
    log.info(f'Final Valid: {valid_perfs.mean():.4f} ± {valid_perfs.std():.4f}')
    log.info(f'Final Test: {test_perfs.mean():.4f} ± {test_perfs.std():.4f}')
    log.info(valid_perfs.tolist())
    log.info(test_perfs.tolist())


if __name__=="__main__":
    main()

