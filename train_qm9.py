"""
script to train on QM9 task
"""

import argparse
import os
import random
import shutil
import time
from json import dumps

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel
from tqdm import tqdm

import train_utils
from data_utils import extract_multi_hop_neighbors, resistance_distance, post_transform
from datasets.QM9Dataset import QM9, conversion
from layers.input_encoder import QM9InputEncoder
from layers.layer_utils import make_gnn_layer
from models.GraphRegression import GraphRegression
from models.model_utils import make_GNN


def get_model(args):
    layer = make_gnn_layer(args)
    init_emb = QM9InputEncoder(args.hidden_size, args.use_pos)
    GNNModel = make_GNN(args)
    gnn = GNNModel(
        num_layer=args.num_layer,
        gnn_layer=layer,
        JK=args.JK,
        norm_type=args.norm_type,
        init_emb=init_emb,
        residual=args.residual,
        virtual_node=args.virtual_node,
        use_rd=args.use_rd,
        num_hop1_edge=args.num_hop1_edge,
        max_edge_count=args.max_edge_count,
        max_hop_num=args.max_hop_num,
        max_distance_count=args.max_distance_count,
        wo_peripheral_edge=args.wo_peripheral_edge,
        wo_peripheral_configuration=args.wo_peripheral_configuration,
        drop_prob=args.drop_prob)

    model = GraphRegression(embedding_model=gnn,
                            pooling_method=args.pooling_method)

    model.reset_parameters()
    if args.parallel:
        model = DataParallel(model, args.gpu_ids)

    return model


class TargetTransform(object):
    def __init__(self, target, pre_convert=False):
        self.target = target
        self.pre_convert = pre_convert

    def __call__(self, data):
        data.y = data.y[:, int(self.target)]  # Specify target: 0 = mu
        if self.pre_convert:  # convert back to original units
            data.y = data.y / conversion[int(self.target)]
        return data


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes > 6  # Remove graphs with less than 6 nodes.


def edge_feature_transform(data):
    if data.edge_attr is not None:
        data.edge_attr = torch.where(data.edge_attr == 1)[1] + 2
    return data


def train(loader, model, optimizer, device, parallel=False):
    model.train()
    loss_all = 0
    for data in loader:
        optimizer.zero_grad()
        if parallel:
            num_graphs = len(data)
            y = torch.cat([d.y for d in data]).to(device)
        else:
            num_graphs = data.num_graphs
            data = data.to(device)
            y = data.y
        loss = F.mse_loss(model(data), y)
        loss.backward()
        loss_all += loss * num_graphs
        optimizer.step()
    return loss_all / len(loader.dataset)


@torch.no_grad()
def test(loader, model, task, std, device, parallel=False):
    model.eval()
    error = 0
    for data in loader:
        if parallel:
            y = torch.cat([d.y for d in data]).to(device)
        else:
            data = data.to(device)
            y = data.y
        error += ((model(data) * std[task].cuda()) -
                  (y * std[task].cuda())).abs().sum().item()  # MAE
    return error / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(f'arguments for training and testing')
    parser.add_argument('--save_dir', type=str, default='./save', help='Base directory for saving information.')
    parser.add_argument('--seed', type=int, default=234, help='Random seed for reproducibility.')
    parser.add_argument('--dataset_name', type=str, default="QM9", help='name of dataset')
    parser.add_argument('--task', type=int, default=11, choices=list(range(19)), help='number of task')
    parser.add_argument('--drop_prob', type=float, default=0.0,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU. Scales automatically when \
                            multiple GPUs are available.')
    parser.add_argument("--parallel", action="store_true",
                        help="If true, use DataParallel for multi-gpu training")
    parser.add_argument('--num_workers', type=int, default=0, help='number of worker.')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load as a model checkpoint.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate.')
    parser.add_argument('--l2_wd', type=float, default=0.0, help='L2 weight decay.')
    parser.add_argument("--kernel", type=str, default="spd", choices=("gd", "spd"),
                        help="the kernel used for K-hop computation")
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of the model")
    parser.add_argument("--model_name", type=str, default="KPGINPlus", choices=("KPGIN", "KPGINPlus"),
                        help="Base GNN model")
    parser.add_argument("--K", type=int, default=8, help="number of hop to consider")
    parser.add_argument("--max_pe_num", type=int, default=50,
                        help="Maximum number of path encoding. Must be equal to or greater than 1")
    parser.add_argument("--max_edge_type", type=int, default=4,
                        help="Maximum number of type of edge to consider in peripheral edge information")
    parser.add_argument("--max_edge_count", type=int, default=20,
                        help="Maximum count per edge type in peripheral edge information")
    parser.add_argument("--max_hop_num", type=int, default=5,
                        help="Maximum number of hop to consider in peripheral configuration information")
    parser.add_argument("--max_distance_count", type=int, default=15,
                        help="Maximum count per hop in peripheral configuration information")
    parser.add_argument('--wo_peripheral_edge', action='store_true',
                        help='remove peripheral edge information from model')
    parser.add_argument('--wo_peripheral_configuration', action='store_true',
                        help='remove peripheral node configuration from model')
    parser.add_argument("--wo_path_encoding", action="store_true", help="remove path encoding from model")
    parser.add_argument("--wo_edge_feature", action="store_true", help="remove edge feature from model")
    parser.add_argument("--num_hop1_edge", type=int, default=4, help="Number of edge type in hop 1")
    parser.add_argument("--num_layer", type=int, default=8, help="Number of layer for feature encoder")
    parser.add_argument("--JK", type=str, default="concat", choices=("sum", "max", "mean", "attention", "last"),
                        help="Jumping knowledge method")
    parser.add_argument("--residual", action="store_true", help="Whether to use residual connection between each layer")
    parser.add_argument("--eps", type=float, default=0., help="Initital epsilon in GIN")
    parser.add_argument("--train_eps", action="store_true", help="Whether the epsilon is trainable")
    parser.add_argument("--combine", type=str, default="attention", choices=("attention", "geometric"),
                        help="Jumping knowledge method")
    parser.add_argument("--pooling_method", type=str, default="attention", choices=("mean", "sum", "attention"),
                        help="pooling method in graph classification")
    parser.add_argument('--norm_type', type=str, default="Batch",
                        choices=("Batch", "Layer", "Instance", "GraphSize", "Pair"),
                        help="normalization method in model")
    parser.add_argument('--factor', type=float, default=0.7,
                        help='factor in the ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience in the ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--reprocess', action="store_true", help='Whether to reprocess the dataset')
    parser.add_argument('--normalize_x', action='store_true', default=False,
                        help='if True, normalize non-binary node features')
    parser.add_argument('--virtual_node', action="store_true", help="virtual node")
    parser.add_argument('--use_pos', action='store_true', default=False,
                        help='use node position (3D) as continuous node features')
    parser.add_argument('--use_rd', action='store_true', help='use resistance distance as additional node labels')
    parser.add_argument('--filter', action='store_true', help='whether to filter graphs with less than 7 nodes')
    parser.add_argument('--convert', type=str, default='post',
                        help='if "post", convert units after optimization; if "pre", \
                        convert units before optimization')

    args = parser.parse_args()
    if args.wo_path_encoding:
        args.num_hopk_edge = 1
    else:
        args.num_hopk_edge = args.max_pe_num

    args.name = args.model_name + "_" + args.kernel + "_" + str(args.K) + "_" + str(args.wo_peripheral_edge) + \
                "_" + str(args.wo_peripheral_configuration) + "_" + str(args.wo_path_encoding) + "_" + \
                str(args.wo_edge_feature) + "_" + str(args.task)

    # Set up logging and devices
    args.save_dir = train_utils.get_save_dir(args.save_dir, args.name, type=args.dataset_name)
    log = train_utils.get_logger(args.save_dir, args.name)
    device, args.gpu_ids = train_utils.get_available_devices()
    if len(args.gpu_ids) > 1 and args.parallel:
        log.info(f'Using multi-gpu training')
        args.parallel = True
        loader = DataListLoader
        args.batch_size *= max(1, len(args.gpu_ids))
    else:
        log.info(f'Using single-gpu training')
        args.parallel = False
        loader = DataLoader

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def multihop_transform(g):
        return extract_multi_hop_neighbors(g, args.K, args.max_pe_num, args.max_hop_num, args.max_edge_type,
                                           args.max_edge_count,
                                           args.max_distance_count, args.kernel)

    if args.use_rd:
        rd_feature = resistance_distance
    else:
        def rd_feature(g):
            return g

    transform = post_transform(args.wo_path_encoding, args.wo_edge_feature)

    path = "data/qm9_"
    path = path + str(args.K) + "_" + args.kernel
    if os.path.exists(path + '/processed') and args.reprocess:
        shutil.rmtree(path + '/processed')

    pre_filter = None
    if args.filter:
        pre_filter = MyFilter()
        path += '_filtered'

    dataset = QM9(path, pre_transform=T.Compose([edge_feature_transform, multihop_transform, rd_feature]),
                  transform=T.Compose([TargetTransform(args.task, args.convert == 'pre'), transform]),
                  pre_filter=pre_filter)
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

    test_loader = loader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = loader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    input_size = 19
    if args.use_pos:
        input_size += 3
    args.input_size = input_size
    args.output_size = 1

    # output argument to log file
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    model = get_model(args)
    model.to(device)
    pytorch_total_params = train_utils.count_parameters(model)
    log.info(f'The total parameters of model :{[pytorch_total_params]}')

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=args.factor, patience=args.patience)

    best_val_error = 1E6
    test_error = 1E6
    start_outer = time.time()
    pbar = tqdm(range(1, args.num_epochs + 1))
    for epoch in pbar:
        start = time.time()
        pbar.set_description('Epoch: {:03d}'.format(epoch))
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(train_loader, model, optimizer, device, parallel=args.parallel)
        val_error = test(val_loader, model, args.task, std, device, parallel=args.parallel)
        scheduler.step(val_error)

        if val_error <= best_val_error:
            best_val_error = val_error
            test_error = test(test_loader, model, args.task, std, device, parallel=args.parallel)
        time_per_epoch = time.time() - start
        info = (
                'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, ' +
                'Test MAE: {:.7f}, Test MAE norm: {:.7f}, Test MAE convert: {:.7f}, Seconds: {:.4f}'
        ).format(
            epoch, lr, loss, val_error,
            test_error,
            test_error / std[args.task].cuda(),
            test_error / conversion[int(args.task)].cuda() if args.convert == 'post' else 0,
            time_per_epoch
        )
        log.info(info)
        if optimizer.param_groups[0]['lr'] < args.min_lr:
            log.info("\n!! LR EQUAL TO MIN LR SET.")
            break
        torch.cuda.empty_cache()  # empty test part memory cost
    time_average_epoch = time.time() - start_outer
    info = ('Validation MAE: {:.7f}, Test MAE: {:.7f}, Test MAE norm: {:.7f}, Test MAE convert: {:.7f}, Seconds/epoch: {:.4f}'
    ).format(
        best_val_error,
        test_error,
        test_error / std[args.task].cuda(),
        test_error / conversion[int(args.task)].cuda() if args.convert == 'post' else 0,
        time_average_epoch / epoch
    )
    log.info(info)


if __name__ == "__main__":
    main()
