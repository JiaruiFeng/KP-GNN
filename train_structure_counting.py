"""
script to train on graph substructure counting task
Adpated from https://github.com/LingxiaoShawn/GNNAsKernel
"""
import argparse
import os
import random
import shutil
import time
from json import dumps

import numpy as np
import torch
import torch_geometric.transforms as T
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import train_utils
from data_utils import extract_multi_hop_neighbors, PyG_collate, resistance_distance, post_transform
from datasets.GraphCountDataset import GraphCountDataset
from layers.input_encoder import EmbeddingEncoder
from layers.layer_utils import make_gnn_layer
from models.GraphRegression import GraphRegression
from models.model_utils import make_GNN


# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
def train(train_loader, model, task, optimizer, device):
    total_loss = 0
    N = 0
    with torch.enable_grad(), \
            tqdm(total=len(train_loader.dataset)) as progress_bar:
        for data in train_loader:
            data = data.to(device)
            batch_size = data.num_graphs
            optimizer.zero_grad()
            pre = model(data).squeeze()
            loss = (pre - data.y[:, task:task + 1].squeeze()).square().mean()

            loss.backward()
            total_loss += loss.item() * data.num_graphs
            N += data.num_graphs
            optimizer.step()
            progress_bar.update(batch_size)

    return total_loss / N


@torch.no_grad()
def test(loader, model, task, device):
    model.eval()
    total_error = 0
    N = 0
    with torch.no_grad(), \
            tqdm(total=len(loader.dataset)) as progress_bar:
        for data in loader:
            data = data.to(device)
            batch_size = data.num_graphs
            total_error += (model(data).squeeze() - data.y[:, task:task + 1].squeeze()).square().sum().item()
            N += data.num_graphs
            progress_bar.update(batch_size)
    model.train()
    return total_error / N


def get_model(args):
    layer = make_gnn_layer(args)
    init_emb = EmbeddingEncoder(args.input_size, args.hidden_size)
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

    # If use multiple gpu, torch geometric model must use DataParallel class
    # if args.parallel:
    #    model = DataParallel(model, args.gpu_ids)

    return model


def main():
    parser = argparse.ArgumentParser(f'arguments for training and testing')
    parser.add_argument('--save_dir', type=str, default='./save', help='Base directory for saving information.')
    parser.add_argument('--seed', type=int, default=234, help='Random seed for reproducibility.')
    parser.add_argument('--dataset_name', type=str, default="GraphCount", help='name of dataset')
    parser.add_argument('--task', type=int, default=0, choices=(0, 1, 2, 3, 4), help='number of task')
    parser.add_argument('--drop_prob', type=float, default=0.,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU. Scales automatically when \
                            multiple GPUs are available.')
    parser.add_argument('--num_workers', type=int, default=0, help='number of worker.')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load as a model checkpoint.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate.')
    parser.add_argument('--l2_wd', type=float, default=3e-7, help='L2 weight decay.')
    parser.add_argument("--kernel", type=str, default="spd", choices=("gd", "spd"),
                        help="the kernel used for K-hop computation")
    parser.add_argument('--num_epochs', type=int, default=250, help='Number of epochs.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Maximum gradient norm for gradient clipping.')
    parser.add_argument("--hidden_size", type=int, default=96, help="hidden size of the model")
    parser.add_argument("--model_name", type=str, default="KPGINPlus", choices=("KPGIN", "KPGINPlus"),
                        help="Jumping knowledge method")
    parser.add_argument("--K", type=int, default=3, help="number of hop to consider")
    parser.add_argument("--max_pe_num", type=int, default=50,
                        help="Maximum number of path encoding. Must be equal to or greater than 1")
    parser.add_argument("--max_edge_type", type=int, default=1,
                        help="Maximum number of type of edge to consider in peripheral edge information")
    parser.add_argument("--max_edge_count", type=int, default=50,
                        help="Maximum count per edge type in peripheral edge information")
    parser.add_argument("--max_hop_num", type=int, default=5,
                        help="Maximum number of hop to consider in peripheral configuration information")
    parser.add_argument("--max_distance_count", type=int, default=100,
                        help="Maximum count per hop in peripheral configuration information")
    parser.add_argument('--wo_peripheral_edge', action='store_true',
                        help='remove peripheral edge information from model')
    parser.add_argument('--wo_peripheral_configuration', action='store_true',
                        help='remove peripheral node configuration from model')
    parser.add_argument("--wo_path_encoding", action="store_true", help="remove path encoding from model")
    parser.add_argument("--wo_edge_feature", action="store_true", help="remove edge feature from model")
    parser.add_argument("--num_hop1_edge", type=int, default=1, help="number of edge type in hop 1")
    parser.add_argument("--num_layer", type=int, default=3, help="number of layer for feature encoder")
    parser.add_argument("--JK", type=str, default="concat",
                        choices=("sum", "max", "mean", "attention", "last", "concat"), help="Jumping knowledge method")
    parser.add_argument("--residual", action="store_true", help="Whether to use residual connection between each layer")
    parser.add_argument("--use_rd", action="store_true", help="Whether to add resistance distance feature to model")
    parser.add_argument("--virtual_node", action="store_true",
                        help="Whether add virtual node information in each layer")
    parser.add_argument("--eps", type=float, default=0., help="Initital epsilon in GIN")
    parser.add_argument("--train_eps", action="store_true", help="Whether the epsilon is trainable")
    parser.add_argument("--combine", type=str, default="geometric", choices=("attention", "geometric"),
                        help="Jumping knowledge method")
    parser.add_argument("--pooling_method", type=str, default="sum", choices=("mean", "sum", "attention"),
                        help="pooling method in graph classification")
    parser.add_argument('--norm_type', type=str, default="Batch",
                        choices=("Batch", "Layer", "Instance", "GraphSize", "Pair"),
                        help="normalization method in model")
    parser.add_argument('--factor', type=float, default=0.5,
                        help='factor in the ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience in the ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--reprocess', type=bool, default=True, help='Whether to reprocess the dataset')
    parser.add_argument('--runs', type=int, default=4, help='number of repeat run')

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
    if len(args.gpu_ids) > 1:
        args.parallel = True
    else:
        args.parallel = False
    args.batch_size *= max(1, len(args.gpu_ids))
    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def pre_transform(g):
        return extract_multi_hop_neighbors(g, args.K, args.max_pe_num, args.max_hop_num, args.max_edge_type,
                                           args.max_edge_count,
                                           args.max_distance_count, args.kernel)

    if args.use_rd:
        rd_feature = resistance_distance
    else:
        def rd_feature(g):
            return g

    transform = post_transform(args.wo_path_encoding, args.wo_edge_feature)

    root = 'data/subgraphcount'
    if os.path.exists(root + '/processed') and args.reprocess:
        shutil.rmtree(root + '/processed')

    dataset = GraphCountDataset(root, pre_transform=T.Compose([pre_transform, rd_feature]), transform=transform)
    dataset.data.y = dataset.data.y / dataset.data.y.std(0)
    train_dataset, val_dataset, test_dataset = dataset[dataset.train_idx], dataset[dataset.val_idx], dataset[
        dataset.test_idx]
    train_dataset = [x for x in train_dataset]
    val_dataset = [x for x in val_dataset]
    test_dataset = [x for x in test_dataset]

    args.input_size = 2
    args.output_size = 1

    # output argument to log file
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=PyG_collate)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=PyG_collate)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=PyG_collate)

    test_perfs = []
    vali_perfs = []

    for run in range(1, args.runs + 1):

        model = get_model(args)
        model.reset_parameters()
        model.to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_wd)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.factor, patience=args.patience)

        start_outer = time.time()
        best_val_perf = test_perf = float('inf')
        for epoch in range(1, args.num_epochs + 1):
            start = time.time()
            model.train()
            train_loss = train(train_loader, model, args.task, optimizer, device=device)
            val_perf = test(val_loader, model, args.task, device=device)
            lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_perf)
            if val_perf < best_val_perf:
                best_val_perf = val_perf
                test_perf = test(test_loader, model, args.task, device=device)
            time_per_epoch = time.time() - start

            # logger here
            log.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                     f'Val: {val_perf:.4f}, Test: {test_perf:.4f}, lr:{lr:.7f}, Seconds: {time_per_epoch:.4f}, ')
            if optimizer.param_groups[0]['lr'] < args.min_lr:
                log.info("\n!! LR EQUAL TO MIN LR SET.")
                break
            torch.cuda.empty_cache()  # empty test part memory cost

        time_average_epoch = time.time() - start_outer
        log.info(
            f'Run {run}, Vali: {best_val_perf}, Test: {test_perf}, Seconds/epoch: {time_average_epoch / args.num_epochs}')
        test_perfs.append(test_perf)
        vali_perfs.append(best_val_perf)

    test_perf = torch.tensor(test_perfs)
    vali_perf = torch.tensor(vali_perfs)
    log.info("-" * 50)
    # logger.info(cfg)
    log.info(
        f'Final Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}, Final Test: {test_perf.mean():.4f} ± {test_perf.std():.4f},'
        f'Seconds/epoch: {time_average_epoch / args.num_epochs}')


if __name__ == "__main__":
    main()
