"""
script to train on CSL classification dataset
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
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.seed import seed_everything
import torch.nn.functional as F
import train_utils
from data_utils import extract_multi_hop_neighbors, post_transform, resistance_distance
from layers.input_encoder import LinearEncoder
from layers.layer_utils import make_gnn_layer
from models.GraphClassification import GraphClassification
from models.model_utils import make_GNN


def train(loader, model, optimizer, device, parallel=False):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        if parallel:
            num_graphs = len(data)
            y = torch.cat([d.y for d in data]).to(device)
        else:
            num_graphs = data.num_graphs
            data = data.to(device)
            y = data.y
        out = model(data).squeeze()
        loss = F.cross_entropy(out, y)
        loss.backward()
        total_loss += loss.item() * num_graphs
        optimizer.step()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def val(loader, model, device, parallel=False):
    model.eval()
    y_preds, y_trues = [], []
    for data in loader:
        if parallel:
            y = torch.cat([d.y for d in data]).to(device)
        else:
            data = data.to(device)
            y = data.y
        y_preds.append(torch.argmax(model(data), dim=-1))
        y_trues.append(y)
    y_preds = torch.cat(y_preds, -1)
    y_trues = torch.cat(y_trues, -1)
    return (y_preds == y_trues).float().mean()


def get_model(args):
    layer = make_gnn_layer(args)
    init_emb = LinearEncoder(args.input_size, args.hidden_size)
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

    model = GraphClassification(embedding_model=gnn,
                                pooling_method=args.pooling_method,
                                output_size=args.output_size)

    model.reset_parameters()
    if args.parallel:
        model = DataParallel(model, args.gpu_ids)

    return model


def node_feature_transform(data):
    if "x" not in data:
        data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)
    return data


def main():
    parser = argparse.ArgumentParser(f'arguments for training and testing')
    parser.add_argument('--save_dir', type=str, default='./save', help='Base directory for saving information.')
    parser.add_argument('--seed', type=int, default=224, help='Random seed for reproducibility.')
    parser.add_argument('--dataset_name', type=str, default="CSL", help='Name of dataset')
    parser.add_argument('--drop_prob', type=float, default=0.,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU. Scales automatically when \
                            multiple GPUs are available.')
    parser.add_argument("--parallel", action="store_true",
                        help="If true, use DataParallel for multi-gpu training")
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker.')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load as a model checkpoint.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--l2_wd', type=float, default=3e-6, help='L2 weight decay.')
    parser.add_argument("--kernel", type=str, default="spd", choices=("gd", "spd"),
                        help="The kernel used for K-hop computation")
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument("--hidden_size", type=int, default=48, help="Hidden size of the model")
    parser.add_argument("--model_name", type=str, default="KPGIN",
                        choices=("KPGCN", "KPGIN", "KPGraphSAGE", "KPGINPlus"), help="Base GNN model")
    parser.add_argument("--K", type=int, default=4, help="Number of hop to consider")
    parser.add_argument("--max_pe_num", type=int, default=1000,
                        help="Maximum number of path encoding. Must be equal to or greater than 1")
    parser.add_argument("--max_edge_type", type=int, default=1,
                        help="Maximum number of type of edge to consider in peripheral edge information")
    parser.add_argument("--max_edge_count", type=int, default=1000,
                        help="Maximum count per edge type in peripheral edge information")
    parser.add_argument("--max_hop_num", type=int, default=4,
                        help="Maximum number of hop to consider in peripheral configuration information")
    parser.add_argument("--max_distance_count", type=int, default=1000,
                        help="Maximum count per hop in peripheral configuration information")
    parser.add_argument('--wo_peripheral_edge', action='store_true',
                        help='If true, remove peripheral edge information from model')
    parser.add_argument('--wo_peripheral_configuration', action='store_true',
                        help='If true, remove peripheral node configuration from model')
    parser.add_argument("--wo_path_encoding", action="store_true", help="If true, remove path encoding from model")
    parser.add_argument("--wo_edge_feature", action="store_true", help="If true, remove edge feature from model")
    parser.add_argument("--num_hop1_edge", type=int, default=1, help="Number of edge type in hop 1")
    parser.add_argument("--num_layer", type=int, default=4, help="Number of layer for feature encoder")
    parser.add_argument("--JK", type=str, default="last", choices=("sum", "max", "mean", "attention", "last"),
                        help="Jumping knowledge method")
    parser.add_argument("--residual", action="store_true", help="If true, use residual connection between each layer")
    parser.add_argument("--use_rd", action="store_true", help="If true, add resistance distance feature to model")
    parser.add_argument("--virtual_node", action="store_true",
                        help="If true, add virtual node information in each layer")
    parser.add_argument("--eps", type=float, default=0., help="Initial epsilon in GIN")
    parser.add_argument("--train_eps", action="store_true", help="Whether the epsilon is trainable")
    parser.add_argument("--combine", type=str, default="geometric", choices=("attention", "geometric"),
                        help="Combine method in k-hop aggregation")
    parser.add_argument("--pooling_method", type=str, default="sum", choices=("mean", "sum", "attention"),
                        help="Pooling method in graph classification")
    parser.add_argument('--norm_type', type=str, default="Batch",
                        choices=("Batch", "Layer", "Instance", "GraphSize", "Pair"),
                        help="Normalization method in model")
    parser.add_argument('--aggr', type=str, default="add",
                        help='Aggregation method in GNN layer, only works in GraphSAGE')
    parser.add_argument('--split', type=int, default=10, help='Number of fold in cross validation')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='Factor in the ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience in the ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--reprocess', action="store_true", help='If true, reprocess the dataset')

    args = parser.parse_args()
    if args.wo_path_encoding:
        args.num_hopk_edge = 1
    else:
        args.num_hopk_edge = args.max_pe_num

    args.name = args.model_name + "_" + args.kernel + "_" + str(args.K) + "_" + str(args.wo_peripheral_edge) + \
                "_" + str(args.wo_peripheral_configuration) + "_" + str(args.wo_path_encoding) + "_" + str(
        args.wo_edge_feature)

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
    seed = train_utils.get_seed(args.seed)
    log.info(f'Using random seed {seed}...')
    seed_everything(seed)

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

    path = "data/GNNBenchmarking"
    if os.path.exists(path + "/" + args.dataset_name + '/processed') and args.reprocess:
        shutil.rmtree(path + "/" + args.dataset_name + '/processed')

    dataset = GNNBenchmarkDataset(root=path, name=args.dataset_name,
                                  pre_transform=T.Compose([node_feature_transform, multihop_transform, rd_feature]),
                                  transform=transform)

    args.input_size = dataset.num_features
    args.output_size = dataset.num_classes

    # output argument to log file
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    val_accs = np.zeros(args.split)
    test_accs = np.zeros(args.split)

    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*train_utils.k_fold(dataset, args.split, args.seed))):

        log.info(f"---------------Training on fold {fold + 1}------------------------")
        # get model
        model = get_model(args)
        model.to(device)
        pytorch_total_params = train_utils.count_parameters(model)
        log.info(f'The total parameters of model :{[pytorch_total_params]}')

        optimizer = Adam(model.parameters(), lr=args.lr)
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]


        val_loader = loader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_loader = loader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        train_loader = loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        best_val_loss, best_val_acc, best_test_acc, best_train_acc = 100, 0, 0, 0
        start_outer = time.time()
        for epoch in range(args.num_epochs):
            start = time.time()
            train_loss = train(train_loader, model, optimizer, device=device, parallel=args.parallel)
            lr = optimizer.param_groups[0]['lr']
            val_acc = val(val_loader, model, device=device, parallel=args.parallel)
            if val_acc >= best_val_acc:
                test_acc = val(test_loader, model, device=device, parallel=args.parallel)
                best_val_acc = val_acc
                best_test_acc = test_acc

            time_per_epoch = time.time() - start

            log.info('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                     'Val Acc: {:.7f}, Test Acc: {:.7f}, Best Val Acc: {:.7f}, Best Test Acc: {:.7f}, Seconds: {:.4f}'.format(
                epoch + 1, lr, train_loss, val_acc, test_acc, best_val_acc, best_test_acc, time_per_epoch))

            torch.cuda.empty_cache()  # empty test part memory cost

        time_average_epoch = time.time() - start_outer
        log.info(
            f'Fold {fold + 1}, Vali: {best_val_acc}, Test: {best_test_acc}, Seconds/epoch: {time_average_epoch / (epoch+1)}')
        val_accs[fold] = best_val_acc
        test_accs[fold] = best_test_acc

    val_accs = torch.tensor(val_accs)
    test_accs = torch.tensor(test_accs)
    log.info("-------------------Print final result-------------------------")
    log.info(f"Validation result: Mean: {val_accs.mean().item()}, Std :{val_accs.std().item()}")
    log.info(f"Test result: Mean: {test_accs.mean().item()}, Std :{test_accs.std().item()}")


if __name__ == "__main__":
    main()
