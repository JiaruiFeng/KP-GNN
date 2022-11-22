"""
script to train on SR classification dataset
"""
import argparse
import os
import random
import shutil
import time
from json import dumps

import numpy as np
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel

import train_utils
from data_utils import extract_multi_hop_neighbors, post_transform
from datasets.SRDataset import SRDataset
from layers.input_encoder import EmbeddingEncoder
from layers.layer_utils import make_gnn_layer
from models.GraphClassification import GraphClassification
from models.model_utils import make_GNN


def train(train_loader, model, optimizer, device, parallel=False):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        if parallel:
            num_graphs = len(data)
            y = torch.cat([d.y for d in data]).to(device)
        else:
            num_graphs = data.num_graphs
            data = data.to(device)
            y = data.y
        out = model(data).squeeze()
        loss = torch.nn.CrossEntropyLoss()(out, y)
        loss.backward()
        total_loss += loss.item() * num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader, model, device, parallel=False):
    model.train()  # eliminate the effect of BN
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

    model = GraphClassification(embedding_model=gnn,
                                pooling_method=args.pooling_method,
                                output_size=args.output_size)

    model.reset_parameters()
    if args.parallel:
        model = DataParallel(model, args.gpu_ids)

    return model


def main():
    parser = argparse.ArgumentParser(f'arguments for training and testing')
    parser.add_argument('--save_dir', type=str, default='./save', help='Base directory for saving information.')
    parser.add_argument('--seed', type=int, default=224, help='Random seed for reproducibility.')
    parser.add_argument('--dataset_name', type=str, default="sr25", help='name of dataset')
    parser.add_argument('--drop_prob', type=float, default=0.,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU. Scales automatically when \
                            multiple GPUs are available.')
    parser.add_argument("--parallel", action="store_true",
                        help="If true, use DataParallel for multi-gpu training")
    parser.add_argument('--num_workers', type=int, default=0, help='number of worker.')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load as a model checkpoint.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--l2_wd', type=float, default=0., help='L2 weight decay.')
    parser.add_argument("--kernel", type=str, default="spd", choices=("gd", "spd"),
                        help="the kernel used for K-hop computation")
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument("--hidden_size", type=int, default=48, help="hidden size of the model")
    parser.add_argument("--model_name", type=str, default="KPGIN",
                        choices=("KPGCN", "KPGIN", "KPGraphSAGE", "KPGINPlus"), help="Base GNN model")
    parser.add_argument("--K", type=int, default=4, help="number of hop to consider")
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
                        help='remove peripheral edge information from model')
    parser.add_argument('--wo_peripheral_configuration', action='store_true',
                        help='remove peripheral node configuration from model')
    parser.add_argument("--wo_path_encoding", action="store_true", help="remove path encoding from model")
    parser.add_argument("--wo_edge_feature", action="store_true", help="remove edge feature from model")
    parser.add_argument("--num_hop1_edge", type=int, default=1, help="Number of edge type in hop 1")
    parser.add_argument("--num_layer", type=int, default=4, help="Number of layer for feature encoder")
    parser.add_argument("--JK", type=str, default="last", choices=("sum", "max", "mean", "attention", "last"),
                        help="Jumping knowledge method")
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
    parser.add_argument('--aggr', type=str, default="add",
                        help='aggregation method in GNN layer, only works in GraphSAGE')
    parser.add_argument('--split', type=int, default=10, help='number of fold in cross validation')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='factor in the ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience in the ReduceLROnPlateau learning rate scheduler')

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

    log.info('multi-gpu training is prohibited due to the size '
             'of SR25 dataset and unstable training performace')
    args.parallel = False
    loader = DataLoader

    args.batch_size *= max(1, len(args.gpu_ids))

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

    transform = post_transform(args.wo_path_encoding, args.wo_edge_feature)

    path = "data/" + args.dataset_name
    if os.path.exists(path + '/processed'):
        shutil.rmtree(path + '/processed')
    dataset = SRDataset(path, pre_transform=multihop_transform, transform=transform)
    dataset.data.x = dataset.data.x.long()
    dataset.data.y = torch.arange(len(dataset.data.y)).long()  # each graph is a unique class
    train_dataset = dataset
    val_dataset = dataset
    test_dataset = dataset

    # 2. create loader
    train_loader = loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = loader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    # additional parameter for SR dataset and training
    args.input_size = 2
    args.output_size = len(dataset)

    # output argument to log file
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    # get model
    model = get_model(args)
    model.to(device)
    model.train()
    pytorch_total_params = train_utils.count_parameters(model)
    log.info(f'The total parameters of model :{[pytorch_total_params]}')

    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val_acc, best_test_acc = 0, 0
    start_outer = time.time()
    for epoch in range(args.num_epochs):
        start = time.time()
        train_loss = train(train_loader, model, optimizer, device=device, parallel=args.parallel)
        lr = optimizer.param_groups[0]['lr']
        model.eval()
        val_acc = test(val_loader, model, device=device, parallel=args.parallel)
        test_acc = test(test_loader, model, device=device, parallel=args.parallel)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
        if test_acc >= best_test_acc:
            best_test_acc = test_acc

        time_per_epoch = time.time() - start

        log.info('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                 'Val Acc: {:.7f}, Test Acc: {:.7f}, Best Val Acc: {:.7f}, Best Test Acc: {:.7f}, Seconds: {:.4f}'.format(
            epoch + 1, lr, train_loss, val_acc, test_acc, best_val_acc, best_test_acc, time_per_epoch))
        torch.cuda.empty_cache()  # empty test part memory cost

    time_average_epoch = time.time() - start_outer
    log.info(
        f'Best val: {best_val_acc}, Best test: {best_test_acc}, Seconds/epoch: {time_average_epoch / (epoch+1)}')


if __name__ == "__main__":
    main()
