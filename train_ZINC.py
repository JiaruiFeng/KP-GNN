"""
script to train on ZINC task
"""
import argparse
import os
import random
import shutil
from collections import OrderedDict
from json import dumps
from time import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import train_utils
from data_utils import extract_multi_hop_neighbors, PyG_collate, resistance_distance, post_transform
from datasets.ZINC_dataset import ZINC
from layers.input_encoder import EmbeddingEncoder
from layers.layer_utils import make_gnn_layer
from models.GraphRegression import GraphRegression
from models.model_utils import make_GNN
from train_utils import count_parameters


# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
def train(model, device, loader, optimizer, epoch):
    total_loss = 0
    with torch.enable_grad(), \
            tqdm(total=len(loader.dataset)) as progress_bar:
        for data in loader:
            y = data.y
            y = y.to(device)
            data = data.to(device)
            batch_size = data.num_graphs
            optimizer.zero_grad()
            # forward
            score = model(data)
            loss = (score - y).abs().mean()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(),5.0)
            optimizer.step()
            mae = loss.detach().item()
            progress_bar.update(batch_size)
            loss_val = loss.item()
            lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(epoch=epoch,
                                     loss=loss_val,
                                     mae=mae,
                                     lr=lr)
            total_loss += loss.item() * batch_size
    return total_loss / len(loader.dataset)


def test(model, device, loader):
    total_mae = 0
    model.eval()
    N = 0
    with torch.no_grad(), \
            tqdm(total=len(loader.dataset)) as progress_bar:
        for data in loader:
            y = data.y
            y = y.to(device)
            data = data.to(device)
            batch_size = data.num_graphs
            # forward
            score = model(data)
            total_mae += (score - y).abs().sum().item()
            N += batch_size
            mae = total_mae / N
            progress_bar.update(batch_size)
            progress_bar.set_postfix(MAE=mae)

    mae = total_mae / N

    model.train()
    results_list = [
        ('Loss', mae),
        ('MAE', mae)
    ]
    results = OrderedDict(results_list)
    return results


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


def convert_edge_labels(data):
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr + 1
    return data


def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE


def main():
    parser = argparse.ArgumentParser(f'arguments for training and testing')
    parser.add_argument('--save_dir', type=str, default='./save', help='Base directory for saving information.')
    parser.add_argument('--seed', type=int, default=234, help='Random seed for reproducibility.')
    parser.add_argument('--dataset_name', type=str, default="ZINC", help='name of dataset')
    parser.add_argument('--drop_prob', type=float, default=0.0,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU. Scales automatically when \
                            multiple GPUs are available.')
    parser.add_argument('--num_workers', type=int, default=0, help='number of worker.')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load as a model checkpoint.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate.')
    parser.add_argument('--l2_wd', type=float, default=0.0, help='L2 weight decay.')
    parser.add_argument("--kernel", type=str, default="spd", choices=("gd", "spd"),
                        help="the kernel used for K-hop computation")
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Maximum gradient norm for gradient clipping.')
    parser.add_argument("--hidden_size", type=int, default=104, help="hidden size of the model")
    parser.add_argument("--model_name", type=str, default="KPGINPlus", choices=("KPGIN", "KPGINPlus"),
                        help="Base GNN model")
    parser.add_argument("--K", type=int, default=8, help="number of hop to consider")
    parser.add_argument("--max_pe_num", type=int, default=50,
                        help="Maximum number of path encoding. Must be equal to or greater than 1")
    parser.add_argument("--max_edge_type", type=int, default=3,
                        help="Maximum number of type of edge to consider in peripheral edge information")
    parser.add_argument("--max_edge_count", type=int, default=50,
                        help="Maximum count per edge type in peripheral edge information")
    parser.add_argument("--max_hop_num", type=int, default=6,
                        help="Maximum number of hop to consider in peripheral configuration information")
    parser.add_argument("--max_distance_count", type=int, default=50,
                        help="Maximum count per hop in peripheral configuration information")
    parser.add_argument('--wo_peripheral_edge', action='store_true',
                        help='remove peripheral edge information from model')
    parser.add_argument('--wo_peripheral_configuration', action='store_true',
                        help='remove peripheral node configuration information from model')
    parser.add_argument('--wo_path_encoding', action='store_true',
                        help='remove path encoding information from model')
    parser.add_argument('--wo_edge_feature', action='store_true',
                        help='remove edge feature from model')
    parser.add_argument("--num_hop1_edge", type=int, default=3, help="number of edge type in hop 1")
    parser.add_argument("--num_layer", type=int, default=8, help="number of layer for feature encoder")
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
    parser.add_argument('--patience', type=float, default=10,
                        help='factor in the ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--reprocess', action="store_true", help='Whether to reprocess the dataset')
    parser.add_argument('--runs', type=int, default=4, help='number of repeat run')

    args = parser.parse_args()
    if args.wo_path_encoding:
        args.num_hopk_edge = 1
    else:
        args.num_hopk_edge = args.max_pe_num

    args.name = args.model_name + "_" + args.kernel + "_" + str(args.K) + "_" + str(args.wo_peripheral_edge) + \
                "_" + str(args.wo_peripheral_configuration) + "_" + str(args.wo_path_encoding) + "_" + \
                str(args.wo_edge_feature)

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

    path = "data/ZINC_"
    path = path + str(args.K) + "_" + args.kernel
    if os.path.exists(path) and args.reprocess:
        shutil.rmtree(path)

    trainset = ZINC(path, subset=True, split="train",
                    pre_transform=T.Compose([convert_edge_labels, rd_feature, pre_transform]),
                    transform=transform)
    valset = ZINC(path, subset=True, split="val",
                  pre_transform=T.Compose([convert_edge_labels, rd_feature, pre_transform]),
                  transform=transform)
    testset = ZINC(path, subset=True, split="test",
                   pre_transform=T.Compose([convert_edge_labels, rd_feature, pre_transform]),
                   transform=transform)

    args.input_size = 21

    # output argument to log file
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    train_loader = data.DataLoader(trainset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=0,
                                   collate_fn=PyG_collate)

    val_loader = data.DataLoader(valset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 collate_fn=PyG_collate)

    test_loader = data.DataLoader(testset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  collate_fn=PyG_collate)

    test_perfs = []
    vali_perfs = []
    for run in range(1, args.runs + 1):
        # Get your model
        log.info('Building model...')
        model = get_model(args)
        model.to(device)
        model.train()
        # ema=train_utils.EMA(model, args.ema_decay)

        pytorch_total_params = count_parameters(model)
        log.info(f'The total parameters of model :{[pytorch_total_params]}')

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_wd)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=args.factor, patience=args.patience)

        # cri=nn.L1Loss()
        # Train
        log.info('Training GNN model...')
        best_valid_perf = 1E6
        best_test_perf = None
        epoch = 0
        try:
            while epoch != args.num_epochs:
                epoch += 1
                log.info(f'Starting epoch {epoch}...')
                start = time()
                loss = train(model, device, train_loader, optimizer, epoch)
                time_per_epoch = time() - start
                log.info(f'Evaluating after epoch {epoch}...')
                valid_perf = test(model, device, val_loader)["MAE"]
                scheduler.step(valid_perf)

                if valid_perf < best_valid_perf:
                    best_valid_perf = valid_perf
                    best_test_perf = test(model, device, test_loader)["MAE"]
                    torch.save(model.state_dict(),
                               os.path.join(args.save_dir, f'best_model.pth'))

                results = {'Epoch': epoch, 'Loss': loss, 'Cur Val': valid_perf,
                           'Best Val': best_valid_perf, 'Best Test': best_test_perf, "Time per epoch": time_per_epoch}
                results_str = ', '.join(f'{k}: {v:05.5f}' for k, v in results.items())
                log.info(f"Run: {run}, {results_str}")

                if optimizer.param_groups[0]['lr'] < args.min_lr:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break


        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early because of KeyboardInterrupt')
        results = {
            'Best Val': best_valid_perf, 'Best Test': best_test_perf}
        results_str = ', '.join(f'{k}: {v:05.5f}' for k, v in results.items())
        log.info(f"Run: {run}, {results_str}")
        test_perfs.append(best_test_perf)
        vali_perfs.append(best_valid_perf)

        test_perf = torch.tensor(test_perfs)
        vali_perf = torch.tensor(vali_perfs)
        log.info("-" * 50)
        # logger.info(cfg)
        log.info(
            f'Final Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}, Final Test: {test_perf.mean():.4f} ± {test_perf.std():.4f}')

    return


if __name__ == "__main__":
    main()
