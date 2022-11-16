"""
script to train on TU dataset with GIN setting:https://github.com/weihua916/powerful-gnns
"""
import argparse
import os
import random
import shutil
import time
from itertools import product
from json import dumps

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch_geometric.data import DenseDataLoader as DenseLoader
from tqdm import tqdm

import train_utils
from data_utils import extract_multi_hop_neighbors, PyG_collate, resistance_distance, post_transform
from datasets.tu_dataset import TUDatasetGINSplit, TUDataset
from layers.input_encoder import LinearEncoder
from layers.layer_utils import make_gnn_layer
from models.GraphClassification import GraphClassification
from models.model_utils import make_GNN


# os.environ["CUDA_LAUNCH_BLOCKING"]="1"

def cross_validation_GIN_split(dataset, model, collate, epochs, batch_size, lr, factor, weight_decay, device, log=None):
    """Cross validation framework with GIN split.
    Args:
        dataset(PyG.dataset): PyG dataset for training and testing
        model(nn.Module): GNN model
        collate(function): function for generate batch data
        epochs(int): number of epochs in the training of each fold
        batch_size: batch size of training
        lr(float): learning rate
        factor(float): reduce factor in learning rate scheduler
        weight_decay(float): L2 weight decay regularization
        device(str): training device
        log(logger): log file
    """
    folds = 10
    lr_decay_step_size = 50
    test_losses, accs, durations = [], [], []
    count = 1
    k_fold_indices = dataset.train_indices, dataset.test_indices
    for fold, (train_idx, test_idx) in enumerate(zip(*k_fold_indices)):
        print("CV fold " + str(count))
        count += 1
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate)

        model.to(device)
        model.reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        t_start = time.perf_counter()

        pbar = tqdm(range(1, epochs + 1), ncols=70)
        for epoch in pbar:
            train_loss = train_utils.train_TU(model, optimizer, train_loader, device)
            test_loss = train_utils.val_TU(model, test_loader, device)
            test_losses.append(test_loss)
            accs.append(train_utils.test_TU(model, test_loader, device))
            eval_info = {
                'fold': fold + 1,
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_losses[-1],
                'test_acc': accs[-1],
            }
            info = 'Fold: %d, train_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f' % (
                fold + 1, eval_info["train_loss"], eval_info["test_loss"], eval_info["test_acc"]
            )
            log.info(info)

            # decay the learning rate
            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = factor * param_group["lr"]

        # if torch.cuda.is_available():
        #    torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = torch.tensor(test_losses), torch.tensor(accs), torch.tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    acc_max, _ = acc.max(1)
    acc_mean = acc.mean(0)
    acc_cross_epoch_max, argmax = acc_mean.max(dim=0)
    acc_final = acc_mean[-1]

    info = ('Test Loss: {:.4f}, Test Max Accuracy:{:.3f} ± {:.3f}, Test Max Cross Epoch Accuracy: {:.3f} ± {:.3f}, ' +
            'Test Final Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}').format(
        loss.mean().item(),
        acc_max.mean().item(),
        acc_max.std().item(),
        acc_cross_epoch_max.item(),
        acc[:, argmax].std().item(),
        acc_final.item(),
        acc[:, -1].std().item(),
        duration.mean().item()
    )
    log.info(info)

    return (acc_max.mean().item(), acc_max.std().item()), \
           (acc_cross_epoch_max.item(), acc[:, argmax].std().item()), \
           (acc_final.item(), acc[:, -1].std().item())


def cross_validation_with_PyG_dataset(dataset, model, collate, folds, epochs, batch_size, lr, factor, weight_decay,
                                      device, log=None, seed=234):
    """Cross validation framework without validation dataset. Adapted from Nested GNN:https://github.com/muhanzhang/NestedGNN
    Args:
        dataset(PyG.dataset): PyG dataset for training and testing
        model(nn.Module): GNN model
        collate(function): function for generate batch data
        folds(int): number of fold in cross validation
        epochs(int): number of epochs in the training of each fold
        batch_size: batch size of training
        lr(float): learning rate
        factor(float): reduce factor in learning rate scheduler
        weight_decay(float): L2 weight decay regularization
        device(str): training device
        log(logger): log file
        seed(int): random seed
    """
    lr_decay_step_size = 50
    test_losses, accs, durations = [], [], []
    count = 1
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*train_utils.k_fold(dataset, folds, seed))):
        print("CV fold " + str(count))
        count += 1

        train_idx = torch.cat([train_idx, val_idx], 0)  # combine train and val
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate)

        model.to(device)
        model.reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # if torch.cuda.is_available():
        #   torch.cuda.synchronize()

        t_start = time.perf_counter()

        pbar = tqdm(range(1, epochs + 1), ncols=70)
        for epoch in pbar:
            train_loss = train_utils.train_TU(model, optimizer, train_loader, device)
            test_loss = train_utils.val_TU(model, test_loader, device)
            test_losses.append(test_loss)
            accs.append(train_utils.test_TU(model, test_loader, device))
            eval_info = {
                'fold': fold + 1,
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_losses[-1],
                'test_acc': accs[-1],
            }
            info = 'Fold: %d, train_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f' % (
                fold + 1, eval_info["train_loss"], eval_info["test_loss"], eval_info["test_acc"]
            )
            log.info(info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = factor * param_group["lr"]

        # if torch.cuda.is_available():
        #   torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = torch.tensor(test_losses), torch.tensor(accs), torch.tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    acc_max, _ = acc.max(1)
    acc_mean = acc.mean(0)
    acc_cross_epoch_max, argmax = acc_mean.max(dim=0)
    acc_final = acc_mean[-1]

    info = ('Test Loss: {:.4f}, Test Max Accuracy:{:.3f} ± {:.3f}, Test Max Cross Epoch Accuracy: {:.3f} ± {:.3f}, ' +
            'Test Final Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}').format(
        loss.mean().item(),
        acc_max.mean().item(),
        acc_max.std().item(),
        acc_cross_epoch_max.item(),
        acc[:, argmax].std().item(),
        acc_final.item(),
        acc[:, -1].std().item(),
        duration.mean().item()
    )
    log.info(info)

    return (acc_max.mean().item(), acc_max.std().item()), \
           (acc_cross_epoch_max.item(), acc[:, argmax].std().item()), \
           (acc_final.item(), acc[:, -1].std().item())


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

    # If use multiple gpu, torch geometric model must use DataParallel class
    # if args.parallel:
    #    model = DataParallel(model, args.gpu_ids)
    return model


def convert_edge_labels(data):
    if data.edge_attr is not None:
        data.edge_attr = torch.where(data.edge_attr == 1)[1] + 2
    return data


def main():
    parser = argparse.ArgumentParser(f'arguments for training and testing')
    parser.add_argument('--save_dir', type=str, default='./save', help='Base directory for saving information.')
    parser.add_argument('--seed', type=int, default=234, help='Random seed for reproducibility.')
    parser.add_argument('--dataset_name', type=str, default="MUTAG",
                        choices=("MUTAG", "DD", "PROTEINS", "PTC", "IMDBBINARY"), help='name of dataset')
    parser.add_argument('--drop_prob', type=float, default=0.5,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU. Scales automatically when \
                            multiple GPUs are available.')
    parser.add_argument('--num_workers', type=int, default=0, help='number of worker.')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load as a model checkpoint.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--l2_wd', type=float, default=3e-4, help='L2 weight decay.')
    parser.add_argument("--kernel", type=str, default="gd", choices=("gd", "spd"),
                        help="the kernel used for K-hop computation")
    parser.add_argument('--num_epochs', type=int, default=350, help='Number of epochs.')
    parser.add_argument("--hidden_size", type=int, default=32, help="hidden size of the model")
    parser.add_argument("--model_name", type=str, default="KPGCN",
                        choices=("KPGCN", "KPGIN", "KPGraphSAGE", "KPGINPlus"), help="Model name")
    parser.add_argument("--K", type=int, default=2, help="number of hop to consider")
    parser.add_argument("--max_pe_num", type=int, default=30,
                        help="Maximum number of path encoding. Must be equal to or greater than 1")
    parser.add_argument("--max_edge_type", type=int, default=1,
                        help="Maximum number of type of edge to consider in peripheral edge information")
    parser.add_argument("--max_edge_count", type=int, default=30,
                        help="Maximum count per edge type in peripheral edge information")
    parser.add_argument("--max_hop_num", type=int, default=5,
                        help="Maximum number of hop to consider in peripheral configuration information")
    parser.add_argument("--max_distance_count", type=int, default=50,
                        help="Maximum count per hop in peripheral configuration information")
    parser.add_argument('--wo_peripheral_edge', action='store_true',
                        help='remove peripheral edge information from model')
    parser.add_argument('--wo_peripheral_configuration', action='store_true',
                        help='remove peripheral node configuration from model')
    parser.add_argument("--wo_path_encoding", action="store_true", help="remove path encoding from model")
    parser.add_argument("--wo_edge_feature", action="store_true", help="remove edge feature from model")
    parser.add_argument("--num_hop1_edge", type=int, default=1, help="number of edge type in hop 1")
    parser.add_argument("--num_layer", type=int, default=2, help="number of layer for feature encoder")
    parser.add_argument("--JK", type=str, default="last", choices=("sum", "max", "mean", "attention", "last", "concat"),
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
    parser.add_argument('--factor', type=float, default=0.5, help='factor for reducing learning rate scheduler')
    parser.add_argument('--reprocess', action="store_true", help='Whether to reprocess the dataset')
    parser.add_argument('--search', action="store_true", help='Search hyperparameters')
    args = parser.parse_args()
    if args.wo_path_encoding:
        args.num_hopk_edge = 1
    else:
        args.num_hopk_edge = args.max_pe_num

    args.name = args.model_name + "_" + args.kernel + "_" + str(args.K) + "_" + str(args.wo_peripheral_edge) + \
                "_" + str(args.wo_peripheral_configuration) + "_" + str(args.wo_path_encoding) + "_" + \
                str(args.wo_edge_feature) + "_" + str(args.search)
    # Set up logging and devices
    args.save_dir = train_utils.get_save_dir(args.save_dir, args.name, type=args.dataset_name + "_GIN")
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

    # output argument to log file
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    if args.search:
        log.info("----------------------------------------------------------------")
        kernels = ["spd", "gd"]
        Ks = [2, 3, 4]
        layers = [2, 3, 4]
        combines = ["geometric", "attention"]
        t = product(kernels, Ks, layers, combines)
        best_graphSNN_result = (0, 1e30)
        best_gin_result = (0, 1e30)
        best_final_result = (0, 1e30)
        try:
            for parameters in t:
                kernel, K, layer, combine = parameters
                args.combine = combine
                args.kernel = kernel
                args.K = K
                args.num_layer = layer
                if K == 3:
                    args.hidden_size = 33
                else:
                    args.hidden_size = 32

                log.info(f"Search on kernel:{kernel},K:{K},layer:{layer},combine:{combine}")
                path = "data/TUGIN_"
                path = path + str(args.K) + "_" + args.kernel
                if os.path.exists(path + "/" + args.dataset_name + '/processed') and args.reprocess:
                    shutil.rmtree(path + "/" + args.dataset_name + '/processed')
                if args.dataset_name == "DD":
                    dataset = TUDataset(path, args.dataset_name,
                                        pre_transform=T.Compose([convert_edge_labels, pre_transform, rd_feature]),
                                        transform=transform,
                                        cleaned=False)
                else:
                    dataset = TUDatasetGINSplit(args.dataset_name, path,
                                                pre_transform=T.Compose(
                                                    [convert_edge_labels, pre_transform, rd_feature]),
                                                transform=transform)

                args.input_size = dataset.num_node_features
                args.output_size = dataset.num_classes

                model = get_model(args)
                if args.dataset_name == "DD":
                    graphsnn_setting_result, gin_setting_result, final_epoch_result = cross_validation_with_PyG_dataset(
                        dataset, model, collate=PyG_collate, folds=10,
                        epochs=args.num_epochs,
                        batch_size=args.batch_size, lr=args.lr,
                        factor=args.factor,
                        weight_decay=args.l2_wd,
                        device=device, log=log)
                else:
                    graphsnn_setting_result, gin_setting_result, final_epoch_result = cross_validation_GIN_split(
                        dataset, model, collate=PyG_collate, epochs=args.num_epochs,
                        batch_size=args.batch_size, lr=args.lr, factor=args.factor,
                        weight_decay=args.l2_wd,
                        device=device, log=log)

                if graphsnn_setting_result[0] > best_graphSNN_result[0] or \
                        (graphsnn_setting_result[0] == best_graphSNN_result[0] and graphsnn_setting_result[0] <
                         best_graphSNN_result[0]):
                    best_graphSNN_result = graphsnn_setting_result
                    best_graphSNN_paras = (kernel, K, layer, combine)

                if gin_setting_result[0] > best_gin_result[0] or \
                        (gin_setting_result[0] == best_gin_result[0] and gin_setting_result[0] < best_gin_result[0]):
                    best_gin_result = gin_setting_result
                    best_gin_paras = (kernel, K, layer, combine)

                if final_epoch_result[0] > best_final_result[0] or \
                        (final_epoch_result[0] == best_final_result[0] and final_epoch_result[0] < best_final_result[
                            0]):
                    best_final_result = final_epoch_result
                    best_final_paras = (kernel, K, layer, combine)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early because of KeyboardInterrupt')

        log.info("----------------------------Complete search-----------------------------------")
        desc = '{:.3f} ± {:.3f}'.format(
            best_graphSNN_result[0], best_graphSNN_result[1]
        )
        log.info(f'Best result on graphSNN setting - {desc}, '
                 f'with {str(best_graphSNN_paras[0])} kernel,'
                 f'{str(best_graphSNN_paras[1])} hop, '
                 f'{str(best_graphSNN_paras[2])} layers, '
                 f'and {str(best_graphSNN_paras[3])} combination')
        log.info("===============================================================================")
        desc = '{:.3f} ± {:.3f}'.format(
            best_gin_result[0], best_gin_result[1]
        )
        log.info(f'Best result on GIN setting - {desc}, '
                 f'with {str(best_gin_paras[0])} kernel,'
                 f'{str(best_gin_paras[1])} hop, '
                 f'{str(best_gin_paras[2])} layers, '
                 f'and {str(best_gin_paras[3])} combination')
        log.info("===============================================================================")
        desc = '{:.3f} ± {:.3f}'.format(
            best_final_result[0], best_final_result[1]
        )
        log.info(f'Best result on final epoch - {desc}, '
                 f'with {str(best_final_paras[0])} kernel,'
                 f'{str(best_final_paras[1])} hop, '
                 f'{str(best_final_paras[2])} layers, '
                 f'and {str(best_final_paras[3])} combination')

    else:
        path = "data/TUGIN_"
        path = path + str(args.K) + "_" + args.kernel

        if os.path.exists(path + "/" + args.dataset_name + '/processed') and args.reprocess:
            shutil.rmtree(path + "/" + args.dataset_name + '/processed')
        if args.dataset_name == "DD":
            dataset = TUDataset(path, args.dataset_name,
                                pre_transform=T.Compose([convert_edge_labels, pre_transform, rd_feature]),
                                transform=transform,
                                cleaned=False)

        else:
            dataset = TUDatasetGINSplit(args.dataset_name, path,
                                        pre_transform=T.Compose([convert_edge_labels, pre_transform, rd_feature]),
                                        transform=transform)

        args.input_size = dataset.num_node_features
        args.output_size = dataset.num_classes

        model = get_model(args)
        if args.dataset_name == "DD":
            graphsnn_setting_result, gin_setting_result, final_epoch_result = cross_validation_with_PyG_dataset(dataset,
                                                                                                                model,
                                                                                                                collate=PyG_collate,
                                                                                                                folds=10,
                                                                                                                epochs=args.num_epochs,
                                                                                                                batch_size=args.batch_size,
                                                                                                                lr=args.lr,
                                                                                                                factor=args.factor,
                                                                                                                weight_decay=args.l2_wd,
                                                                                                                device=device,
                                                                                                                log=log)
        else:
            graphsnn_setting_result, gin_setting_result, final_epoch_result = cross_validation_GIN_split(dataset, model,
                                                                                                         collate=PyG_collate,
                                                                                                         epochs=args.num_epochs,
                                                                                                         batch_size=args.batch_size,
                                                                                                         lr=args.lr,
                                                                                                         factor=args.factor,
                                                                                                         weight_decay=args.l2_wd,
                                                                                                         device=device,
                                                                                                         log=log)


if __name__ == "__main__":
    main()
