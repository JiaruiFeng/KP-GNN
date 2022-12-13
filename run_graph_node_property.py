import os
from itertools import product
import argparse

parser = argparse.ArgumentParser(f'add multi-gpu option')
parser.add_argument("--parallel", action="store_true",
                    help="If true, use DataParallel for multi-gpu training")
parallel_arg = parser.parse_args()
parallel = parallel_arg.parallel

tasks = [0, 1, 2]
ks = [3, 4, 5, 6]
grid = product(tasks, ks)

if parallel:
    for parameter in grid:
        task, k = parameter
        # KP-GNN with path encoding graph tasks
        script = f"python train_graph_property.py --task={str(task)} --K={k} --num_layer={k} --parallel"
        os.system(script)
        # KP-GNN with path encoding node tasks
        script = f"python train_node_property.py --task={str(task)} --K={k} --num_layer={k} --parallel"
        os.system(script)
        # KP-GNN without path encoding graph tasks
        script = f"python train_graph_property.py --task={str(task)} --K={k} --num_layer={k} --wo_path_encoding --parallel"
        os.system(script)
        # KP-GNN without path encoding node tasks
        script = f"python train_node_property.py --task={str(task)} --K={k} --num_layer={k} --wo_path_encoding --parallel"
        os.system(script)
else:
    for parameter in grid:
        task, k = parameter
        # KP-GNN with path encoding graph tasks
        script = f"python train_graph_property.py --task={str(task)} --K={k} --num_layer={k}"
        os.system(script)
        # KP-GNN with path encoding node tasks
        script = f"python train_node_property.py --task={str(task)} --K={k} --num_layer={k}"
        os.system(script)
        # KP-GNN without path encoding graph tasks
        script = f"python train_graph_property.py --task={str(task)} --K={k} --num_layer={k} --wo_path_encoding"
        os.system(script)
        # KP-GNN without path encoding node tasks
        script = f"python train_node_property.py --task={str(task)} --K={k} --num_layer={k} --wo_path_encoding"
        os.system(script)
