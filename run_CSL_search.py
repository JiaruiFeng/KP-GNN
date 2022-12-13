import os
import argparse
from itertools import product

parser = argparse.ArgumentParser(f'add multi-gpu option')
parser.add_argument("--parallel", action="store_true",
                    help="If true, use DataParallel for multi-gpu training")
parallel_arg = parser.parse_args()
parallel = parallel_arg.parallel

ks = [1, 2, 3, 4]
kernels = ["spd", "gd"]
grid = product(kernels, ks)

if parallel:
    for parameters in grid:
        kernel, k = parameters
        # KP-GNN
        script = f"python train_CSL.py --kernel={kernel} --K={k} --num_layer=2 --reprocess --wo_path_encoding --parallel"
        os.system(script)
        # K-hop GNN
        script = f"python train_CSL.py --kernel={kernel} --K={k} --num_layer=2 --wo_peripheral_edge --wo_peripheral_configuration --wo_path_encoding --reprocess  --parallel"
        os.system(script)
else:
    for parameters in grid:
        kernel, k = parameters
        # KP-GNN
        script = f"python train_CSL.py --kernel={kernel} --K={k} --num_layer=2 --reprocess --wo_path_encoding"
        os.system(script)
        # K-hop GNN
        script = f"python train_CSL.py --kernel={kernel} --K={k} --num_layer=2 --wo_peripheral_edge --wo_peripheral_configuration --wo_path_encoding --reprocess"
        os.system(script)


