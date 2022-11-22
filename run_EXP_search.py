import os
from itertools import product
import argparse

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
        script = f"python train_EXP.py --kernel={kernel} --K={k} --num_layer=2 --reprocess --parallel"
        os.system(script)
        script = f"python train_EXP.py --kernel={kernel} --K={k} --num_layer=2 --wo_peripheral_edge --wo_peripheral_configuration --reprocess --parallel"
        os.system(script)
else:
    for parameters in grid:
        kernel, k = parameters
        script = f"python train_EXP.py --kernel={kernel} --K={k} --num_layer=2 --reprocess"
        os.system(script)
        script = f"python train_EXP.py --kernel={kernel} --K={k} --num_layer=2 --wo_peripheral_edge --wo_peripheral_configuration --reprocess"
        os.system(script)
