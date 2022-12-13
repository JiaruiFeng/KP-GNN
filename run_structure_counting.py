import os
from itertools import product
import argparse

parser = argparse.ArgumentParser(f'add multi-gpu option')
parser.add_argument("--parallel", action="store_true",
                    help="If true, use DataParallel for multi-gpu training")
parallel_arg = parser.parse_args()
parallel = parallel_arg.parallel


tasks = [0, 1, 2, 3]
ks = [1, 2, 3, 4]
grid = product(ks, tasks)


if parallel:
    for parameter in grid:
        k, task = parameter
        #KPGIN-prime
        script = f"python train_structure_counting.py --task={str(task)} --K={k} --num_layer=2 --model_name=KPGINPrime --wo_path_encoding --parallel "
        os.system(script)
        #KGIN-prime
        script = f"python train_structure_counting.py --task={str(task)} --K={k} --num_layer=2 --model_name=KPGINPrime " \
                 f"--wo_peripheral_edge --wo_peripheral_configuration --wo_path_encoding --parallel "
        os.system(script)
else:
    for parameter in grid:
        k, task = parameter
        #KPGIN-prime
        script = f"python train_structure_counting.py --task={str(task)} --K={k} --num_layer=2 --model_name=KPGINPrime --wo_path_encoding"
        os.system(script)
        #KGIN-prime
        script = f"python train_structure_counting.py --task={str(task)} --K={k} --num_layer=2 --model_name=KPGINPrime " \
                 f"--wo_peripheral_edge --wo_peripheral_configuration --wo_path_encoding"
        os.system(script)
