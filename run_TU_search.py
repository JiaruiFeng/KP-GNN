import os
from itertools import product
import argparse

parser = argparse.ArgumentParser(f'add multi-gpu option')
parser.add_argument("--parallel", action="store_true",
                    help="If true, use DataParallel for multi-gpu training")
parallel_arg = parser.parse_args()
parallel = parallel_arg.parallel

datasets = ["MUTAG", "DD", "PROTEINS", "PTC", "IMDBBINARY"]
models = ["KPGCN", "KPGIN", "KPGraphSAGE"]
grid = product(datasets, models)

if parallel:
    for parameters in grid:
        dataset, model = parameters
        script = f"python train_TU.py --dataset_name={dataset} --model_name={model} --search --parallel"
        os.system(script)
else:
    for parameters in grid:
        dataset, model = parameters
        script = f"python train_TU.py --dataset_name={dataset} --model_name={model} --search"
        os.system(script)
