import os
import argparse

parser = argparse.ArgumentParser(f'add multi-gpu option')
parser.add_argument("--parallel", action="store_true",
                    help="If true, use DataParallel for multi-gpu training")
parallel_arg = parser.parse_args()
parallel = parallel_arg.parallel

tasks = list(range(12))

if parallel:
    for task in tasks:
        script = f"python train_qm9.py --task={str(task)} --virtual_node --use_rd --parallel"
        os.system(script)
else:
    for task in tasks:
        script = f"python train_qm9.py --task={str(task)} --virtual_node --use_rd"
        os.system(script)
