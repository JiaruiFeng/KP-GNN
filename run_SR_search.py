import os
from itertools import product

kernels = ["spd", "gd"]
ks = [1, 2, 3, 4]
grid = product(kernels, ks)
for parameter in grid:
    kernel, k = parameter
    # KP-GNN
    script = f"python train_SR.py --K={k} --num_layer=2 --kernel={kernel} --wo_path_encoding"
    os.system(script)
    # K-hop GNN
    script = f"python train_SR.py --K={k} --num_layer=2 --kernel={kernel} --wo_path_encoding --wo_peripheral_edge --wo_peripheral_configuration"
    os.system(script)