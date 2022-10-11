import os
from itertools import product
kernels=["spd","gd"]
ks=[1,2,3,4]
grid=product(kernels,ks)
for parameter in grid:
    kernel,k=parameter
    script=f"python train_SR.py --K={k} --num_layer=2 --kernel={kernel}"
    os.system(script)