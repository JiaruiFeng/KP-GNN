import os
from itertools import product


ks=[1,2,3,4]
kernels=["spd","gd"]
t=product(kernels,ks)
for parameters in t:
    kernel,k=parameters
    script=f"python train_EXP.py --kernel={kernel} --K={k} --num_layer=2 --reprocess"
    os.system(script)
    script=f"python train_EXP.py --kernel={kernel} --K={k} --num_layer=2 --wo_peripheral_edge --wo_peripheral_configuration --reprocess"
    os.system(script)
