import os
from itertools import product
tasks=[0,1,2,3]
ks=[1,2,3,4]
grid=product(ks,tasks)

for parameter in grid:
    k,task=parameter
    script=f"python train_structure_counting.py --task={str(task)} --K={k} --num_layer={k}"
    os.system(script)
    script=f"python train_structure_counting.py --task={str(task)} --K={k} --num_layer={k} --wo_path_encoding"
    os.system(script)




