import os
from itertools import  product

tasks=[0,1,2,3]
for task in tasks:
    script=f"python train_structure_counting.py --task={str(task)}"
    os.system(script)