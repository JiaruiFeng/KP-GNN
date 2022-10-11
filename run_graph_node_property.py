import os
from itertools import product
tasks=[0,1,2]
ks=[3,4,5,6]
grid=product(tasks,ks)

for parameter in grid:
    task,k=parameter
    script=f"python train_graph_property.py --task={str(task)} --K={k} --num_layer={k}"
    os.system(script)
    script=f"python train_node_property.py --task={str(task)} --K={k} --num_layer={k}"
    os.system(script)
    script=f"python train_graph_property.py --task={str(task)} --K={k} --num_layer={k} --wo_path_encoding"
    os.system(script)
    script=f"python train_node_property.py --task={str(task)} --K={k} --num_layer={k} --wo_path_encoding"
    os.system(script)

