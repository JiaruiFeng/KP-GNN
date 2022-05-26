import os

tasks=[0,1,2]

for task in tasks:
    script=f"python train_graph_property.py --task={str(task)}"
    os.system(script)
    script=f"python train_node_property.py --task={str(task)}"
    os.system(script)

