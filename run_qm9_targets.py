import os
tasks=list(range(12))

for task in tasks:
    script=f"python train_qm9.py --task={str(task)} --virtual_node --use_rd"
    os.system(script)
