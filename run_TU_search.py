import os
from itertools import product


datasets=["MUTAG","DD", "PROTEINS","PTC","IMDBBINARY"]
models=["KPGCN","KPGIN","KPGraphSAGE"]

grid=product(datasets,models)

for parameters in grid:
    dataset,model=parameters
    script=f"python train_TU.py --dataset_name={dataset} --model_name={model} --search"
    os.system(script)

