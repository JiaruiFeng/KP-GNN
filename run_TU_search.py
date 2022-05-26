import os
from itertools import product


datasets=[ 'MUTAG', 'PROTEINS', 'PTC_MR', 'DD',"IMDBBINARY"]
models=["KPGCN","KPGIN","KPGraphSAGE"]
layers=[2,3,4]
Ks=[2,3,4]
combines=["attention","geometric"]
kernels=["gd","spd"]

t=product(datasets,models,layers,Ks,combines,kernels)

for parameters in t:
    dataset,model,layer,K,combine,kernel=parameters
    factor = K
    if factor == 3:
        hidden_size = 33
    else:
        hidden_size = 32

        script=f"python train_TU.py --dataset_name={dataset} --model_name={model} " \
               f"--K={str(K)} --num_layer={str(layer)} --combine={combine} --kernel={kernel} --hidden_size={str(hidden_size)}"
    os.system(script)

