# How powerful are K-hop message passing graph neural networks
This repository is the official implementation of the model in the [**How powerful are K-hop message passing graph neural networks**](https://openreview.net/forum?id=nN3aVRQsxGd&noteId=TBGwgubYuA6)
## News
In version 3.0 we add support for multi-gpu training with DataParallel in torch. Code will auto-detect gpu devices in your machine. To use multi-gpu training, simply add `--parallel`. For example:
```
python train_CSL.py --parallel
```
## Requirements
```
python=3.8
torch=1.11.0
PyG=2.1.0
OGB=1.3.4
```
## Usages
Switch between the shortest path distance kernel and graph diffusion kernel:
```
--kernel=spd
--kernel=gd
```
Use different number of hop:
```
--K=6
```
To run KP-GNN, set:
```
--model_name=KPGIN
--model_name=KPGraphSAGE
--model_name=KPGCN
```
To run KP-GIN+, set:
```
--model_name=KPGINPlus
```
To run normal K-hop GNN, set:
```
--wo_peripheral_edge --wo_peripheral_configuration 
```
For more details about these models and parameters, please check our paper.

### Simulation datasets for validating expressive power
EXP dataset:
```
# run single model
python train_EXP.py
# search for different K and model
python run_EXP_search.py
# multi-gpu
python run_EXP_search.py --parallel
```
SR25 dataset:
```
# run single model
python train_SR.py
# search for different K and model
python run_SR_search.py
# multi-gpu
python run_SR_search.py --parallel
```
CSL dataset:
```
# run single model
python train_CSL.py

# search for different K and model
python run_CSL_search.py
# multi-gpu
python run_CSL_search.py --parallel
```
### Simulation dataset for node/graph properties and substructure
Node/graph properties:
```
# single task
python train_graph_property.py --task=0
python train_node_property.py --task=0
# run all tasks with search
python run_graph_node_property.py
# multi-gpu
python run_graph_node_property.py --parallel
```
Substructure counting:
```
# single task
python train_structure_counting.py --task=0
# run all tasks with search
python run_structure_counting.py
#multi-gpu
python run_structure_counting.py --parallel
```
### Real world datasets
Run MUTAG dataset with 3-hop KP-GCN:
```
python train_TU.py --dataset_name=MUTAG --model_name=KPGCN --K=3 --kernel=spd
```
Run TU dataset search:
```
python run_TU_search.py
#multi-gpu
python run_TU_search.py --parallel
```
Run QM9 targets:
```
# single target
python train_qm9.py --task=7
#all targets
python run_qm9_search.py
#multi-gpu
python run_qm9_search.py --parallel
```
Run ZINC dataset:
```
python train_ZINC.py --residual
```
## Reference
If you find the code useful, please cite our paper:
```
@inproceedings{
feng2022how,
title={How Powerful are K-hop Message Passing Graph Neural Networks},
author={Jiarui Feng and Yixin Chen and Fuhai Li and Anindya Sarkar and Muhan Zhang},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=nN3aVRQsxGd}
}
```

