# How powerful are K-hop message passing graph neural networks
This repository is the official implementation of the model in the [**How powerful are K-hop message passing graph neural networks**](https://openreview.net/forum?id=nN3aVRQsxGd&noteId=TBGwgubYuA6)
## News
In version 4.0, we:
* add a new model variant called KP-GIN', which only runs KP-GIN at the first layer and uses normal GIN at the rest of the model. This model hugely reduces the variance of normal K-hop GNN but still achieves great results in real-world datasets.
* Fix the loss computation bug in **counting substructure dataset**. 
* Fix minor bugs.
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
Use different numbers of hop:
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
To run KP-GIN', set:
```
--model_name=KPGINPrime
```
To run normal K-hop GNN, set:
```
--wo_peripheral_edge --wo_peripheral_configuration 
```
Parallel training:
```
python train_CSL.py --parallel
```
For more details about these models and parameters, please check our paper.

### Simulation datasets for validating expressive power
Simulation of regular graph:
```
# node level
python run_simulation.py --n 20 40 80 160 320 640 1280 --save_appendix=_node --N=10
# graph level
python run_simulation.py --n 20 40 80 160 320 640 1280 --save_appendix=_graph --N=100 --graph
```
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
# run a single model
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
# run all tasks with a search
python run_graph_node_property.py
# multi-gpu
python run_graph_node_property.py --parallel
```
Substructure counting:
```
# single task
python train_structure_counting.py --task=0
# run all tasks with a search
python run_structure_counting.py
#multi-gpu
python run_structure_counting.py --parallel
```
### Real-world datasets
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
python train_ZINC.py --residual --K=8 --model_name=KPGINPlus --num_layer=8 --hidden_size=104
python train_ZINC.py --K=16 --num_layer=17 --hidden_size=96 --residual --model_name=KPGINPrime
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

