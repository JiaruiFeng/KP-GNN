# KP-GNN
Source code for [**How powerful are K-hop message passing graph neural networks**](https://arxiv.org/abs/2205.13328)

## Requirements
```
python=3.8
torch=1.11.0
PyG=2.0.4
OGB=1.3.4
```
## Usages
Switch between the shortest path distance kernel or graph diffusion kernel:
```
--kernel=spd
--kernel=gd
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
To run K-GIN, set:
```
--wo_peripheral_edge --wo_peripheral_configuration 
```
### Simulation datasets for validating expressive power
Run EXP with searching of different kernel and K:
```
python run_EXP_search.py
```
Run SR25 with searching of different kernel and K:
```
python run_SR_search.py
```
Run CSL with searching of different kernel and K:
```
python run_CSL_search.py
```
### Simulation dataset for node/graph properites and substructure
Run node/graph properites search:
```
python run_graph_node_property.py
```
Run substructure counting search:
```
python run_structure_counting.py
```
### Real world datasets
Run MUTAG dataset with 3-hop KP-GCN:
```
python train_TU.py --dataset_name=MUTAG --model_name=KPGCN --K=3 --kernel=spd
```
Run TU dataset search:
```
python run_TU_search.py
```
Run QM9 targets:
```
python run_qm9_search.py
```
Run ZINC dataset:
```
python train_ZINC.py
```
## Reference
If you find the code useful, please cite our paper:


