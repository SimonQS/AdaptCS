## Repository Overview
We provide the PyTorch implementation for HCS framework here. The repository is organised as follows:

```python
|-- PyTorch # experiments on 10 small-scale datasets
    |-- experiment/ # experiment bash script 
    |-- models/ # model definition
    |-- splits/ # split files for datasets
    |-- arg_parser.py  # the argument parser code for training/hyperparameter searching script
    |-- logger.py # the logger code
    |-- train.py # the model trainig script
    |-- utils.py # data process and others
|-- data/ # 3 old datasets, including cora, citeseer, and pubmed
|-- new-data/ # 6 new datasets, including texas, wisconsin, cornell, actor, squirrel, and chameleon
    |-- feature_generation.py  # generate node features with base datasets
    |-- graph_generation.py  # generate graphs with different homophily levels
    |-- train.py # the training code
    |-- utils.py # data process and others
    |-- logger.py # logger
    |-- homophily.py # different homophily metrics
|-- plots # all experimental plots and visualizations in our paper
```

## Dependencies

The script has been tested running under Python 3.7.4, with the following packages installed (along with their dependencies):

- `dgl-cpu==0.4.3.post2`
- `dgl-gpu==0.4.3.post2`
- `ogb==1.3.1`
- `numpy==1.19.2`
- `scipy==1.4.1`
- `networkx==2.5`
- `torch==1.5.0`
- `torch-cluster==1.5.7`
- `torch-geometric==1.6.3`
- `torch-scatter==2.0.5`
- `torch-sparse==0.6.6`
- `torch-spline-conv==1.2.0`

```
pip install -r requirements.txt
```

## Running Experiments (PyTorch)

```
# training with default hyperparameters (e.g. Texas)
python train.py --dataset_name texas

# training with user defined hyperparameters
python train.py --dataset_name texas --lr 0.06 --weight_decay 0.0006 --dropout 0.6
```
The training/hyperparameter seacrhing logs are saved into the `logs/` folder located at `<your-working-directory>/HCS/PyTorch/logs`.


## Attribution
Parts of the code are based on
- [GCN](https://github.com/tkipf/pygcn)

- [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale)

- [ACM](https://github.com/SitaoLuan/ACM-GNN)


## License
MIT
