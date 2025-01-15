## DKS: A Unified GNN-Based Method for Keyword Search on Dirty Graphs

A PyTorch + DGL implementation of DKS, as described in the paper: A Unified GNN-Based Framework for Keyword Search on Dirty Graphs

### Requirements
```
- cuda11.3
- cudnn=8.2.0
- python=3.7.16
- torch=1.12.0+cu113
- dgl=1.1.1.cu113
-torch-geometric=2.3.1
-torch-cluster=1.6.1+pt112cu113
-torch-scatter=2.1.1+pt112cu113
-torch-sparse=0.6.17+pt112cu113
```
### Running
```
python main.py
```

### Project Structure
```
├  dan_net.py      # DAN net
│  gat_layer.py      # GAT layer
│  gat_net.py      # GAT net
│  gat_params.json   # the parameters of GAT
│  gat_trainer.py    # GAT train and evaluate 
│  gat_utils.py
│  main.py              # project extrance, parameters settings
│  mlp_layer.py
│  README.md
│  utils.py          # other functions
│  
└─data      
    ├─citeseer       # dataset
    │  │  edge_index.npz
    │  │  spX.npz
    │  │  
    │  ├─contaminated_graph
    │  │      30%contaminate_nodes.out
    │  │      30%contaminate_spX.npz
    │  │      50%contaminate_nodes.out
    │  │      50%contaminate_spX.npz
    │  │      70%contaminate_nodes.out
    │  │      70%contaminate_spX.npz
    │  │      
    │  ├─incomplete_graph
    │  │      30%incomplete_nodes.out
    │  │      30%incomplete_spX.npz
    │  │      50%incomplete_nodes.out
    │  │      50%incomplete_spX.npz
    │  │      70%incomplete_nodes.out
    │  │      70%incomplete_spX.npz
    │  │      edge_10_edge_index.npz
    │  │      edge_20_edge_index.npz
    │  │      edge_30_edge_index.npz
    │  │      edge_40_edge_index.npz
    │  │      edge_50_edge_index.npz
    │  │      
    │  └─test
    │      ├─kw3
    │      │      kw3_top100_output.txt
    │      │      kw3_top10_output.txt
    │      │      kw3_top50_output.txt
    │      │      
    │      ├─kw5
    │      │      kw5_top100_output.txt
    │      │      kw5_top10_output.txt
    │      │      kw5_top50_output.txt
    │      │      
    │      ├─kw7
    │      │      kw7_top100_output.txt
    │      │      kw7_top10_output.txt
    │      │      kw7_top50_output.txt
    │      │      
    │      └─kw9
    │              kw9_top100_output.txt
    │              kw9_top10_output.txt
    │              kw9_top50_output.txt
```

