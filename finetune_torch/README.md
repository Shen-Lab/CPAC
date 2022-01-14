## Dependencies
Please refer to https://github.com/Shen-Lab/CPAC/tree/main/cross_modality_torch#dependencies for environment.

## Finetuning for Cross-Modality Models:
```
mkdir ./weights
```

Finetuning for mask language modeling on concatenation models:
```
python main_concatenation_parallel.py --train 1 --ss_seq mask --ss_graph none --p_seq_mask 0.05
python main_concatenation_parallel.py --train 1 --ss_seq mask --ss_graph none --p_seq_mask 0.15
python main_concatenation_parallel.py --train 1 --ss_seq mask --ss_graph none --p_seq_mask 0.25
```

Finetuning for mask language modeling on cross interaction models:
```
python main_crossInteraction.py --train 1 --ss_seq mask --ss_graph none --p_seq_mask 0.05
python main_crossInteraction.py --train 1 --ss_seq mask --ss_graph none --p_seq_mask 0.15
python main_crossInteraction.py --train 1 --ss_seq mask --ss_graph none --p_seq_mask 0.25
```

Finetuning for graph completion on concatenation models:
```
python main_concatenation_parallel.py --train 1 --ss_seq none --ss_graph mask --p_graph_mask 0.05
python main_concatenation_parallel.py --train 1 --ss_seq none --ss_graph mask --p_graph_mask 0.15
python main_concatenation_parallel.py --train 1 --ss_seq none --ss_graph mask --p_graph_mask 0.25
```

Finetuning for graph completion on cross interaction models:
```
python main_crossInteraction.py --train 1 --ss_seq none --ss_graph mask --p_graph_mask 0.05
python main_crossInteraction.py --train 1 --ss_seq none --ss_graph mask --p_graph_mask 0.15
python main_crossInteraction.py --train 1 --ss_seq none --ss_graph mask --p_graph_mask 0.25
```

Finetuning for joint pre-training on concatenation models:
```
python main_concatenation_parallel.py --train 1 --ss_seq mask --ss_graph mask --p_seq_mask 0.05--p_graph_mask 0.05
python main_concatenation_parallel.py --train 1 --ss_seq mask --ss_graph mask --p_seq_mask 0.15--p_graph_mask 0.15
python main_concatenation_parallel.py --train 1 --ss_seq mask --ss_graph mask --p_seq_mask 0.25--p_graph_mask 0.25
```

Finetuning for joint pre-training on cross interaction models:
```
python main_crossInteraction.py --train 1 --ss_seq mask --ss_graph mask --p_seq_mask 0.05--p_graph_mask 0.05
python main_crossInteraction.py --train 1 --ss_seq mask --ss_graph mask --p_seq_mask 0.15--p_graph_mask 0.15
python main_crossInteraction.py --train 1 --ss_seq mask --ss_graph mask --p_seq_mask 0.25--p_graph_mask 0.25
```

You can set or tune the hyper-parameters of regularization terms by adding, ```--l0 ${L0} --l1 ${L1} --l2 ${L2} --l3 ${L3}``` where ```${L0}, ${L1}, ${L2}``` are selected from {0.01, 0.001, 0.0001} and ```${L3}``` from {1, 10, 100, 1000, 10000, 100000}.

