## Dependencies
Please refer to https://github.com/Shen-Lab/CPAC/tree/main/cross_modality_torch#dependencies for environment.

## Pre-training Strategies for Cross-Modality Models:
```
mkdir ./weights
```

Mask language modeling pre-training for 1D encoder:
```
python main_pretrain.py --ss_seq mask --ss_graph none --p_seq_mask ${MASK_RETIO}
```
where ```${MASK_RETIO}``` is selected from {0.05, 0.15, 0.25}.

Graph completion pre-training for 2D encoder:
```
python main_pretrain.py --ss_seq none --ss_graph mask --p_graph_mask ${MASK_RETIO}
```
where ```${MASK_RETIO}``` is selected from {0.05, 0.15, 0.25}.

Joint pre-training for cross-modality models:
```
python main_pretrain.py --ss_seq mask --ss_graph none --p_seq_mask ${MASK_RETIO} --p_graph_mask ${MASK_RETIO}
```

## Acknowledgements

The graph completion implementation is reference to https://github.com/Shen-Lab/SS-GCNs.

