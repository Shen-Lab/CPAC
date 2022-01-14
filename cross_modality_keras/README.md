## Dependencies
You can create the environment by:
```
conda env create -n envname -f environment.yml
```

## Training Models of Different Modalities
```
cd ./training
mkdir ./weights
```

1D-sequence model:
```
python main_seq.py
```

2D-graph model:
```
python main_graph.py
```

Cross-modality concatenation:
```
python main_concat.py
```

Cross-modality cross interaction:
```
python main_crossInteraction.py
```

## Evaluation
```
python inference.py --model seq
python inference.py --model graph
python inference.py --model concat
python inference.py --model crossInteraction
```

## Acknowledgements

The backbone implementation is reference to https://github.com/Shen-Lab/DeepAffinity.
