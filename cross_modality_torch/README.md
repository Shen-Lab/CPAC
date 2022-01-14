## Dependencies
You can create the environment by:
```
conda env create -n envname -f environment.yml
```

## Training and evaluate Models of Different Modalities
```
mkdir ./weights
```

Cross-modality concatenation:
```
python main_concatenation_parallel.py
```

Cross-modality cross interaction:
```
python main_crossInteraction_parallel.py
```

You can set or tune the hyper-parameters of regularization terms by adding, e.g. ```--l0 0.01 --l1 0.01 --l2 0.01 --l3 1000 ```.

