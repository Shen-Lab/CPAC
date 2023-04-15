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

You can set or tune the hyper-parameters of regularization terms by adding, ```--l0 ${L0} --l1 ${L1} --l2 ${L2} --l3 ${L3}``` where ```${L0}, ${L1}, ${L2}``` are selected from {0.01, 0.001, 0.0001} and ```${L3}``` from {1, 10, 100, 1000, 10000, 100000}.

## Weights
Model weights can be downloaded at https://drive.google.com/file/d/1TOh9iUttHMlErL61QEens18FMVtBXZIi/view?usp=sharing.
