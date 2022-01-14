# Cross-Modality and Self-Supervised Pre-Trained Protein Embeddings for Compound-Protein Affinity and Contact Prediction

## Motivation
Computational structure-free method for compound-protein affinity and contact prediction (CPAC) is aimed at modeling the interaction relation between compound-protein pairs. However, the "structure-hungry" interaction outputs made by CPAC are learned from the lone structure-unaware 1D sequential inputs of proteins, plus the limited number of pairwise supervised labels of affinities and contacts, which seriously caps the more general applicability of CPAC models.

## Results
We for the first time adopt cross-modality learning in CPAC to introduce structure-awareness into protein embeddings.
Protein data here are treated as available in both modalities of 1D amino-acid sequences and 2D contact maps, with empirical comparison between single-modality models indicating the 1D or 2D modality
dominates each other in affinity and contact prediction, respectively due to different information utilization.
To capture and fuse the information from both modalities, two cross-modality schemes, concatenation and cross interaction are invented with state-of-the-art performance achieved.
Moreover, we leverage the promising self-supervised pre-training techniques for 1D sequences and 2D graphs on top of cross-modality models, to address supervision starvation via exploiting rich unpaired and unlabelled protein sequences.
We numerically assay the pre-training strategies and deduce that,
different strategies are supreme in their own "comfort zones":
competitive performances are separately acquired with sequence pre-training and joint modality pre-training for affinity and contact prediction, similar to the phenomenon occurring in single-modality models.

## Data
Please download the processed data from https://drive.google.com/file/d/1jvMHKpmg-iU8uqfJrmU-MQHSP_e46a-k/view?usp=sharing and https://drive.google.com/file/d/19g9jUt4BF_-MouUooqzs6iGIslWG-mmJ/view?usp=sharing, and extract them by:
```
unzip data.zip
unzip pretrain_data.zip
```

## Experiments
* Cross-modality protein embeddings [keras](https://github.com/Shen-Lab/CPAC/tree/main/cross_modality_keras) [pytorch](https://github.com/Shen-Lab/CPAC/tree/main/cross_modality_torch)
* [Pre-training with MLM and GraphComp](https://github.com/Shen-Lab/CPAC/tree/main/pretrain_torch)
* [Finetuning](https://github.com/Shen-Lab/CPAC/tree/main/finetune_torch)

## Citation

If you use this code for you research, please cite our paper.
```
TBD
```

