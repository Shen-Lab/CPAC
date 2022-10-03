# Cross-Modality and Self-Supervised Protein Embedding for Compound-Protein Affinity and Contact Prediction

## Motivation
Computational methods for compound-protein affinity and contact (CPAC) prediction aim at facilitating rational drug discovery by simultaneous prediction of the strength and the pattern of compound-protein interactions. Although the desired outputs are highly structure-dependent, the lack of protein   structures often force structure-free methods to rely on protein sequence inputs alone.  The scarcity of compound-protein pairs with affinity and contact labels further limits the accuracy and the generalizability of CPAC models.

## Results
To overcome the aforementioned challenges of structure naivety and labeled-data scarcity, we, for the first time, introduce cross-modality and self-supervised learning, respectively, for structure-aware and task-relevant protein embedding.  Specifically, protein data are  available in both modalities of 1D amino-acid sequences and predicted 2D contact maps, that are separately embedded with recurrent and graph neural networks, respectively, as well as jointly embedded with two cross-modality schemes.  Furthermore, both protein modalities are pretrained under various self-supervised learning strategies, by leveraging massive amount of unlabeled protein data.  Our results indicate that individual protein modalities differ in their strengths of predicting affinities or contacts.  Proper cross-modality protein embedding combined with self-supervised learning improves model  generalizability when predicting both affinities and contacts for unseen proteins. 

## Data
Please download the processed data from https://drive.google.com/file/d/1jvMHKpmg-iU8uqfJrmU-MQHSP_e46a-k/view?usp=sharing and https://drive.google.com/file/d/19g9jUt4BF_-MouUooqzs6iGIslWG-mmJ/view?usp=sharing, and extract them by:
```
unzip data.zip
unzip pretrain_data.zip
```

## Experiments
* Cross-modality protein embeddings [[keras]](https://github.com/Shen-Lab/CPAC/tree/main/cross_modality_keras) [[pytorch]](https://github.com/Shen-Lab/CPAC/tree/main/cross_modality_torch)
* [Pre-training with MLM and GraphComp](https://github.com/Shen-Lab/CPAC/tree/main/pretrain_torch)
* [Finetuning](https://github.com/Shen-Lab/CPAC/tree/main/finetune_torch)

## Citation

If you use this code for you research, please cite our paper.
```
@article{10.1093/bioinformatics/btac470,
    author = {You, Yuning and Shen, Yang},
    title = "{Cross-modality and self-supervised protein embedding for compound–protein affinity and contact prediction}",
    journal = {Bioinformatics},
    volume = {38},
    number = {Supplement_2},
    pages = {ii68-ii74},
    year = {2022},
    month = {09},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac470},
    url = {https://doi.org/10.1093/bioinformatics/btac470},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/38/Supplement\_2/ii68/45884189/btac470.pdf},
}
```

