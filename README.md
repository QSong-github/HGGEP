# Gene Expression Prediction from Histology Images via Hypergraph Neural Networks [Briefings in Bioinformatics](https://academic.oup.com/bib/article/25/6/bbae500/7821151)
### 

Spatial transcriptomics reveals the spatial distribution of genes in complex tissues, providing crucial insights into biological processes, disease mechanisms, and drug development. The prediction of gene expression based on cost-effective histology images is a promising yet challenging field of research. Existing methods for gene prediction from histology images exhibit two major limitations. First, they ignore the intricate relationship between cell morphological information and gene expression. Second, these methods do not fully utilize the different latent stages of features extracted from the images. To address these limitations, this paper proposes a novel hypergraph neural network model, HGGEP, to predict gene expressions from histology images. HGGEP includes a gradient enhancement module to enhance the model’s perception of cell morphological information. A lightweight backbone network extracts multiple latent stage features from the image, followed by attention mechanisms to refine the representation of features at each latent stage and capture their relations with nearby features. To explore higher-order associations among multiple latent stage features, we stack them and feed into the hypergraph to establish associations among features at different scales. Experimental results on multiple datasets from disease samples including cancers and tumor disease, demonstrate the superior performance of our HGGEP model than existing methods.
       

![(Variational) gcn](Figures/workflow.png)

## Installation

Download HGGEP:
```
git clone https://github.com/QSong-github/HGGEP
```

## System environment
Required package:
- PyTorch >= 1.10
- pytorch-lightning >= 1.4
- scanpy >= 1.8
- python >= 3.7
- torch_geometric


# HGGEP pipeline

See [tutorial.ipynb](tutorial.ipynb)


NOTE: Run the following command if you want to run the script tutorial.ipynb
 
1.  Please run the script `download.sh` in the folder [data](https://github.com/biomed-AI/Hist2ST/tree/main/data) 

or 

Run the command line `git clone https://github.com/almaan/her2st.git` in the dir [data](https://github.com/biomed-AI/Hist2ST/tree/main/data) 

2. Run `gunzip *.gz` in the dir `HGGEP/data/her2st/data/ST-cnts/` to unzip the gz files


# Datasets

 -  human HER2-positive breast tumor ST data https://github.com/almaan/her2st/.
 -  human cutaneous squamous cell carcinoma 10x Visium data (GSE144240).

# Trained models

All Trained models of our method on HER2+ and cSCC datasets can be found at [synapse](https://www.synapse.org/Synapse:syn60239950/files/) 

# Train models

```
# go to /path/to/HGGEP
# for HER2+ dataset
python HGGEP_train.py --data "her2st"

# for cSCC dataset
python HGGEP_train.py --data "cscc"
```

# Test models

See [test_model.ipynb](test_model.ipynb)


# Reference
If you find this project is useful for your research, please cite:
```

@article{li2024gene,
  title={Gene expression prediction from histology images via hypergraph neural networks},
  author={Li, Bo and Zhang, Yong and Wang, Qing and Zhang, Chengyang and Li, Mengran and Wang, Guangyu and Song, Qianqian},
  journal={Briefings in Bioinformatics},
  volume={25},
  number={6},
  pages={bbae500},
  year={2024},
  publisher={Oxford University Press}
}

```

