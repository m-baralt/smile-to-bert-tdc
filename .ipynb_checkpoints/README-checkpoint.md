# smile-to-bert-tdc

Molecular property prediction is crucial for drug discovery. Over the years, deep learning models have been widely used for these tasks; however, large datasets are often needed to achieve strong performances. Pre-training models on vast unlabeled data has emerged as a method to extract contextualized embeddings that boost performance on smaller datasets. The Simplified Molecular Input Line Entry System (SMILES) encodes molecular structures as strings, making them suitable for natural language processing. Transformers, known for capturing long-range dependencies, are well suited for processing SMILES. One such transformer-based architecture is Bidirectional Encoder Representations from Transformers (BERT), which only uses the encoder part of the Transformer and performs classification and regression tasks. Pre-trained transformer-based architectures using SMILES have significantly improved predictions on smaller datasets. Public data repositories such as PubChem, which provide SMILES, among other data, are essential for pre-training these models. 
SMILES embeddings that combine chemical structure and physicochemical property information could further improve performance on tasks such as Absorption, Distribution, Metabolism, Excretion, and Toxicity prediction. To this end, we introduce Smile-to-Bert, a pre-trained BERT architecture designed to predict 113 RDKit-computed molecular descriptors from PubChem SMILES. This model generates embeddings that integrate both molecular structure and physicochemical properties. We evaluate Smile-to-Bert on 22 datasets from the Therapeutics Data Commons and compare its performance with that of the 2-encoder model and a Transformer model. Smile-to-Bert achieves the best result on one dataset, while the combination of Smile-to-Bert with the other models leads to improved performance on 8 datasets. Additionally, the state-of-the-art Transformer is applied to Absorption, Distribution, Metabolism, Excretion, and Toxicity prediction for the first time, achieving the best performance on the Therapeutics Data Commons leaderboard of one dataset.

This repository contains the code to evaluate Smile-to-Bert, the 2-encoder model, and the Transformer as individual models, as well as the combination of the 2-encoder and the Transformer with Smile-to-Bert across 22 datasets from the Therapeutics Data Commons (ADMET group).

## Installation

1. Install Python3 if required.
2. Git clone this repository:
```
git clone https://github.com/m-baralt/smile-to-bert-tdc.git
```
3. (Optional but recommended) Create a conda environment and activate it:
```
conda create --name tdc_smiletobert python=3.11.5
conda activate tdc_smiletobert
```
5. Install the required python libraries using:
```
python3 -m pip install -r requirements.txt
```

## TDC ADMET evaluation

In order to use the pre-trained models, the weights need to be downloaded using the `download_ckpt.sh` file. To download data using .sh files, the following commands need to be run:

```
cd path/to/repo
chmod +x download_ckpt.sh
./download_ckpt.sh
```

This should create a directory named ckpts with all the weights from the pre-trained models. 

Once this is done, the code to obtain the results shown in the paper can be run for each model. 

For Smile-to-Bert:

```
python3 smiletobert/admet_prediction_spe.py
```

For the Transformer:

```
python3 transformer/admet_prediction_transformer.py
```

For the 2-encoder model:

```
python3 model2encoder/admet_prediction_m2encoder.py
```

For the Tranformer - Smile-to-Bert combination:

```
python3 combined_models/admet_prediction_spe_transf.py
```

For the 2-encoder - Smile-to-Bert combination:

```
python3 combined_models/admet_prediction_spe_2enc.py
```


