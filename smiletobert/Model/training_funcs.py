######################
### import modules ###
######################
import pandas as pd
import numpy as np
import sys
import torch
from transformers import BertTokenizer
import os
import codecs
from SmilesPE.tokenizer import SPE_Tokenizer
from torch.utils.data import Dataset, DataLoader
import tqdm
import re
from rdkit import Chem
import random
from sklearn import preprocessing
import time
from torch import nn

#sys.path.append(os.getcwd())
#from smiletobert.Model.BERT import BERT, SMILESLM

#from tdc.benchmark_group import admet_group
#from tdc.single_pred import ADME

class admetPrediction(torch.nn.Module):
    """
    General class to predict ADMET properties from contextualized vectors.
    
    Attributes:
        task_type (str): Type of task - 'classification' or 'regression'.
        linear1 (torch.nn.Linear): Fully connected layer mapping hidden to output size.
        linear2 (torch.nn.Linear): Intermediate fully connected layer.
        dropout (torch.nn.Dropout): Dropout layer to prevent overfitting.
        act (torch.nn.Sigmoid): Activation function for classification tasks.
        relu (torch.nn.ReLU): Activation function for hidden layers.
    """

    def __init__(self, input: int, hidden: int, output: int, task_type: str, dropout: float = 0.1):
        """
        Initializes the ADMET prediction model.

        Args:
            input (int): BERT model output size.
            hidden (int): Size of hidden layer.
            output (int): Prediction output size.
            task_type (str): Type of task - 'classification' or 'regression'.
            dropout (float, optional): Dropout probability. Default is 0.1.
        """
        super().__init__()
        self.task_type = task_type
        self.linear1 = torch.nn.Linear(hidden, output)
        self.linear2 = torch.nn.Linear(input, hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ADMET prediction model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted output tensor.
        """
        if self.task_type == 'classification':
            return self.act(self.linear1(self.dropout(self.relu(self.linear2(self.dropout(x))))))
        else:
            return self.linear1(self.dropout(self.relu(self.linear2(self.dropout(x)))))

class AdmetSMILES(torch.nn.Module):
    """
    Predicts ADMET properties using the Smile-to-Bert approach.

    Attributes:
        bert (torch.nn.Module): Pre-trained Smile-to-Bert model.
        admetPrediction (admetPrediction): ADMET property predictor using contextualized vectors.
        task_type (str): Type of task - 'classification' or 'regression'.
    """

    def __init__(self, bert_model: torch.nn.Module, hidden: int, task_type: str, output: int = 1, dropout: float = 0.1):
        """
        Initializes the AdmetSMILES model.

        Args:
            bert_model (torch.nn.Module): Pre-trained Smile-to-Bert model.
            task_type (str): Type of task - 'classification' or 'regression'.
            output (int, optional): Prediction output size. Default is 1.
            dropout (float, optional): Dropout probability. Default is 0.1.
        """
        super().__init__()
        self.bert = bert_model
        self.admetPrediction = admetPrediction(self.bert.d_model, hidden, output, task_type, dropout)
        self.task_type = task_type  # Ensure task_type is stored for later reference

    def forward(self, smiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the AdmetSMILES model.

        The model processes SMILES tokens, extracts contextualized vectors using a BERT-like encoder, 
        and predicts ADMET properties.

        Args:
            smiles (torch.Tensor): Input tensor of tokenized SMILES sequences. 
                                   Shape: [batch_size, seq_len]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Predicted ADMET value. Shape: [batch_size, output_dim]
                - Contextualized embeddings. Shape: [batch_size, d_model]
        """
        mask = (smiles > 0).unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
        embeddings = self.bert(smiles)  # Shape: [batch_size, seq_len, d_model]
        embeddings = embeddings * mask  # Zero out padding tokens

        sum_embeddings = embeddings.sum(dim=1)  # Sum along seq_len
        valid_tokens = mask.sum(dim=1)  # Count valid tokens along seq_len
        mean_embeddings = sum_embeddings / valid_tokens  # Compute mean embeddings

        return self.admetPrediction(mean_embeddings), mean_embeddings

class SMILESDatasetPE(Dataset):
    """
    Dataset class for ADMET TDC datasets and Smile-to-Bert.

    Attributes:
        data (pandas.core.frame.DataFrame): ADMET TDC dataset.
        seq_len (int): Maximum length of the input tensor.
        tokenizer (transformers.models.bert.tokenization_bert.BertTokenizer): BertTokenizer with SMILES 
            Pair Encoding vocabulary.
        spe (SmilesPE.tokenizer.SPE_Tokenizer): SMILES Pair Encoding tokenizer.
        mean (float): Mean of the ADMET property in training.
        sd (float): Standard deviation of the ADMET property in training.
        task_type (str): Type of task - 'classification' or 'regression'.
        change_smi (bool): If True, modifies SMILES strings using RDKit by setting a new root atom.
    """

    def __init__(self, data, tokenizer, spe, seq_len, mean, sd, task_type, change_smi):
        """
        Initializes the SMILESDatasetPE class.

        Args:
            data (pandas.core.frame.DataFrame): ADMET TDC dataset.
            seq_len (int): Maximum length of the input tensor.
            tokenizer (transformers.models.bert.tokenization_bert.BertTokenizer): BertTokenizer with SMILES 
                Pair Encoding vocabulary.
            spe (SmilesPE.tokenizer.SPE_Tokenizer): SMILES Pair Encoding tokenizer.
            mean (float): Mean of the ADMET property in training.
            sd (float): Standard deviation of the ADMET property in training.
            task_type (str): Type of task - 'classification' or 'regression'.
            change_smi (bool): If True, modifies SMILES strings using RDKit by setting a new root atom.
        """
        self.data = data
        self.seq_len = seq_len
        self.len_compounds = len(self.data)
        self.tokenizer = tokenizer
        self.spe = spe
        self.mean = mean
        self.sd = sd
        self.task_type = task_type
        self.change_smi = change_smi

    def __len__(self) -> int:
        """
        Returns:
            int: Length of the dataset.
        """
        return self.len_compounds

    def change_smiles(self, smi: str) -> str:
        """
        Randomly modifies the SMILES representation by setting a new root atom.

        Args:
            smi (str): Input SMILES string.

        Returns:
            str: SMILES string rooted at a randomly selected atom.
        """
        if smi.find('.') >= 0:
            return smi

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        num_atoms = mol.GetNumAtoms()
        pos_list = list(range(num_atoms))

        pos = random.choice(pos_list)
        new_smi = Chem.MolToSmiles(mol, rootedAtAtom=pos)
        return new_smi if len(new_smi) < self.seq_len else smi

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:   
        """
        Retrieves a sample from the dataset.

        It selects one SMILES string and its corresponding property, applies the `change_smiles()` function 
        if `change_smi` is True, tokenizes the SMILES, and standardizes the property value. 

        Args:
            item (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Input tokens vector
                - Z-score of the property value
        """
        smiles = self.data['Drug'].iloc[item]
        if self.change_smi:
            smiles = self.change_smiles(smiles)
        
        tokens = self.tokenizer.encode(self.spe.tokenize(smiles).split(' '))[0:-1][0:self.seq_len]
        padding = [self.tokenizer.encode('[PAD]')[1] for _ in range(self.seq_len - len(tokens))]
        tokens.extend(padding)
        tokens = torch.tensor(tokens)  
        label = torch.tensor(self.data['Y'].iloc[item]).float()
        
        if self.task_type == 'regression':
            label = (label - self.mean) / self.sd

        label = label.unsqueeze(0)
    
        return tokens, label







