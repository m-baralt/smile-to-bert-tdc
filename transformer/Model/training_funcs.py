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

sys.path.append(os.getcwd())
from transformer.Model.Transformer import EXTRA_CHARS, Encoder
from smiletobert.Model.training_funcs import admetPrediction


def create_masks(src, trg=None, pad_idx=ord(EXTRA_CHARS['pad']), device=None):
    """
    This function creates a mask for an input tensor. It masks padding tokens. 

    Args:
        src (torch.Tensor): Input tensor of tokenized SMILES sequences. 
        pad_idx (int): Padding index.

    Returns:
        torch.Tensor: Mask for an input tensor. 
    """
    src_mask = (src != pad_idx).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != pad_idx).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size, device)
        np_mask.to(device)
        trg_mask = trg_mask & np_mask
        return src_mask, trg_mask
    return src_mask

class TransformerBert(nn.Module):
    """
    Predicts ADMET properties using the Transformer from Morris et al.

    Attributes:
        d_model (int): Embedding size.
        encoder (torch.nn.Module): Encoder part of the Transformer model.
        admetPrediction (torch.nn.Module): ADMET property predictor using contextualized vectors.
    """
    def __init__(self, alphabet_size, d_model, hidden, N, task_type, heads=8, dropout=0.1):
        """
        Initializes the TransformerBert model.

        Args:
            alphabet_size (int): Alphabet size. 
            d_model (int): Embedding size.
            hidden (int): Size of hidden layer.
            N (int): Number of encoder layers.
            task_type (str): Type of task - 'classification' or 'regression'.
            heads (int, optional): Number of attention heads. Default 8.
            dropout (float, optional): Dropout probability. Default is 0.1.
        """
        super().__init__()
        self.d_model = d_model
        self.encoder = Encoder(alphabet_size, d_model, N, heads, dropout)
        self.admetPrediction = admetPrediction(self.d_model, hidden, 1, task_type, dropout = 0.1)
        
    def forward(self, src, src_mask):
        """
        Forward pass of the TransformerBert model.

        The model processes SMILES tokens, extracts contextualized vectors using the encoder of a Transformer, 
        and predicts ADMET properties.

        Args:
            src (torch.Tensor): Input tensor of tokenized SMILES sequences. 
                                   Shape: [batch_size, seq_len]
            src_mask (torch.Tensor): Tensor with a mask for padding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Predicted ADMET value. Shape: [batch_size, output_dim]
        """
        e_outputs = self.encoder(src, src_mask)
        mask = src_mask.squeeze(1).unsqueeze(-1)  # Shape: (batch_size, seq_len, 1) 
        masked_e_outputs = e_outputs * mask  
        sum_embeddings = masked_e_outputs.sum(dim=1)  
        non_padding_tokens = mask.sum(dim=1) 
        e_outputs_mean = sum_embeddings / non_padding_tokens  
        output = self.admetPrediction(e_outputs_mean)
        
        return output

class SMILESDatasetT(Dataset):
    """
    Dataset class for ADMET TDC datasets and Transformer.

    Attributes:
        data (pandas.core.frame.DataFrame): ADMET TDC dataset.
        task_type (str): Type of task - 'classification' or 'regression'.
        mean (float): Mean of the ADMET property in training.
        sd (float): Standard deviation of the ADMET property in training.
        change_smi (bool): If True, modifies SMILES strings using RDKit by setting a new root atom.
    """
    def __init__(self, data, task_type, mean, sd, change_smi):
        """
        Initializes the SMILESDatasetT class.

        Args:
            data (pandas.core.frame.DataFrame): ADMET TDC dataset.
            task_type (str): Type of task - 'classification' or 'regression'.
            mean (float): Mean of the ADMET property in training.
            sd (float): Standard deviation of the ADMET property in training.
            change_smi (bool): If True, modifies SMILES strings using RDKit by setting a new root atom.
        """
        self.data = data
        self.len_compounds = len(self.data)
        self.task_type = task_type
        self.mean = mean
        self.sd = sd
        self.change_smi = change_smi

    def __len__(self):
        return self.len_compounds

    def encode_char(self, c): 
        return ord(c) - 32

    def encode_string_np(self, string, start_char=chr(0), pad_char=chr(0)): 
        if len(string) > 255:
            string = string[:255]
            
        arr = np.full((256,), ord(pad_char), dtype=np.float32)
        arr[:len(string)+1] = np.array([ord(start_char)] + [self.encode_char(c) for c in string])
        return arr

    def change_smiles(self, smi):
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
        if len(new_smi) < 256:
            return new_smi
        else:
            return smi

    def __getitem__(self, item):   
        """
        Retrieves a sample from the dataset.

        It selects one SMILES string and its corresponding property, 
        applies the `change_smiles()` function if `change_smi` is True, 
        tokenizes the SMILES, and standardizes the property value. 

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
            
        encoded = self.encode_string_np(smiles, start_char=EXTRA_CHARS['seq_start'], 
                                        pad_char=EXTRA_CHARS['pad'])
        encoded = encoded[:255]
        tokens = torch.tensor(encoded, dtype = torch.int)
        
        label = np.array(self.data['Y'].iloc[item])
        
        if self.task_type == 'regression':
            label = (label-self.mean)/self.sd

        label = torch.tensor(label).float()
        label = label.unsqueeze(0)
    
        return tokens, label





