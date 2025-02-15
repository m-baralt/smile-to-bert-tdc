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
from transformer.Model.Transformer import EXTRA_CHARS
from smiletobert.Model.training_funcs import admetPrediction

class CombinedTransfBert(torch.nn.Module):
    """
    Predicts ADMET properties using a combination of embeddings created by the 
    Transformer from Morris et al. and Smile-to-Bert.

    Attributes:
        bert (torch.nn.Module): Smile-to-Bert model.
        transformer (torch.nn.Module): Transformer model.
        admetPrediction (torch.nn.Module): ADMET property predictor using contextualized vectors.
    """
    def __init__(self, hidden, bert_model, transformer, task_type, output = 1, dropout = 0.1):
        """
        Initializes the CombinedTransfBert model.

        Args:
            bert (torch.nn.Module): Smile-to-Bert model.
            transformer (torch.nn.Module): Transformer model.
            task_type (str): Type of task - 'classification' or 'regression'.
            output (int, optional): Output size. Default 1.
            dropout (float, optional): Dropout probability. Default is 0.1.
        """
        super().__init__()
        self.bert = bert_model
        self.transformer = transformer
        self.hidden = hidden
        self.admetPrediction = admetPrediction(self.bert.d_model+self.transformer.module.d_model, 
                                               hidden, output, task_type, dropout)

    def forward(self, xbert, xtransf, src_mask):
        """
        Forward pass of the CombinedTransfBert model.

        This function computes embeddings from a SMILES string for the Smile-to-Bert model and
        the Transformer. It combines the embeddings and predicts ADMET properties.

        Args:
            xbert (torch.Tensor): Input tensor of tokenized SMILES sequences for Smile-to-Bert.
            xtransf (torch.Tensor): Input tensor of tokenized SMILES sequences for the Transformer.
            src_mask (torch.Tensor): Tensor with a mask for padding for the Transformer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Predicted ADMET value. Shape: [batch_size, output_dim]
                - Combined embeddings.
        """

        # smile-to-bert embeddings
        mask_bert = (xbert > 0).unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
        embeddings_bert = self.bert(xbert)  # Shape: [batch_size, seq_len, d_model]
        embeddings_bert = embeddings_bert * mask_bert
        sum_embeddings_bert = embeddings_bert.sum(dim=1)  # Sum along seq_len
        valid_tokens = mask_bert.sum(dim=1)          # Count valid tokens along seq_len
        mean_embeddings_bert = sum_embeddings_bert/valid_tokens
        # transformer embeddings
        e_outputs = self.transformer(xtransf, src_mask)
        mask_transf = src_mask.squeeze(1).unsqueeze(-1)  # Shape: (batch_size, seq_len, 1) 
        masked_e_outputs = e_outputs * mask_transf  
        sum_embeddings_transf = masked_e_outputs.sum(dim=1)  
        non_padding_tokens = mask_transf.sum(dim=1) 
        mean_embeddings_transf = sum_embeddings_transf/non_padding_tokens  
        # embeddings combination
        combined_embeddings = torch.cat((mean_embeddings_transf, mean_embeddings_bert), dim=1) 
        # [batch_size, bert.d_model+d_model]
        
        output = self.admetPrediction(combined_embeddings)

        return output, combined_embeddings

class SMILESDatasetComb(Dataset):
    """
    Dataset class for ADMET TDC datasets for Smile-to-Bert and Transformer.

    Attributes:
        data (pandas.core.frame.DataFrame): ADMET TDC dataset.
        seq_len (int): Maximum length for the Smile-to-Bert token sequence.
        tokenizer (transformers.models.bert.tokenization_bert.BertTokenizer): BertTokenizer with SMILES 
            Pair Encoding vocabulary.
        spe (SmilesPE.tokenizer.SPE_Tokenizer): SMILES Pair Encoding tokenizer.
        mean (float): Mean of the ADMET property in training.
        sd (float): Standard deviation of the ADMET property in training.
        task_type (str): Type of task - 'classification' or 'regression'.
        change_smi (bool): If True, modifies SMILES strings using RDKit by setting a new root atom.
    """
    def __init__(self, data, tokenizer, spe, seq_len, mean, sd, task_type, change_smi = True):
        """
        Initializes the SMILESDatasetComb class.

        Args:
            data (pandas.core.frame.DataFrame): ADMET TDC dataset.
            tokenizer (transformers.models.bert.tokenization_bert.BertTokenizer): BertTokenizer 
            with SMILES Pair Encoding vocabulary.
            spe (SmilesPE.tokenizer.SPE_Tokenizer): SMILES Pair Encoding tokenizer.
            seq_len (int): Maximum length for the Smile-to-Bert token sequence.
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

    def __len__(self):
        return self.len_compounds

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
        if len(new_smi) < self.seq_len:
            return new_smi
        else:
            return smi

    def encode_char(self, c): 
        return ord(c) - 32

    def encode_string_np(self, string, start_char=chr(0), pad_char=chr(0)): 
        if len(string) > 255:
            string = string[:255]
            
        arr = np.full((256,), ord(pad_char), dtype=np.float32)
        arr[:len(string)+1] = np.array([ord(start_char)] + [self.encode_char(c) for c in string])
        return arr

    def __getitem__(self, item):  
        """
        Retrieves a sample from the dataset.

        It selects one SMILES string and its corresponding property, 
        applies the `change_smiles()` function if `change_smi` is True, 
        tokenizes the SMILES for Smile-to-Bert and for the Transformer, 
        and standardizes the property value. 

        Args:
            item (int): Index of the sample.

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: 
                - Input tokens sequence for Smile-to-Bert
                - Input tokens sequence for the Transformer
                - Z-score of the property value
        """
        smiles = self.data['Drug'].iloc[item]
        if self.change_smi:
            smiles = self.change_smiles(smiles)
        # bert smilesPE
        tokens = self.tokenizer.encode(self.spe.tokenize(smiles).split(' '))[0:-1][0:self.seq_len]
        padding = [self.tokenizer.encode('[PAD]')[1] for _ in range(self.seq_len - len(tokens))]
        tokens.extend(padding)
        tokens_bert = torch.tensor(tokens)  
        # transformer
        encoded = self.encode_string_np(smiles, start_char=EXTRA_CHARS['seq_start'],
                                        pad_char=EXTRA_CHARS['pad'])
        encoded = encoded[:255]
        tokens_transf = torch.tensor(encoded, dtype = torch.int)

        
        label = torch.tensor(self.data['Y'].iloc[item]).float()
        
        if self.task_type == 'regression':
            label = (label-self.mean)/self.sd

        label = label.unsqueeze(0)
    
        return [tokens_bert, tokens_transf], label





