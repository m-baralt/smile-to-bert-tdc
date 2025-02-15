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

from smiletobert.Model.training_funcs import admetPrediction

class Combined2encBert(torch.nn.Module):
    """
    Predicts ADMET properties using a combination of embeddings created by the 
    2-encoder model and Smile-to-Bert.

    Attributes:
        bert (torch.nn.Module): Smile-to-Bert model.
        model2enc (torch.nn.Module): 2-encoder model.
        admetPrediction (torch.nn.Module): ADMET property predictor using contextualized vectors.
    """
    def __init__(self, hidden, bert_model, model2enc, task_type, output = 1, dropout = 0.1):
        """
        Initializes the Combined2encBert model.

        Args:
            bert (torch.nn.Module): Smile-to-Bert model.
            model2enc (torch.nn.Module): 2-encoder model.
            task_type (str): Type of task - 'classification' or 'regression'.
            output (int, optional): Output size. Default 1.
            dropout (float, optional): Dropout probability. Default is 0.1.
        """
        super().__init__()
        self.bert = bert_model
        self.model2enc = model2enc
        self.admetPrediction = admetPrediction(input = self.bert.d_model+self.model2enc.d_model, 
                                               hidden = hidden, output = output,
                                               task_type = task_type, dropout = dropout)

    def forward(self, xbert, x2enc):
        """
        Forward pass of the Combined2encBert model.

        This function computes embeddings from a SMILES string for the Smile-to-Bert model and
        the 2-encoder model. It combines the embeddings and predicts ADMET properties.

        Args:
            xbert (torch.Tensor): Input tensor of tokenized SMILES sequences for Smile-to-Bert.
            x2enc (torch.Tensor): Input tensor of tokenized SMILES sequences for the 2-encoder model.

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
        # 2encoder embeddings
        pred, embeddings_2enc = self.model2enc(x2enc)
         
        # embeddings combination
        combined_embeddings = torch.cat((embeddings_2enc, mean_embeddings_bert), dim=1) 
        # [batch_size, bert.d_model+d_model]
        
        output = self.admetPrediction(combined_embeddings)

        return output, combined_embeddings

class SMILESDatasetComb_2enc(Dataset):
    """
    Dataset class for ADMET TDC datasets for Smile-to-Bert and the 2-encoder model.

    Attributes:
        data (pandas.core.frame.DataFrame): ADMET TDC dataset.
        seq_len (int): Maximum length for the input token sequence.
        tokenizer (transformers.models.bert.tokenization_bert.BertTokenizer): BertTokenizer with SMILES 
            Pair Encoding vocabulary.
        spe (SmilesPE.tokenizer.SPE_Tokenizer): SMILES Pair Encoding tokenizer.
        mean (float): Mean of the ADMET property in training.
        sd (float): Standard deviation of the ADMET property in training.
        task_type (str): Type of task - 'classification' or 'regression'.
        change_smi (bool): If True, modifies SMILES strings using RDKit by setting a new root atom.
        symbol_ID (dict): Vocabulary of SMILES symbols.
        max_length_symbol (int, optional): Maximum length of SMILES symbols. Default 5. 
        BOS_ID (int, optional): Begin of sentence token identifier. Default 2. 
        EOS_ID (int, optional): End of sentence token identifier. Default 3. 
        PAD_ID (int, optional): Padding token identifier. Default 0. 
    """
    def __init__(self, data, tokenizer, spe, seq_len, symbol_ID, mean, sd, task_type, 
                 max_length_symbol = 5, change_smi = True, BOS_ID = 2, EOS_ID = 3, PAD_ID = 0):
        """
        Initializes the SMILESDatasetComb_2enc class.

        Args:
            data (pandas.core.frame.DataFrame): ADMET TDC dataset.
            tokenizer (transformers.models.bert.tokenization_bert.BertTokenizer): BertTokenizer 
            with SMILES Pair Encoding vocabulary.
            spe (SmilesPE.tokenizer.SPE_Tokenizer): SMILES Pair Encoding tokenizer.
            seq_len (int): Maximum length for the input token sequence.
            symbol_ID (dict): Vocabulary of SMILES symbols.
            mean (float): Mean of the ADMET property in training.
            sd (float): Standard deviation of the ADMET property in training.
            task_type (str): Type of task - 'classification' or 'regression'.
            change_smi (bool): If True, modifies SMILES strings using RDKit by setting a new root atom.
            max_length_symbol (int, optional): Maximum length of SMILES symbols. Default 5. 
            BOS_ID (int, optional): Begin of sentence token identifier. Default 2. 
            EOS_ID (int, optional): End of sentence token identifier. Default 3. 
            PAD_ID (int, optional): Padding token identifier. Default 0. 
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
        self.symbol_ID = symbol_ID
        self.max_length_symbol = max_length_symbol
        self.BOS_ID = BOS_ID
        self.EOS_ID = EOS_ID
        self.PAD_ID = PAD_ID

    def __len__(self):
        return self.len_compounds

    def smiles_str2smiles(self, smi):
        """
        Translates SMILES string to a list of symbol identifiers.

        Args:
            smi (str): Input SMILES string.

        Returns:
            list: List of symbol identifiers.
        """
        smiles = []
        i=0
        while i < len(smi):
            NotFindID = True
            for j in range(self.max_length_symbol, 0, -1):
              if i+j <= len(smi) and smi[i:i+j] in self.symbol_ID: # 
                smiles.append(self.symbol_ID[smi[i:i+j]])
                i += j-1 
                NotFindID = False
                break
            if NotFindID:
              smiles.append(4) # using MSK_ID to replace unknown symbol
            i += 1
        return smiles

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

    def __getitem__(self, item):   
        """
        Retrieves a sample from the dataset.

        It selects one SMILES string and its corresponding property, 
        applies the `change_smiles()` function if `change_smi` is True, 
        tokenizes the SMILES for Smile-to-Bert and for the 2-encoder model, 
        and standardizes the property value. 

        Args:
            item (int): Index of the sample.

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: 
                - Input tokens sequence for Smile-to-Bert.
                - Input tokens sequence for the 2-encoder model.
                - Z-score of the property value.
        """
        smiles = self.data['Drug'].iloc[item]
        if self.change_smi:
            smiles = self.change_smiles(smiles)
        # bert smilesPE
        tokens = self.tokenizer.encode(self.spe.tokenize(smiles).split(' '))[0:-1][0:self.seq_len]
        padding = [self.tokenizer.encode('[PAD]')[1] for _ in range(self.seq_len - len(tokens))]
        tokens.extend(padding)
        tokens_bert = torch.tensor(tokens)  
        

        # 2encoder
        tokens2enc = self.smiles_str2smiles(smiles)
        tokens2enc = [self.BOS_ID]+tokens2enc+[self.EOS_ID]
        tokens2enc = tokens2enc[0:self.seq_len]
        padding2enc = [self.PAD_ID for _ in range(self.seq_len - len(tokens2enc))]
        tokens2enc.extend(padding2enc)
        tokens2enc = torch.tensor(tokens2enc)  

        
        label = torch.tensor(self.data['Y'].iloc[item]).float()    
        if self.task_type == 'regression':
            label = (label-self.mean)/self.sd
        label = label.unsqueeze(0)
    
        return [tokens_bert, tokens2enc], label







