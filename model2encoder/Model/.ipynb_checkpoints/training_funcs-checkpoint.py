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

class SMILESDataset2enc(Dataset):
    """
    Dataset class for ADMET TDC datasets and the 2-encoder model.

    Attributes:
        data (pandas.core.frame.DataFrame): ADMET TDC dataset.
        seq_len (int): Maximum length of SMILES tokens tensor.
        task_type (str): Type of task - 'classification' or 'regression'.
        symbol_ID (dict): Vocabulary of SMILES symbols.
        mean (float): Mean of the ADMET property in training.
        sd (float): Standard deviation of the ADMET property in training.
        max_length_symbol (int, optional): Maximum length of SMILES symbols. Default 5. 
        change_smi (bool): If True, modifies SMILES strings using RDKit by setting a new root atom.
        BOS_ID (int, optional): Begin of sentence token identifier. Default 2. 
        EOS_ID (int, optional): End of sentence token identifier. Default 3. 
        PAD_ID (int, optional): Padding token identifier. Default 0. 
    """
    def __init__(self, data, seq_len, task_type, symbol_ID, mean, sd,
                 max_length_symbol = 5, change_smi = True, BOS_ID = 2, 
                 EOS_ID = 3, PAD_ID = 0):
        """
        Initializes the SMILESDataset2enc class.

        Args:
            data (pandas.core.frame.DataFrame): ADMET TDC dataset.
            seq_len (int): Maximum length of SMILES tokens tensor.
            task_type (str): Type of task - 'classification' or 'regression'.
            symbol_ID (dict): Vocabulary of SMILES symbols.
            mean (float): Mean of the ADMET property in training.
            sd (float): Standard deviation of the ADMET property in training.
            max_length_symbol (int, optional): Maximum length of SMILES symbols. Default 5. 
            change_smi (bool): If True, modifies SMILES strings using RDKit by setting a new root atom.
            BOS_ID (int, optional): Begin of sentence token identifier. Default 2. 
            EOS_ID (int, optional): End of sentence token identifier. Default 3. 
            PAD_ID (int, optional): Padding token identifier. Default 0. 
        """
        self.data = data
        self.seq_len = seq_len
        self.len_compounds = len(self.data)
        self.task_type = task_type
        self.symbol_ID = symbol_ID
        self.max_length_symbol = max_length_symbol
        self.change_smi = change_smi
        self.BOS_ID = BOS_ID
        self.EOS_ID = EOS_ID
        self.PAD_ID = PAD_ID
        self.mean = mean
        self.sd = sd

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
        tokenizes the SMILES, and standardizes the property value if task type
        is equal to 'regression'. 

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

        tokens = self.smiles_str2smiles(smiles)

        tokens = [self.BOS_ID]+tokens+[self.EOS_ID]
        tokens = tokens[0:self.seq_len]
        padding = [self.PAD_ID for _ in range(self.seq_len - len(tokens))]
        tokens.extend(padding)
        tokens = torch.tensor(tokens)  

        label = np.array(self.data['Y'].iloc[item])
        
        if self.task_type == 'regression':
            label = (label-self.mean)/self.sd

        label = torch.tensor(label).float()
        label = label.unsqueeze(0)
    
        return tokens, label



