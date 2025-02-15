######################
### import modules ###
######################
import sys
import torch
from transformers import BertTokenizer
import os
import codecs
from SmilesPE.tokenizer import SPE_Tokenizer
from tdc.benchmark_group import admet_group
import json
import pickle

sys.path.append(os.getcwd())
from model2encoder.Model.Admet_prediction_func import admet_performance

#####################################
### Prepare tokenizer - smilesPE ###
#####################################

device = 'cuda'

group = admet_group(path = '/vol/bertdata/admet/')

batch_size = 64
epoch = 50
PAD_ID = 0
CLS_ID = 1
BOS_ID = 2
EOS_ID = 3
MSK_ID = 4

f = open('model2encoder/data/OdorCode-40 Symbol Dictionary', 'rb')
[symbol_ID, ID_symbol, sID] = pickle.load(f)
f.close()

max_length_symbol = max([len(s) for s in ID_symbol])

benchmark_config = {
    'caco2_wang': 'regression',
    'bioavailability_ma': 'classification',
    'lipophilicity_astrazeneca':'regression',
    'solubility_aqsoldb': 'regression',
    'hia_hou': 'regression',
    'pgp_broccatelli': 'classification',
    'bbb_martins': 'classification',
    'ppbr_az': 'regression',
    'vdss_lombardo': 'regression',
    'cyp2c9_veith': 'classification',
    'cyp2d6_veith': 'classification',
    'cyp3a4_veith': 'classification',
    'cyp2c9_substrate_carbonmangels': 'classification',
    'cyp2d6_substrate_carbonmangels': 'classification',
    'cyp3a4_substrate_carbonmangels': 'classification',
    'half_life_obach': 'regression',
    'clearance_hepatocyte_az': 'regression',
    'clearance_microsome_az': 'regression',
    'ld50_zhu': 'regression',
    'herg': 'classification',
    'ames': 'classification',
    'dili': 'classification'
}


results_onelayer_2encoder = admet_performance(seed_options = [1, 2, 3, 4, 5], sID = sID,
                                              device = device, lr = 0.0001, batch_size = batch_size, 
                                              num_epochs = epoch, group = group, 
                                              group_config = benchmark_config,
                                              model_dir = "/vol/bertdata/smiletobert_admet/best_models/",
                                              twolayers = False, seq_len = 100, symbol_ID = symbol_ID)

print("Results 2-encoder - one layer:")
print(results_onelayer_2encoder)

with open("/home/ubuntu/Bertsmiles_git/TDC_ADMET/results_paper/onelayer_2encoder.json", "w") as json_file:
    json.dump(results_onelayer_2encoder, json_file, indent=4)

results_twolayer_2encoder = admet_performance(seed_options = [1, 2, 3, 4, 5], sID = sID,
                                              device = device, lr = 0.0001, batch_size = batch_size, 
                                              num_epochs = epoch, group = group, 
                                              group_config = benchmark_config,
                                              model_dir = "/vol/bertdata/smiletobert_admet/best_models/",
                                              twolayers = True, seq_len = 100, symbol_ID = symbol_ID)

print("Results 2-encoder - two layer:")
print(results_twolayer_2encoder)

with open("/home/ubuntu/Bertsmiles_git/TDC_ADMET/results_paper/twolayer_2encoder.json", "w") as json_file:
    json.dump(results_twolayer_2encoder, json_file, indent=4)
    










