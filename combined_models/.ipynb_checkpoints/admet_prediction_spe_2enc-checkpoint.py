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
from combined_models.Model_2enc_spe.Admet_prediction_func import admet_performance

#####################################
### Prepare tokenizer - smilesPE ###
#####################################

spe_vob = codecs.open('smiletobert/data/spe_tokenizer/SPE_ChEMBL.txt')
spe = SPE_Tokenizer(spe_vob)
spe_tokenizer = BertTokenizer("smiletobert/data/spe_tokenizer/vocab_spe.txt")

device = "cuda"
d_model = 512
n_layers = 4
heads = 8
dropout = 0.1


if not os.path.exists('admet/'):
    os.makedirs('admet/')
group = admet_group(path = 'admet/')

batch_size = 64
epoch = 50

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

if not os.path.exists('results/'):
    os.makedirs('results/')

results_onelayer_256_2enc_bert = admet_performance(seed_options = [1, 2, 3, 4, 5], 
                                                   tokenizer = spe_tokenizer, 
                                                   symbol_ID = symbol_ID,
                                                   d_model = d_model, n_layers = n_layers, 
                                                   heads = heads, dropout = dropout, 
                                                   seq_len = 100, device = device, lr = 0.0001,
                                                   batch_size = batch_size, num_epochs = epoch, 
                                                   group = group, group_config = benchmark_config, 
                                                   model_dir = "best_models/", 
                                                   hidden_layer = 256, lID = 1, sID = sID,
                                                   blocks_freeze_b = ['bert.encoder_blocks.3'], 
                                                   spe = spe)

print("Results Smile-to-Bert and 2-encoder combined - one layer - hidden 256:")
print(results_onelayer_256_2enc_bert)


with open("results/c2enc_bert_onelayer_256.json", "w") as json_file:
    json.dump(results_onelayer_256_2enc_bert, json_file, indent=4)


results_onelayer_768_2enc_bert = admet_performance(seed_options = [1, 2, 3, 4, 5], 
                                                   tokenizer = spe_tokenizer, 
                                                   symbol_ID = symbol_ID,
                                                   d_model = d_model, n_layers = n_layers, 
                                                   heads = heads, dropout = dropout, 
                                                   seq_len = 100, device = device, lr = 0.0001, 
                                                   batch_size = batch_size, num_epochs = epoch, 
                                                   group = group, group_config = benchmark_config, 
                                                   model_dir = "best_models/", 
                                                   hidden_layer = 768, lID = 1, sID = sID,
                                                   blocks_freeze_b = ['bert.encoder_blocks.3'], 
                                                   spe = spe)

print("Results Smile-to-Bert and 2-encoder combined - one layer - hidden 768:")
print(results_onelayer_768_2enc_bert)


with open("results/c2enc_bert_onelayer_768.json", "w") as json_file:
    json.dump(results_onelayer_768_2enc_bert, json_file, indent=4)

results_twolayer_256_2enc_bert = admet_performance(seed_options = [1, 2, 3, 4, 5], 
                                                   tokenizer = spe_tokenizer, 
                                                   symbol_ID = symbol_ID,
                                                   d_model = d_model, n_layers = n_layers, 
                                                   heads = heads, dropout = dropout, 
                                                   seq_len = 100, device = device, lr = 0.0001, 
                                                   batch_size = batch_size, num_epochs = epoch, 
                                                   group = group, group_config = benchmark_config, 
                                                   model_dir = "best_models/", 
                                                   hidden_layer = 256, lID = 1, sID = sID,
                                                   blocks_freeze_b = ['bert.encoder_blocks.3',
                                                                       'bert.encoder_blocks.2'], 
                                                   spe = spe)

print("Results Smile-to-Bert and 2-encoder combined - two layer - hidden 256:")
print(results_twolayer_256_2enc_bert)


with open("results/c2enc_bert_twolayer_256.json", "w") as json_file:
    json.dump(results_twolayer_256_2enc_bert, json_file, indent=4)


results_twolayer_768_2enc_bert = admet_performance(seed_options = [1, 2, 3, 4, 5], 
                                                   tokenizer = spe_tokenizer, 
                                                   symbol_ID = symbol_ID,
                                                   d_model = d_model, n_layers = n_layers, 
                                                   heads = heads, dropout = dropout, 
                                                   seq_len = 100, device = device, lr = 0.0001, 
                                                   batch_size = batch_size, num_epochs = epoch, 
                                                   group = group, group_config = benchmark_config, 
                                                   model_dir = "best_models/", 
                                                   hidden_layer = 768, lID = 1, sID = sID,
                                                   blocks_freeze_b = ['bert.encoder_blocks.3',
                                                                       'bert.encoder_blocks.2'], 
                                                   spe = spe)

print("Results Smile-to-Bert and 2-encoder combined - two layer - hidden 768:")
print(results_twolayer_768_2enc_bert)


with open("results/c2enc_bert_twolayer_768.json", "w") as json_file:
    json.dump(results_twolayer_768_2enc_bert, json_file, indent=4)


