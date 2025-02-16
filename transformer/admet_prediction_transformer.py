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

sys.path.append(os.getcwd())
from transformer.Model.Admet_prediction_func import admet_performance

#####################################
### Prepare tokenizer - smilesPE ###
#####################################


device = "cuda"
d_model = 512
n_layers = 6
heads = 8
dropout = 0.1

if not os.path.exists('admet/'):
    os.makedirs('admet/')
group = admet_group(path = '/vol/bertdata/admet/')

batch_size = 64
epoch = 50

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
results_onelayer_256_transformer = admet_performance(seed_options = [1, 2, 3, 4, 5], d_model = d_model, 
                                             n_layers = n_layers, heads = heads, dropout = dropout, 
                                             device = device, lr = 0.0001, batch_size = batch_size, 
                                             num_epochs = epoch, group = group, group_config = benchmark_config,
                                             model_dir = "best_models/", 
                                             hidden_layer = 256, 
                                             blocks_freeze = ['module.encoder.layers.5.', 
                                                              'module.encoder.norm.alpha', 
                                                              'module.encoder.norm.bias'])

print("Results Transformer - one layer - hidden 256:")
print(results_onelayer_256_transformer)

with open("results/transformer_onelayer_256.json", "w") as json_file:
    json.dump(results_onelayer_256_transformer, json_file, indent=4)
    

results_twolayer_256_transformer = admet_performance(seed_options = [1, 2, 3, 4, 5], d_model = d_model, 
                                             n_layers = n_layers, heads = heads, dropout = dropout, 
                                             device = device, lr = 0.0001, batch_size = batch_size, 
                                             num_epochs = epoch, group = group, group_config = benchmark_config,
                                             model_dir = "best_models/", 
                                             hidden_layer = 256, 
                                             blocks_freeze = ['module.encoder.layers.5.',
                                                              'module.encoder.layers.4.'
                                                              'module.encoder.norm.alpha', 
                                                              'module.encoder.norm.bias'])

print("Results Transformer - two layer - hidden 256:")
print(results_twolayer_256_transformer)

with open("results/transformer_twolayer_256.json", "w") as json_file:
    json.dump(results_twolayer_256_transformer, json_file, indent=4)


results_onelayer_512_transformer = admet_performance(seed_options = [1, 2, 3, 4, 5], d_model = d_model, 
                                             n_layers = n_layers, heads = heads, dropout = dropout, 
                                             device = device, lr = 0.0001, batch_size = batch_size, 
                                             num_epochs = epoch, group = group, group_config = benchmark_config,
                                             model_dir = "best_models/", 
                                             hidden_layer = 512, 
                                             blocks_freeze = ['module.encoder.layers.5.', 
                                                              'module.encoder.norm.alpha', 
                                                              'module.encoder.norm.bias'])

print("Results Transformer - one layer - hidden 512:")
print(results_onelayer_512_transformer)

with open("results/transformer_onelayer_512.json", "w") as json_file:
    json.dump(results_onelayer_512_transformer, json_file, indent=4)
    

results_twolayer_512_transformer = admet_performance(seed_options = [1, 2, 3, 4, 5], d_model = d_model, 
                                             n_layers = n_layers, heads = heads, dropout = dropout, 
                                             device = device, lr = 0.0001, batch_size = batch_size, 
                                             num_epochs = epoch, group = group, group_config = benchmark_config,
                                             model_dir = "best_models/", 
                                             hidden_layer = 512, 
                                             blocks_freeze = ['module.encoder.layers.5.',
                                                              'module.encoder.layers.4.'
                                                              'module.encoder.norm.alpha', 
                                                              'module.encoder.norm.bias'])

print("Results Transformer - two layer - hidden 512:")
print(results_twolayer_512_transformer)

with open("results/transformer_twolayer_512.json", "w") as json_file:
    json.dump(results_twolayer_512_transformer, json_file, indent=4)









