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
from combined_models.Model_transf_spe.Admet_prediction_func import admet_performance
from transformer.Model.Transformer import ALPHABET_SIZE

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


#results_onelayer_256_transf_bert = admet_performance(seed_options = [1, 2, 3, 4, 5], 
 #                                                    tokenizer = spe_tokenizer, 
  #                                                   d_model = d_model, n_layers = n_layers, 
   #                                                  heads = heads, dropout = dropout, 
    #                                                 seq_len = 100, device = device, lr = 0.0001, 
     #                                                batch_size = batch_size, num_epochs = epoch, 
      #                                               group = group, 
       #                                              group_config = benchmark_config, 
        #                                             model_dir = "/vol/bertdata/smiletobert_admet/best_models/", 
         #                                            hidden_layer = 256, alphabet_size = ALPHABET_SIZE,
          #                                           blocks_freeze_t = ['module.encoder.layers.5.',
           #                                                             'module.encoder.norm.alpha', 
            #                                                            'module.encoder.norm.bias'],
             #                                        blocks_freeze_b = ['bert.encoder_blocks.3'], spe = spe)

#print("Results Smile-to-Bert and transformer combined - one layer - hidden 256:")
#print(results_onelayer_256_transf_bert)


#with open("/home/ubuntu/Bertsmiles_git/TDC_ADMET/results_paper/transf_bert_onelayer_256.json", "w") as json_file:
 #   json.dump(results_onelayer_256_transf_bert, json_file, indent=4)

results_twolayer_256_transf_bert = admet_performance(seed_options = [1, 2, 3, 4, 5], 
                                                     tokenizer = spe_tokenizer, 
                                                     d_model = d_model, n_layers = n_layers, 
                                                     heads = heads, dropout = dropout, 
                                                     seq_len = 100, device = device, lr = 0.0001, 
                                                     batch_size = batch_size, num_epochs = epoch, 
                                                     group = group, 
                                                     group_config = benchmark_config, 
                                                     model_dir = "/vol/bertdata/smiletobert_admet/best_models/", 
                                                     hidden_layer = 256, alphabet_size = ALPHABET_SIZE,
                                                     blocks_freeze_t = ['module.encoder.layers.5.',
                                                                        'module.encoder.layers.4.',
                                                                        'module.encoder.norm.alpha', 
                                                                        'module.encoder.norm.bias'],
                                                     blocks_freeze_b = ['bert.encoder_blocks.3',
                                                                       'bert.encoder_blocks.2'], spe = spe)

print("Results Smile-to-Bert and transformer combined - two layers - hidden 256:")
print(results_twolayer_256_transf_bert)


with open("/home/ubuntu/Bertsmiles_git/TDC_ADMET/results_paper/transf_bert_twolayer_256.json", "w") as json_file:
    json.dump(results_twolayer_256_transf_bert, json_file, indent=4)


results_onelayer_512_transf_bert = admet_performance(seed_options = [1, 2, 3, 4, 5], 
                                                     tokenizer = spe_tokenizer, 
                                                     d_model = d_model, n_layers = n_layers, 
                                                     heads = heads, dropout = dropout, 
                                                     seq_len = 100, device = device, lr = 0.0001, 
                                                     batch_size = batch_size, num_epochs = epoch, 
                                                     group = group, 
                                                     group_config = benchmark_config, 
                                                     model_dir = "/vol/bertdata/smiletobert_admet/best_models/", 
                                                     hidden_layer = 512, alphabet_size = ALPHABET_SIZE,
                                                     blocks_freeze_t = ['module.encoder.layers.5.',
                                                                        'module.encoder.norm.alpha', 
                                                                        'module.encoder.norm.bias'],
                                                     blocks_freeze_b = ['bert.encoder_blocks.3'], spe = spe)

print("Results Smile-to-Bert and transformer combined - one layer - hidden 512:")
print(results_onelayer_512_transf_bert)

with open("/home/ubuntu/Bertsmiles_git/TDC_ADMET/results_paper/transf_bert_onelayer_512.json", "w") as json_file:
    json.dump(results_onelayer_512_transf_bert, json_file, indent=4)

results_twolayer_512_transf_bert = admet_performance(seed_options = [1, 2, 3, 4, 5], 
                                                     tokenizer = spe_tokenizer, 
                                                     d_model = d_model, n_layers = n_layers, 
                                                     heads = heads, dropout = dropout, 
                                                     seq_len = 100, device = device, lr = 0.0001, 
                                                     batch_size = batch_size, num_epochs = epoch, 
                                                     group = group, 
                                                     group_config = benchmark_config, 
                                                     model_dir = "/vol/bertdata/smiletobert_admet/best_models/", 
                                                     hidden_layer = 512, alphabet_size = ALPHABET_SIZE,
                                                     blocks_freeze_t = ['module.encoder.layers.5.',
                                                                        'module.encoder.layers.4.',
                                                                        'module.encoder.norm.alpha', 
                                                                        'module.encoder.norm.bias'],
                                                     blocks_freeze_b = ['bert.encoder_blocks.3',
                                                                       'bert.encoder_blocks.2'], spe = spe)

print("Results Smile-to-Bert and transformer combined - two layers - hidden 512:")
print(results_twolayer_512_transf_bert)


with open("/home/ubuntu/Bertsmiles_git/TDC_ADMET/results_paper/transf_bert_twolayer_512.json", "w") as json_file:
    json.dump(results_twolayer_512_transf_bert, json_file, indent=4)




