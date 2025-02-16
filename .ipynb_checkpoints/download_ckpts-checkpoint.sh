mkdir -p ckpts

wget https://github.com/m-baralt/smile-to-bert/releases/download/v1.0/ckpt_descriptors.pt -O ckpts/ckpt_smiletobert.pt
wget https://github.com/m-baralt/smile-to-bert/releases/download/v1.0/OdorCode-40.smiles_encoder.D256.Hidden512.Head8.L10.R0.5.S100000-epoch.600 -O ckpts/OdorCode-40.smiles_encoder.D256.Hidden512.Head8.L10.R0.5.S100000-epoch.600
wget https://github.com/m-baralt/smile-to-bert/releases/download/v1.0/OdorCode-40.symbol_encoder.D256.Hidden512.Head8.L10.R0.5.S100000-epoch.600 -O ckpts/OdorCode-40.symbol_encoder.D256.Hidden512.Head8.L10.R0.5.S100000-epoch.600
wget https://github.com/m-baralt/smile-to-bert/releases/download/v1.0/transformer_ckpt.ckpt -O ckpts/transformer_ckpt.ckpt



