import math
import torch
import torch.nn as nn

#----------------------------------#
#           smiles_str2smiles      #
#----------------------------------#
# translate a string of SMILES to a list of symbol ID

def smiles_str2smiles(smiles_str, max_length_symbol, symbol_ID, flag=False):
  #translate a string of SMILES to a list of symbol ID（symbols such as 'Na' will translated to ID of 'Na')

  smiles = []
  i=0
  while i < len(smiles_str):
    NotFindID = True
    for j in range(max_length_symbol,0,-1) :
      if i+j <= len(smiles_str) and smiles_str[i:i+j] in symbol_ID: # 
        smiles.append(symbol_ID[smiles_str[i:i+j]])
        i += j-1 
        NotFindID = False
        break
    if NotFindID:
      smiles.append(4) # using MSK_ID to replace unknown symbol
    i += 1
  return smiles

#----------------------------------#
#           smiles2smiles_str      #
#----------------------------------#
def smiles2smiles_str(smiles):
  smiles_str = ''
  for id in smiles:
    smiles_str += ID_symbol[id]
  return smiles_str

# ----------------------------------------------------------------------------
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# ----------------------------------------------------------------------------
class SymbolEncoder(nn.Module):
    def __init__(self, num_token, d_model, InitRange = 0.1, PAD_ID = 0):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(num_token, d_model, padding_idx=PAD_ID)
        self.embed.weight.data.uniform_(-InitRange, InitRange)  # embedding init

    def forward(self, src):
        src = self.embed(src) * math.sqrt(self.d_model)
        return src

# ----------------------------------------------------------------------------
class MyTransformerEncoder(nn.Module):
    def __init__(self, d_model, num_head, d_hidden, NumLayers = 10, Dropout = 0.1, Activation = 'gelu', NormFirst = True):
        super().__init__()

        encoder_layers = nn.TransformerEncoderLayer(
            d_model,
            num_head,
            dim_feedforward=d_hidden,
            norm_first=NormFirst,
            activation=Activation,
            dropout=Dropout,
            batch_first=True
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, NumLayers, norm=encoder_norm)

    def forward(self, x, padding_mask):
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        return x

# ----------------------------------------------------------------------------
class Model_2enc(nn.Module):
    def __init__(self, lID, task_type, NumToken, DimEmbed = 256, DimTfHidden = 512, 
                 Dropout = 0.1, NumHead = 8, PAD_ID = 0, device = 'cuda'):
        super().__init__()
        self.task_type = task_type
        self.PAD_ID = PAD_ID
        self.device = device
        self.d_model = DimEmbed

        self.positional_encoder = PositionalEncoder(DimEmbed)
        self.symbol_encoder = SymbolEncoder(NumToken, DimEmbed)
        self.smiles_encoder = MyTransformerEncoder(DimEmbed, NumHead, DimTfHidden)
        self.drop0 = nn.Dropout(p=Dropout)

        self.drop = nn.Dropout(p=Dropout)
        self.fnn = nn.Linear(DimEmbed, lID)
        self.act = nn.Sigmoid()

        self.drop2 = nn.Dropout(p=Dropout)
        self.fnn2 = nn.Linear(DimEmbed, DimEmbed)
        self.act2 = nn.ReLU()

    def forward(self, smiles):
        # add cls token
        cls = torch.ones(smiles.size(0), 1, dtype=torch.long).to(self.device)
        x = torch.concat((cls, smiles), dim=1)
        padding_mask = (x == self.PAD_ID)
        x = self.drop0(self.symbol_encoder(x) + self.positional_encoder(x))
        x = self.smiles_encoder(x, padding_mask)  # x : OdorCode

        embedding_vector = x[:, 0, :]
        x = self.act2(self.fnn2(self.drop2(x[:, 0, :])))

        if self.task_type == 'classification':
            return self.act(self.fnn(self.drop(x))), embedding_vector
        else:
            return self.fnn(self.drop(x)), embedding_vector

# ----------------------------------------------------------------------------

