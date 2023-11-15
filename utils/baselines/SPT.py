# adapted from https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/models.py
import torch
from torch import nn
import torch.nn.functional as F

from utils.baselines.submodels import SPEncoder
from utils.models_transformer import BertTextEncoder


class Superformer(nn.Module):
    def __init__(self):
        """
        Construct a Superformer model.
        """
        super(Superformer, self).__init__()

        self.text_model = BertTextEncoder()  # BERT version

        self.d_model = 32
        self.input_dims = dict(t=768,
                               a=74,  # MOSEI - 74 / MOSI - 5
                               v=35)  # MOSEI - 35 / MOSI - 20
        self.num_heads = 8
        self.layers = 4
        self.attn_dropout = 0.2
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.1
        self.embed_dropout = 0.3
        self.S, self.r = 5, [8, 4, 3]
        self.shift_mode = dict(I=['S,P,R'], X=['S'], S=['S'], C=[1, 0.25, 0.05])
        self.use_fast = False
        self.use_dense = False
        combined_dim = 3 * self.d_model

        self.spe = SPEncoder(embed_dim=self.d_model,
                             input_dims=self.input_dims,
                             num_heads=self.num_heads,
                             layers=self.layers,
                             attn_dropout=self.attn_dropout,
                             relu_dropout=self.relu_dropout,
                             res_dropout=self.res_dropout,
                             embed_dropout=self.embed_dropout,
                             S=self.S, r=self.r,
                             shift_mode=self.shift_mode,
                             use_fast=self.use_fast,
                             use_dense=self.use_dense)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, 1)

    def forward(self, t, a, v):  # [BS,SL,D]

        t = self.text_model(t)  # BERT version

        h_a, h_t, h_v = self.spe(a, t, v)
        last_hs = torch.cat([h_t[-1], h_a[-1], h_v[-1]], dim=1)
        # last_hs = torch.cat([torch.mean(h_t,0), torch.mean(h_a,0), torch.mean(h_v,0)], dim=1)
        # last_hs = self.spe(a,t,v)[-1]

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return output
