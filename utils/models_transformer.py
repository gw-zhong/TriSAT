import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from utils.multimodal_transformer import MultimodalTransformerEncoder
from utils.transformer import TransformerEncoder

from transformers import BertTokenizer, BertModel


class Net(nn.Module):
    def __init__(self,
                 text_input_size,
                 video_input_size,
                 audio_input_size,
                 embed_dim,
                 dim_total_proj,
                 num_layers,
                 num_heads,
                 output_dim,
                 seq_len,
                 attn_dropout,
                 embed_dropout,
                 use_bert,
                 kernel_size=1,
                 attn_dropout_v=0.0,
                 attn_dropout_a=0.0,
                 relu_dropout=0.1,
                 res_dropout=0.1):
        super(Net, self).__init__()

        ########################################################
        # parameters

        self.text_input_size = text_input_size
        self.video_input_size = video_input_size
        self.audio_input_size = audio_input_size

        self.embed_dim = embed_dim
        self.dim_total_proj = dim_total_proj
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.seq_len = seq_len

        self.attn_dropout = attn_dropout
        self.embed_dropout = embed_dropout

        temperature = torch.ones([]) * 1.0
        self.temp = nn.Parameter(temperature)

        self.use_bert = use_bert
        if use_bert:
            self.text_model = BertTextEncoder()

        ########################################################
        # use conv1d to proj

        self.text_proj = nn.Conv1d(
            self.text_input_size, self.embed_dim, kernel_size=kernel_size, padding=0, bias=False)
        self.video_proj = nn.Conv1d(
            self.video_input_size, self.embed_dim, kernel_size=kernel_size, padding=0, bias=False)
        self.audio_proj = nn.Conv1d(
            self.audio_input_size, self.embed_dim, kernel_size=kernel_size, padding=0, bias=False)

        ########################################################
        # contrastive learning

        self.weight_t = nn.Linear(embed_dim, 1)
        self.weight_v = nn.Linear(embed_dim, 1)
        self.weight_a = nn.Linear(embed_dim, 1)

        self.t_tran = TransformerEncoder(embed_dim=self.embed_dim,
                                         num_heads=self.num_heads,
                                         layers=max(3, self.num_layers),
                                         attn_dropout=self.attn_dropout,
                                         relu_dropout=relu_dropout,
                                         res_dropout=res_dropout,
                                         embed_dropout=self.embed_dropout)

        ########################################################
        # transformer & gru & fc

        self.v_multi_tran = MultimodalTransformerEncoder(embed_dim=self.embed_dim,
                                                         num_heads=self.num_heads,
                                                         layers=self.num_layers,
                                                         attn_dropout=attn_dropout_v,
                                                         relu_dropout=relu_dropout,
                                                         res_dropout=res_dropout,
                                                         embed_dropout=self.embed_dropout)
        self.a_multi_tran = MultimodalTransformerEncoder(embed_dim=self.embed_dim,
                                                         num_heads=self.num_heads,
                                                         layers=self.num_layers,
                                                         attn_dropout=attn_dropout_a,
                                                         relu_dropout=relu_dropout,
                                                         res_dropout=res_dropout,
                                                         embed_dropout=self.embed_dropout)

        self.gru = nn.GRU(input_size=self.embed_dim * 2, hidden_size=self.dim_total_proj)
        # self.gru = nn.GRU(input_size=self.embed_dim, hidden_size=self.dim_total_proj)

        self.output_layer = nn.Linear(self.dim_total_proj, self.output_dim)

        ########################################################
        # weights initialize
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                init.kaiming_normal_(m.weight_ih_l0)
                if m.bias_ih_l0 is not None:
                    init.constant_(m.bias_ih_l0, 0)
                init.orthogonal_(m.weight_hh_l0)
                if m.bias_hh_l0 is not None:
                    init.constant_(m.bias_hh_l0, 0)
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if not self.use_bert:
                if isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

    def forward(self, x_t, x_v, x_a):

        if self.use_bert:
            x_t = self.text_model(x_t)
        ########################################################
        # use conv1d to proj

        x_t = F.dropout(x_t.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = x_v.transpose(1, 2)
        x_a = x_a.transpose(1, 2)

        proj_t = self.text_proj(x_t)
        proj_v = self.video_proj(x_v)
        proj_a = self.audio_proj(x_a)
        proj_t = proj_t.permute(2, 0, 1)
        proj_v = proj_v.permute(2, 0, 1)
        proj_a = proj_a.permute(2, 0, 1)  # (seq, batch, embed_dim)

        proj_t, _ = self.t_tran(proj_t)  # self-attention to language

        ########################################################
        # contrastive learning

        # use attention mechanism to get the batch feature

        w_t = torch.tanh(
            self.weight_t(proj_t.contiguous().view(-1, self.embed_dim)).view(-1, self.seq_len))  # (batch, seq)
        w_t = F.softmax(w_t, dim=-1)
        c_t = torch.sum(w_t.unsqueeze(-1) * proj_t.transpose(0, 1), dim=1)  # (batch, embed_dim)

        w_v = torch.tanh(
            self.weight_v(proj_v.contiguous().view(-1, self.embed_dim)).view(-1, self.seq_len))  # (batch, seq)
        w_v = F.softmax(w_v, dim=-1)
        c_v = torch.sum(w_v.unsqueeze(-1) * proj_v.transpose(0, 1), dim=1)  # (batch, embed_dim)

        w_a = torch.tanh(
            self.weight_a(proj_a.contiguous().view(-1, self.embed_dim)).view(-1, self.seq_len))  # (batch, seq)
        w_a = F.softmax(w_a, dim=-1)
        c_a = torch.sum(w_a.unsqueeze(-1) * proj_a.transpose(0, 1), dim=1)  # (batch, embed_dim)

        similarity_cube = torch.einsum('at,bt,ct->abc', c_a, c_t, c_v) / self.temp

        ########################################################
        # transformer layer & gru layer & fc layer

        h_t_1, _ = self.v_multi_tran(proj_t, proj_v, proj_v)  # (seq, batch, embed_dim)
        h_t_2, _ = self.a_multi_tran(proj_t, proj_a, proj_a)
        h_t = torch.cat((h_t_1, h_t_2), dim=-1)
        _, h = self.gru(h_t)  # (1, batch, dim_total_proj)
        # _, h = self.gru(h_t_2)  # (1, batch, dim_total_proj)
        y = self.output_layer(h).squeeze(0)

        return y, similarity_cube


class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=True):
        """
        language: en / cn
        """
        super(BertTextEncoder, self).__init__()

        tokenizer_class = BertTokenizer
        model_class = BertModel
        self.tokenizer = tokenizer_class.from_pretrained('bert_en', do_lower_case=True)
        self.model = model_class.from_pretrained('bert_en')

        self.use_finetune = use_finetune

    def get_tokenizer(self):
        return self.tokenizer

    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()

    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states
