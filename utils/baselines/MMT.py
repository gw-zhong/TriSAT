import torch
import time
from torch import nn
import torch.nn.functional as F
from six.moves import reduce

from utils.baselines.submodels import MMTLayer
from utils.models_transformer import BertTextEncoder


class MMT(nn.Module):
    def __init__(self, num_layers, text_dim, video_dim, audio_dim, r1, r2, r3, alpha, output_dim=1, seq_len=50):
        super().__init__()

        self.text_model = BertTextEncoder()  # BERT version

        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            new_layer = MMTLayer(text_dim=text_dim,
                                 video_dim=video_dim,
                                 audio_dim=audio_dim,
                                 r1=r1,
                                 r2=r2,
                                 r3=r3,
                                 alpha=alpha,
                                 seq_len=seq_len)
            self.layers.append(new_layer)
        self.out_layer = nn.Linear(text_dim + video_dim + audio_dim, output_dim)

    def forward(self, t, v, a):

        t = self.text_model(t)  # BERT version

        for layer in self.layers:
            t, v, a = layer(t, v, a)
        t = torch.mean(t, dim=1)  # (B, d_t)
        v = torch.mean(v, dim=1)  # (B, d_v)
        a = torch.mean(a, dim=1)  # (B, d_a)
        output = self.out_layer(torch.cat((t, v, a), dim=-1))  # (B, output_dim)
        return output
