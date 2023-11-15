import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from einops import rearrange, repeat
from functools import partial


class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device()
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name, getattr(make_positions, buf_name).type_as(tensor))
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=getattr(make_positions, buf_name))
    mask = tensor.ne(padding_idx)
    positions = getattr(make_positions, buf_name)[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()  # device --> actual weight; due to nn.DataParallel :-(
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # recompute/expand embeddings if needed
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights[device].index_select(0, positions.contiguous().view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout  # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)  # The "Add & Norm" part in the paper
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1 + abs(dim2 - dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


class SPEncoder(nn.Module):

    def __init__(self, input_dims, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0,
                 res_dropout=0.0, embed_dropout=0.0, S=1, r=[1, 1, 1],
                 shift_mode=dict(I=['S', 'P', 'R'], X=['S'], S=['S'], C=[1, 0.25, 0.05]),
                 use_fast=False, use_dense=False, device='cuda'):
        super().__init__()
        self.dropout = embed_dropout  # Embedding dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SPSinusoidalPositionalEmbedding(embed_dim)
        self.init_hiddens = nn.Parameter(torch.Tensor(3, embed_dim))
        nn.init.xavier_uniform_(self.init_hiddens)
        self.shift_mode = shift_mode
        self.use_fast = use_fast
        self.use_dense = use_dense
        self.device = device

        if self.use_fast:
            self.proj_a = nn.Conv1d(input_dims['a'], self.embed_dim, kernel_size=1)
            self.proj_t = nn.Conv1d(input_dims['t'], self.embed_dim, kernel_size=1)
            self.proj_v = nn.Conv1d(input_dims['v'], self.embed_dim, kernel_size=1)
            input_dims = dict(a=self.embed_dim, t=self.embed_dim, v=self.embed_dim)

        self.layers, self.stride = layers, S
        self.hiddenlayer = SPEncoderHiddenLayer(embed_dim,
                                                input_dims,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                stride=S,
                                                pf_hidden=r[0] * 2 + 1,
                                                use_fast=self.use_fast,
                                                use_dense=self.use_dense,
                                                device=self.device)
        self.crosslayer = SPEncoderCrossLayer(embed_dim,
                                              input_dims,
                                              num_heads=num_heads,
                                              attn_dropout=attn_dropout,
                                              relu_dropout=relu_dropout,
                                              res_dropout=res_dropout,
                                              pf_cross=r[1] * 2 + 1,
                                              use_fast=self.use_fast,
                                              use_dense=self.use_dense,
                                              device=self.device)
        self.selflayer = SPEncoderSelfLayer(embed_dim,
                                            input_dims,
                                            num_heads=num_heads,
                                            attn_dropout=attn_dropout,
                                            relu_dropout=relu_dropout,
                                            res_dropout=res_dropout,
                                            pf_self=r[2] * 2 + 1,
                                            use_fast=self.use_fast,
                                            use_dense=self.use_dense,
                                            device=self.device)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def init_emb(self, x):
        x = self.embed_scale * x
        if self.embed_positions is not None:
            x += self.embed_positions(x.transpose(0, 1)[:, :, 0], x.shape[-1]).transpose(0,
                                                                                         1)  # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, a, t, v):  # [BS, SL, D]
        sla, slt, slv, bs = a.shape[1], t.shape[1], v.shape[1], a.shape[0]
        hla, hlt, hlv = math.ceil(sla / self.stride), math.ceil(slt / self.stride), math.ceil(slv / self.stride)
        h_a = self.init_hiddens[0, :].unsqueeze(0).repeat(hla, bs, 1)
        h_t = self.init_hiddens[1, :].unsqueeze(0).repeat(hlt, bs, 1)
        h_v = self.init_hiddens[2, :].unsqueeze(0).repeat(hlv, bs, 1)

        # embed tokens and positions
        if self.use_fast:
            a = self.proj_a(a.transpose(1, 2)).permute(2, 0, 1)
            t = self.proj_t(t.transpose(1, 2)).permute(2, 0, 1)
            v = self.proj_v(v.transpose(1, 2)).permute(2, 0, 1)
        else:
            a, t, v = a.permute(1, 0, 2), t.permute(1, 0, 2), v.permute(1, 0, 2)
        a, t, v = self.init_emb(a), self.init_emb(t), self.init_emb(v)
        h_a, h_t, h_v = self.init_emb(h_a), self.init_emb(h_t), self.init_emb(h_v)

        # encoder layers
        shift_mode = self.shift_mode
        for i in range(self.layers):
            shift_a, shift_t, shift_v = 0, 0, 0
            if 'S' in shift_mode['I']:
                A = shift_mode['C'][0]
                shift_a += i * A;
                shift_t += i * A;
                shift_v += i * A
            if 'P' in shift_mode['I']:
                B = shift_mode['C'][1]
                shift_a += (sla * torch.sin(torch.arange(hla) * B)).long().to(self.device)
                shift_t += (slt * torch.sin(torch.arange(hlt) * B)).long().to(self.device)
                shift_v += (slv * torch.sin(torch.arange(hlv) * B)).long().to(self.device)
            if 'R' in shift_mode['I']:
                G = shift_mode['C'][2]
                shift_a += torch.randint(0, math.ceil(G * sla / hla), [hla], device=self.device)
                shift_t += torch.randint(0, math.ceil(G * slt / hlt), [hlt], device=self.device)
                shift_v += torch.randint(0, math.ceil(G * slv / hlt), [hlv], device=self.device)
            h_a, h_t, h_v = self.hiddenlayer(a, t, v, h_a, h_t, h_v, shift_a, shift_t, shift_v)

            shift_a, shift_t, shift_v = 0, 0, 0
            if 'S' in shift_mode['X']:
                A = shift_mode['C'][0]
                shift_a += i * A;
                shift_t += i * A;
                shift_v += i * A
            if 'P' in shift_mode['X']:
                B = shift_mode['C'][1]
                shift_a += (sla * torch.sin(torch.arange(hla) * B)).long().to(self.device)
                shift_t += (slt * torch.sin(torch.arange(hlt) * B)).long().to(self.device)
                shift_v += (slv * torch.sin(torch.arange(hlv) * B)).long().to(self.device)
            if 'R' in shift_mode['X']:
                G = shift_mode['C'][2]
                shift_a += torch.randint(0, math.ceil(G * sla / hla), [hla], device=self.device)
                shift_t += torch.randint(0, math.ceil(G * slt / hlt), [hlt], device=self.device)
                shift_v += torch.randint(0, math.ceil(G * slv / hlt), [hlv], device=self.device)
            h_a, h_t, h_v = self.crosslayer(h_a, h_t, h_v, shift_a, shift_t, shift_v)

            shift_a, shift_t, shift_v = 0, 0, 0
            if 'S' in shift_mode['S']:
                A = shift_mode['C'][0]
                shift_a += i * A;
                shift_t += i * A;
                shift_v += i * A
            if 'P' in shift_mode['S']:
                B = shift_mode['C'][1]
                shift_a += (sla * torch.sin(torch.arange(hla) * B)).long().to(self.device)
                shift_t += (slt * torch.sin(torch.arange(hlt) * B)).long().to(self.device)
                shift_v += (slv * torch.sin(torch.arange(hlv) * B)).long().to(self.device)
            if 'R' in shift_mode['S']:
                G = shift_mode['C'][2]
                shift_a += torch.randint(0, math.ceil(G * sla / hla), [hla], device=self.device)
                shift_t += torch.randint(0, math.ceil(G * slt / hlt), [hlt], device=self.device)
                shift_v += torch.randint(0, math.ceil(G * slv / hlt), [hlv], device=self.device)
            h_a, h_t, h_v = self.selflayer(h_a, h_t, h_v, shift_a, shift_t, shift_v)

        h_a = self.layer_norm(h_a)
        h_t = self.layer_norm(h_t)
        h_v = self.layer_norm(h_v)
        return h_a, h_t, h_v

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class FFN(nn.Module):
    def __init__(self, embed_dim, relu_dropout, res_dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)  # The "Add & Norm" part in the paper
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        return x


# helpers

def exists(val):
    return val is not None


def empty(tensor):
    return tensor.numel() == 0


def default(val, d):
    return val if exists(val) else d


# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data -
                          torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash.type_as(data)


def generalized_kernel(data, *, projection_matrix, kernel_fn=nn.ReLU(), kernel_epsilon=0.001, normalize_data=True,
                       device=None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.qr(unstructured_block.cpu(), some=True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    if multiplier.is_cuda: final_matrix = final_matrix.cuda()
    return torch.diag(multiplier) @ final_matrix


def linear_attention(q, k, v):
    k_cumsum = k.sum(dim=-2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out


class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, generalized_attention=False, kernel_fn=nn.ReLU()):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features,
                                         nb_columns=dim_heads, scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

    def forward(self, q, k, v):
        device = q.device

        if self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn=self.kernel_fn,
                                    projection_matrix=self.projection_matrix, device=device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = linear_attention
        out = attn_fn(q, k, v)
        return out


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            nb_features=None,
            generalized_attention=False,
            kernel_fn=nn.ReLU(),
            dropout=0.,
            qkv_bias=False,
            attn_out_bias=True
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, generalized_attention=generalized_attention,
                                            kernel_fn=kernel_fn)

        self.heads = heads
        self.global_heads = heads

        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb=None, context=None, mask=None, context_mask=None, reverse=False, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        if not reverse:
            q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        else:
            q, k, v = self.to_k(x), self.to_q(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim=1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class SPF(nn.Module):  # Sparse Phased Fast attention
    def __init__(self, embed_dim, num_heads, attn_dropout, res_dropout, input_dim=None, stride=1, pf=None,
                 generalized_attention=False, dim_head_down=1, use_fast=True, use_dense=False, device='cuda', ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.res_dropout = res_dropout
        self.stride, self.pf = stride, pf
        self.generalized_attention = generalized_attention
        self.dim_head = int(embed_dim / num_heads / dim_head_down)
        self.use_fast = use_fast
        self.use_dense = use_dense

        if not use_dense and use_fast:
            self.attn = Attention(
                self.embed_dim,
                heads=self.num_heads,
                dim_head=self.dim_head,
                generalized_attention=self.generalized_attention
            )
        else:
            self.attn = SPMultiheadAttention(self.embed_dim, self.num_heads, input_dim=input_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        input_dim = self.embed_dim if input_dim is None else input_dim
        self.layer_norm_kv = nn.LayerNorm(input_dim)
        self.device = device

    def forward(self, x, x_k=None, x_v=None, shift=0, reverse=False):
        sl, bs, _ = x.shape
        residual = x
        x = self.layer_norm(x)
        context = x if x_k is None else x_k
        if self.use_dense:
            mask = sparse_mask(x, context, self.stride, self.pf);
            c = context
        else:
            fetch = sparsify(x, context, self.stride, self.pf, shift, self.device)
            x = x.unsqueeze(2).reshape(-1, 1, self.embed_dim)
            c = fetch.permute(1, 2, 0, 3).reshape(-1, self.pf, fetch.shape[-1])
            if not self.use_fast: x = x.permute(1, 0, 2); c = c.permute(1, 0, 2)
        if x_k is not None: c = self.layer_norm_kv(c)
        if self.use_dense:
            x, _ = self.attn(x, c, c, reverse=reverse, attn_mask=mask)
        else:
            if self.use_fast:
                x = self.attn(x, context=c, reverse=reverse)
            else:
                x, _ = self.attn(x, c, c, reverse=reverse)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = x.squeeze(1).reshape(sl, bs, -1)
        x = residual + x
        return x


class SPEncoderHiddenLayer(nn.Module):
    def __init__(self, embed_dim, input_dims, num_heads=4, attn_dropout=0.1, relu_dropout=0.1,
                 res_dropout=0.1, stride=None, pf_hidden=None, use_fast=False, use_dense=False, device='cuda'):
        super().__init__()
        self.mha_a = SPF(embed_dim, num_heads, attn_dropout, res_dropout, input_dims['a'], stride, pf_hidden,
                         use_fast=use_fast, use_dense=use_dense, device=device)
        self.ffn_a = FFN(embed_dim, relu_dropout, res_dropout)
        self.mha_t = SPF(embed_dim, num_heads, attn_dropout, res_dropout, input_dims['t'], stride, pf_hidden,
                         use_fast=use_fast, use_dense=use_dense, device=device)
        self.ffn_t = FFN(embed_dim, relu_dropout, res_dropout)
        self.mha_v = SPF(embed_dim, num_heads, attn_dropout, res_dropout, input_dims['v'], stride, pf_hidden,
                         use_fast=use_fast, use_dense=use_dense, device=device)
        self.ffn_v = FFN(embed_dim, relu_dropout, res_dropout)

    def forward(self, a, t, v, h_a, h_t, h_v, shift_a=0, shift_t=0, shift_v=0):
        h_a = self.ffn_a(self.mha_a(h_a, a, a, shift_a))
        h_t = self.ffn_t(self.mha_t(h_t, t, t, shift_t))
        h_v = self.ffn_v(self.mha_v(h_v, v, v, shift_v))
        return h_a, h_t, h_v


def sum_fuse(x, y): return x + y


class SPSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()  # device --> actual weight; due to nn.DataParallel :-(
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, dim=None):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        # if device not in self.weights or max_pos > self.weights[device].size(0):
        #     # recompute/expand embeddings if needed
        self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
            max_pos,
            self.embedding_dim if dim is None else dim,
            self.padding_idx,
        )
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights[device].index_select(0, positions.contiguous().view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class SPMultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0., input_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = embed_dim if input_dim is None else input_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(embed_dim + 2 * self.input_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(embed_dim * 3))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, attn_mask=None, reverse=False):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if reverse:
            q = self.in_proj_k(query)
            k = self.in_proj_q(key)
            v = self.in_proj_v(key)
        else:
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        q = q * self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim, startb=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, endb=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=self.embed_dim + self.input_dim,
                             startb=self.embed_dim, endb=self.embed_dim * 2)

    def in_proj_v(self, key):
        return self._in_proj(key, start=self.embed_dim + self.input_dim, end=self.embed_dim + self.input_dim * 2,
                             startb=self.embed_dim * 2, endb=self.embed_dim * 3)

    def _in_proj(self, input, start=0, end=None, startb=0, endb=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        bias = bias[startb:endb]
        if bias.shape[0] != weight.shape[0]: weight = weight.reshape(-1, self.embed_dim * 2).T  # KV
        return F.linear(input, weight, bias)


class SPEncoderCrossLayer(nn.Module):
    def __init__(self, embed_dim, input_dims, num_heads=4, attn_dropout=0.1, relu_dropout=0.1,
                 res_dropout=0.1, pf_cross=None, use_fast=False, use_dense=False, device='cuda'):
        super().__init__()
        self.mha_at = SPF(embed_dim, num_heads, attn_dropout, res_dropout, pf=pf_cross, use_fast=use_fast,
                          use_dense=use_dense, device=device)
        self.mha_tv = SPF(embed_dim, num_heads, attn_dropout, res_dropout, pf=pf_cross, use_fast=use_fast,
                          use_dense=use_dense, device=device)
        self.mha_va = SPF(embed_dim, num_heads, attn_dropout, res_dropout, pf=pf_cross, use_fast=use_fast,
                          use_dense=use_dense, device=device)
        self.ffn_at = FFN(embed_dim, relu_dropout, res_dropout)
        self.ffn_tv = FFN(embed_dim, relu_dropout, res_dropout)
        self.ffn_va = FFN(embed_dim, relu_dropout, res_dropout)
        self.fuse_a = self.fuse_t = self.fuse_v = sum_fuse

    def forward(self, h_a, h_t, h_v, shift_a=0, shift_t=0, shift_v=0):
        h_at = self.ffn_at(self.mha_at(h_a, h_t, h_t, shift_a))
        h_tv = self.ffn_tv(self.mha_tv(h_t, h_v, h_v, shift_t))
        h_va = self.ffn_va(self.mha_va(h_v, h_a, h_a, shift_v))
        h_ta = self.ffn_at(self.mha_at(h_t, h_a, h_a, shift_t, True))
        h_vt = self.ffn_tv(self.mha_tv(h_v, h_t, h_t, shift_v, True))
        h_av = self.ffn_va(self.mha_va(h_a, h_v, h_v, shift_a, True))
        return self.fuse_a(h_at, h_av), self.fuse_t(h_ta, h_tv), self.fuse_v(h_va, h_vt)


class SPEncoderSelfLayer(nn.Module):
    def __init__(self, embed_dim, input_dims, num_heads=4, attn_dropout=0.1, relu_dropout=0.1,
                 res_dropout=0.1, pf_self=None, use_fast=False, use_dense=False, device='cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.mha_a = SPF(embed_dim, num_heads, attn_dropout, res_dropout, pf=pf_self, use_fast=use_fast,
                         use_dense=use_dense, device=device)
        self.ffn_a = FFN(embed_dim, relu_dropout, res_dropout)
        self.mha_t = SPF(embed_dim, num_heads, attn_dropout, res_dropout, pf=pf_self, use_fast=use_fast,
                         use_dense=use_dense, device=device)
        self.ffn_t = FFN(embed_dim, relu_dropout, res_dropout)
        self.mha_v = SPF(embed_dim, num_heads, attn_dropout, res_dropout, pf=pf_self, use_fast=use_fast,
                         use_dense=use_dense, device=device)
        self.ffn_v = FFN(embed_dim, relu_dropout, res_dropout)

    def forward(self, h_a, h_t, h_v, shift_a=0, shift_t=0, shift_v=0):
        h_a = self.ffn_a(self.mha_a(h_a, shift=shift_a))
        h_t = self.ffn_t(self.mha_t(h_t, shift=shift_t))
        h_v = self.ffn_v(self.mha_v(h_v, shift=shift_v))
        return h_a, h_t, h_v


def sparsify(hidden, context, stride, pf, shift=0, device='cuda'):
    h, bs, _ = hidden.shape
    c, _, dc = context.shape
    r = (torch.arange(h).to(device) + 1) * stride - pf + shift
    r = r.unsqueeze(0).repeat(pf, 1) + torch.arange(pf).unsqueeze(1).to(device)
    r = r.reshape([pf * h])
    return context[r % c].reshape(pf, h, bs, dc)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias: nn.init.constant_(m.bias, 0.)
    return m


def sparse_mask(hidden=None, context=None, stride=None,
                pf=None, h=None, c=None, cuda=None, shift=0):  # generate
    h = hidden.size(0) if h is None else h
    c = context.size(0) if c is None else c
    mask = torch.ones(h, c) * torch.tensor(float('-inf'))
    for i in range(pf):
        k = (torch.arange(h) + 1) * stride - pf + i + shift
        mask[torch.arange(h), k % c] = 0
    if cuda or (context is not None and context.is_cuda): mask = mask.cuda()
    return mask


def mode_one_khatri_rao(x, y):
    return torch.einsum('ia, ib -> iab', x, y)


def three_one_tensor_contraction(x, y):
    return torch.einsum('itab, ibc -> itac', x, y)


class MMTLayer(nn.Module):
    def __init__(self, text_dim, video_dim, audio_dim, r1, r2, r3, alpha, seq_len):
        super().__init__()
        self.Q_t_1 = nn.Linear(text_dim, r2, bias=False)
        self.Q_t_2 = nn.Linear(text_dim, r3, bias=False)
        self.Q_v_1 = nn.Linear(video_dim, r1, bias=False)
        self.Q_v_2 = nn.Linear(video_dim, r2, bias=False)
        self.Q_a_1 = nn.Linear(audio_dim, r3, bias=False)
        self.Q_a_2 = nn.Linear(audio_dim, r1, bias=False)

        self.K_t_1 = nn.Linear(text_dim, r2, bias=False)
        self.K_t_2 = nn.Linear(text_dim, r3, bias=False)
        self.K_v_1 = nn.Linear(video_dim, r1, bias=False)
        self.K_v_2 = nn.Linear(video_dim, r2, bias=False)
        self.K_a_1 = nn.Linear(audio_dim, r3, bias=False)
        self.K_a_2 = nn.Linear(audio_dim, r1, bias=False)

        self.linear_t = nn.Linear(r2 * r2, text_dim)
        self.linear_v = nn.Linear(r1 * r1, video_dim)
        self.linear_a = nn.Linear(r3 * r3, audio_dim)

        self.r1 = r1
        self.r2 = r2
        self.r3 = r3

        self.alpha = alpha
        self.seq_len = seq_len

    def forward(self, t, v, a):
        bz = t.shape[0]

        G_Q_t = mode_one_khatri_rao(self.Q_t_1(t).view(bz * self.seq_len, -1),
                                    self.Q_t_2(t).view(bz * self.seq_len, -1)).view(bz, self.seq_len,
                                                                                    self.r2,
                                                                                    self.r3)
        G_K_t = mode_one_khatri_rao(self.K_t_1(t).view(bz * self.seq_len, -1),
                                    self.K_t_2(t).view(bz * self.seq_len, -1)).view(bz, self.seq_len,
                                                                                    self.r2,
                                                                                    self.r3)
        G_Q_v = mode_one_khatri_rao(self.Q_v_1(v).view(bz * self.seq_len, -1),
                                    self.Q_v_2(v).view(bz * self.seq_len, -1)).view(bz, self.seq_len,
                                                                                    self.r1,
                                                                                    self.r2)
        G_K_v = mode_one_khatri_rao(self.K_v_1(v).view(bz * self.seq_len, -1),
                                    self.K_v_2(v).view(bz * self.seq_len, -1)).view(bz, self.seq_len,
                                                                                    self.r1,
                                                                                    self.r2)
        G_Q_a = mode_one_khatri_rao(self.Q_a_1(a).view(bz * self.seq_len, -1),
                                    self.Q_a_2(a).view(bz * self.seq_len, -1)).view(bz, self.seq_len,
                                                                                    self.r3,
                                                                                    self.r1)
        G_K_a = mode_one_khatri_rao(self.K_a_1(a).view(bz * self.seq_len, -1),
                                    self.K_a_2(a).view(bz * self.seq_len, -1)).view(bz, self.seq_len,
                                                                                    self.r3,
                                                                                    self.r1)

        G_QK_t = G_Q_t * G_K_t  # (B, T, R2, R3)
        G_QK_v = G_Q_v * G_K_v  # (B, T, R1, R2)
        G_QK_a = G_Q_a * G_K_a  # (B, T, R3, R1)

        M_QK_t = torch.mean(G_QK_t, dim=1)  # (B, R2, R3)
        M_QK_v = torch.mean(G_QK_v, dim=1)  # (B, R1, R2)
        M_QK_a = torch.mean(G_QK_a, dim=1)  # (B, R3, R1)

        attn_t = three_one_tensor_contraction(three_one_tensor_contraction(G_QK_t, M_QK_a), M_QK_v)  # (B, T, R2, R2)
        attn_v = three_one_tensor_contraction(three_one_tensor_contraction(G_QK_v, M_QK_t), M_QK_a)  # (B, T, R1, R1)
        attn_a = three_one_tensor_contraction(three_one_tensor_contraction(G_QK_a, M_QK_v), M_QK_t)  # (B, T, R3, R3)

        attn_t = self.linear_t(attn_t.view(bz, self.seq_len, -1))  # (B, T, d_t)
        attn_v = self.linear_v(attn_v.view(bz, self.seq_len, -1))  # (B, T, d_v)
        attn_a = self.linear_a(attn_a.view(bz, self.seq_len, -1))  # (B, T, d_a)

        y_t_aware = attn_t * t + self.alpha * t
        y_v_aware = attn_v * v + self.alpha * v
        y_a_aware = attn_a * a + self.alpha * a

        return y_t_aware, y_v_aware, y_a_aware
