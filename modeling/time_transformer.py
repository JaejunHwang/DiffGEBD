import math

import einops
from einops import rearrange
import copy
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
from typing import Optional
from .resnet import ResNetXX

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (T, C)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (T, 1, C)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x
    
class PositionalEncodingLearned(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=8):
        super(PositionalEncodingLearned, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embed = nn.Embedding(max_len, dim)
        
    def forward(self, x):
        out = x + self.embed(torch.arange(x.size(0), device=x.device)).unsqueeze(1) # L B C
        out = self.dropout(out)
        return out
    
    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings).cuda()
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                time = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, time=time)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        time_dim = d_model * 4
        self.block_time_mlp = nn.Sequential(
                                       nn.SiLU(),
                                       nn.Linear(time_dim,d_model*2))

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                time=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        scale_shift = self.block_time_mlp(time).unsqueeze(0) # 1 B C
        scale_shift = torch.repeat_interleave(scale_shift, src.shape[0], dim=0)
        scale, shift = scale_shift.chunk(2, dim=2)
        
        src = src * (scale + 1) + shift

        return src

class DiffDecoder(Module):
    def __init__(self, cfg):
        super().__init__()
        
        dim = cfg.MODEL.DIMENSION
        num_layers = cfg.MODEL.NUM_LAYERS
        nhead = cfg.MODEL.NUM_HEADS
        dim_feedforward = cfg.MODEL.FEED_FORWARD_DIM
        class_dim = cfg.MODEL.CLASS_DIM

        time_dim = dim * 4
        self.position_encoding = PositionalEncoding(dim)
        
        encoder_layer = TransformerEncoderLayer(dim, nhead, dim_feedforward)
        encoder_norm = nn.LayerNorm(dim)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        
        self.label_emb = nn.Sequential(nn.Conv1d(class_dim, dim, kernel_size=1),
                                       nn.GELU(),
                                       nn.Conv1d(dim, dim, kernel_size=5, padding=2))
        
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(dim),
                                       nn.Linear(dim,time_dim),
                                       nn.GELU(),
                                       nn.Linear(time_dim,time_dim))
        
        self.transform = nn.Linear(dim*2, dim)
        self.classifier = nn.Conv1d(dim, 1, 1)
        
    #     self._reset_parameters()
        
    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)
                
    def forward(self, x, cond, t):
        '''
        x: diffused gt [B, 1,T]
        cond: video feature [T, B, C]
        t: time [B]   
        '''
        T, B, C = cond.shape
        
        x = self.label_emb(x)
        x = rearrange(x, 'b c t -> t b c')
        time = self.time_mlp(t)
        
        x = torch.cat((x, cond), dim=2)
        x = self.transform(x)
        
        x = self.position_encoding(x)
        
        out = self.encoder(src=x, time=time) # L B C
        
        logits = self.classifier(out.permute(1,2,0))
        return logits