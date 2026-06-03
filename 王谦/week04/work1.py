"""
用pytorch实现transform 层
12层transformer encoder，每层12个head，hidden size 768

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden, n_head):
        super(MultiHeadAttention, self).__init__()
        self.hidden = hidden
        self.n_head = n_head
        self.d_k = hidden // n_head
        # 4 个线性层：Q K V 和输出
        self.linears = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(4)])
    def forward(self, x, mask=None):
        B, T, H = x.size()
        # 线性变换并分头 B T n_head d_k -> B n_head T d_k
        q, k, v = [l(x).view(B, T, self.n_head, self.d_k).transpose(1, 2) for l in self.linears[:3]]
        print(f"q.shape = {q.shape}, k.shape = {k.shape}, v.shape = {v.shape}")
        # 计算注意力得分 q * k.T / sqrt(d_k).  q.shape = [B, n_head, T, d_k], k.shape = [B, n_head, T, d_k], scores.shape = [B, n_head, T, T]   
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        print(f"k.transpose(-2, -1).shape = {k.transpose(-2, -1).shape}, scores.shape = {scores.shape}")
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        #v.shape = [B, n_head, T, d_k], attn.shape = [B, n_head, T, T], out.shape = [B, n_head, T, d_k]
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, H)
        print(f"attn.shape = {attn.shape}, out.shape = {out.shape}")
        return self.linears[3](out)
class EncoderLayer(nn.Module):
    def __init__(self, hidden, n_head):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(hidden, n_head)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.ReLU(),
            nn.Linear(hidden * 4, hidden)
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
    def forward(self, x, mask=None):
        # 多头注意力 + 残差连接 + LayerNorm
        attn_out = self.attn(x, mask)
        x = self.norm1(x + attn_out)
        # 前馈网络 + 残差连接 + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
class TransformerEncoder(nn.Module):
    def __init__(self, hidden=768, n_layer=12, n_head=12):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(hidden, n_head) for _ in range(n_layer)])
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
if __name__ == '__main__':
    x = torch.randn(2, 10, 768)
    model = TransformerEncoder()
    out = model(x)
    print(out.shape)