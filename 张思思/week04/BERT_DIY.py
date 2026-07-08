import torch
import torch.nn as nn
import numpy as np

# 超参数
TRANSFORMER_COLS = 64
SEQ_LEN = 5
VOCAB_SIZE = 3000
EMBED_DIM = 768
LR = 0.01

# 模型
class MyDIYBERT(nn.Module):
    def __init__(self, vocab_size, embed_dim=768):
        super().__init__()
        self.token_embd = nn.Embedding(vocab_size, embed_dim)
        self.pos_embd = nn.Embedding(512, embed_dim)
        self.sent_embd = nn.Embedding(2, embed_dim)
        # 错误写法:self.q = nn.Linear(seq_len, embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.qkv_fc = nn.Linear(embed_dim, embed_dim)
        self.ffn_fc1 = nn.Linear(embed_dim, 3072)
        self.ffn_fc2 = nn.Linear(3072, embed_dim)
        self.head_num = (embed_dim // TRANSFORMER_COLS)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def transformer(self, Q, K, V):
        results = []
        for i in range(self.head_num):
            # 错误写法：k = K[i * TRANSFORMER_COLS: (i+1) * TRANSFORMER_COLS]
            q = Q[:, i * TRANSFORMER_COLS: (i+1) * TRANSFORMER_COLS]
            k = K[:,i * TRANSFORMER_COLS: (i+1) * TRANSFORMER_COLS]
            v = V[:,i * TRANSFORMER_COLS: (i+1) * TRANSFORMER_COLS]
            # print(f'q:{q.shape}, k:{k.shape}, v:{v.shape}')
            l = torch.softmax((q @ k.T) / np.sqrt(TRANSFORMER_COLS), dim=1)
            # 维度对齐错误：results.append(v @ l)
            results.append(l @ v)
        return torch.cat(results, dim=1)
    
    def forward(self, x):
        # 构造embedding输入层
        # print(f'x[0] shape:{x[0].shape} , {x[1].shape}, {x[2].shape}')
        token_emb = self.token_embd(x[0])
        pos_emb = self.pos_embd(x[1])
        sent_emb = self.token_embd(x[2])
        emb = token_emb + pos_emb + sent_emb
        # print(f'emb shape:{emb.shape}')

        # 构造Transformer
        # 错误写法：v = self.v @ emb
        q = self.q(emb)
        k = self.k(emb)
        v = self.v(emb)
        transform = self.transformer(q, k, v)
        y = self.qkv_fc(transform)
        # print(f'transformer y :{y.shape}')

        # 第一层残差+LayerNorm
        y = self.layer_norm(y + emb)
        # print(f'残差 y :{y.shape}')

        # 前馈层
        y = self.ffn_fc1(y)
        # print(f'ffn1 y :{y.shape}')
        y = self.ffn_fc2(y)
        # print(f'ffn2 y :{y.shape}')

        # 第二层残差+LayerNorm
        # print(f'残差 y :{y.shape}')
        # print(f'x :{x.shape}')
        # 错误写法：y = self.layer_norm(y + x)
        y = self.layer_norm(y + emb)
        return y

# 构造数据
# 句子的词典索引,句子的位置, 句子的分句
x = torch.tensor([[50, 20, 30, 10, 2],[0,1,2,3,4], [0,0,0,1,1]], dtype=torch.long)

model = MyDIYBERT(VOCAB_SIZE, EMBED_DIM)

y = model(x)
print(y)
