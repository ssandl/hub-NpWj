"""
文本匹配模型定义

教学重点：
  1. BiEncoder（表示型）— 共享 BERT 骨干，对两句分别编码，计算余弦相似度
     对应 Sentence-BERT 论文中的 Siamese 架构
  2. CrossEncoder（交互型）— 两句拼接后整体送入 BERT，直接输出匹配概率
  3. L2 归一化 — encode() 输出归一化向量后，余弦相似度等价于点积（更高效）
  4. num_hidden_layers — 限制 BERT 层数加速训练（4 层约为全量的 1/3 时间）
     原理：从完整 12 层权重中只加载前 N 层，其余丢弃

使用方式：
  from model import BiEncoder, CrossEncoder, build_biencoder, build_crossencoder

依赖：
  pip install torch transformers
"""
import torch
from transformers import BertConfig, BertModel, BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
import transformers

class BiEncoder(nn.Module):
    def __init__(self, bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
        super().__init__()
        #校验参数
        assert pool in ("cls", "mean", "max"), f"pool 须为 cls/mean/max，收到: {pool}"
        #
        config = BertConfig.from_pretrained(bert_path)
        # 限制 BERT 层数加速训练
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        _prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity(transformers.logging.ERROR)
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers.logging.set_verbosity(_prev)

        self.pool = pool
        self.dropout = nn.Dropout(dropout)
    def encode(self, input_ids, attention_mask,token_type_ids):
        #送入共享BERT主干，返回模型输出字典
        #参数含义：input_ids - 输入文本的 token id, attention_mask - 对应 token 是否有效，token_type_ids - 句子标识,返回值：字典，包含各层输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids,return_dict=True)
        #调用内部池化函数，把每个句子变长token序列压缩成单个向量 [B, H]
        vec = self._pool(outputs.last_hidden_state, attention_mask)
        vac = self.dropout(vec)
        return F.normalize(vac, dim=-1, p=2) #L2归一化，余弦相似度等价于点积，更高效
    def _pool(self, last_hidden, attention_mask):
        #把 BERT 输出的 token 序列 [B, seq_len, H] 压缩成单个句子向量 [B, H]，提供三种主流方案
        if self.pool == "cls":
            return last_hidden[:, 0, :]

        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]

        if self.pool == "mean":
            sum_h = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            return sum_h / count

        if self.pool == "max":
            masked = last_hidden + (1 - mask) * (-1e9)
            return masked.max(dim=1).values

    def forward(self, batch_a, batch_b):
        """返回 (emb_a, emb_b)，各形状 [B, H]，可直接计算余弦相似度"""
        emb_a = self.encode(**batch_a)
        emb_b = self.encode(**batch_b)
        return emb_a, emb_b
    
#交互型
class CrossEncoder(nn.Module):
    """
    交互型文本匹配：Cross-Encoder

    结构：
      BertModel([CLS] s1 [SEP] s2 [SEP]) → CLS 向量 → Dropout → Linear(H, 2) → logits

    对比 BiEncoder：
      优点：两句在每一层都交互，表达能力更强，精度更高
      缺点：无法预计算向量，每对句子都要完整过 BERT，不适合大规模检索
      典型用途：Reranker（对召回的 Top-K 候选精排），即 rag_annual_report 中的做法

    参数：
      bert_path         : 预训练权重路径
      dropout           : 分类头 Dropout 比例
      num_hidden_layers : 同 BiEncoder，限层数加速
    """
    def __init__(self, bert_path, dropout=0.1, num_hidden_layers=None):
        super().__init__()
        config = BertConfig.from_pretrained(bert_path)
        # 限制 BERT 层数加速训练
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        _prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity(transformers.logging.ERROR)
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers.logging.set_verbosity(_prev)

        self.dropout = nn.Dropout(dropout)
        # print(f"test1{config.hidden_size}")
        # print(f"test2{self.bert.config.hidden_size}")
        #参数含义：config.hidden_size - BERT隐藏层维度，2 - 二分类
        self.classifier = nn.Linear(config.hidden_size, 2)
    def forward(self, input_ids, attention_mask,token_type_ids):
        """返回 logits [B, 2]，未经 softmax（CrossEntropyLoss 内部处理）"""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids,return_dict=True)
        cls_vec = out.last_hidden_state[:, 0, :]
        cls_vec = self.dropout(cls_vec)
        return self.classifier(cls_vec)
    
# ── 工厂函数 ──────────────────────────────────────────────────────────────

def build_biencoder(bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
    """构建 BiEncoder 并打印参数量。"""
    model = BiEncoder(bert_path, pool=pool, dropout=dropout,
                      num_hidden_layers=num_hidden_layers)
    _print_param_info(model, f"BiEncoder (pool={pool}, layers={num_hidden_layers or 12})")
    return model


def build_crossencoder(bert_path, dropout=0.1, num_hidden_layers=None):
    """构建 CrossEncoder 并打印参数量。"""
    model = CrossEncoder(bert_path, dropout=dropout,
                         num_hidden_layers=num_hidden_layers)
    _print_param_info(model, f"CrossEncoder (layers={num_hidden_layers or 12})")
    return model


def _print_param_info(model, name):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    bert  = sum(p.numel() for p in model.bert.parameters()) / 1e6
    print(f"模型: {name}")
    print(f"参数量: {total:.1f}M  (BERT 骨干: {bert:.1f}M)")
