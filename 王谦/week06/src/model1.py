"""
BertNER（线性头）和 BertCRFNER（CRF头）两个模型

教学重点：
  1. 线性头（BertNER）：每个 token 独立预测标签
     - 问题：softmax 的独立预测忽略标签间的依赖关系
     - 可能产生非法序列：B-name 后接 I-company，I-name 开头等

  2. CRF 层（BertCRFNER）：加入转移矩阵，全局最优解码
     - 转移矩阵学习"什么标签之后可以接什么标签"
     - Viterbi 算法保证输出合法序列，永远不会 B-name 后接 I-company
     - 代价：训练时需要前向-后向算法，比线性头慢约 20~30%

  3. 两者区别的量化：evaluate.py 会统计非法序列数

依赖：
  pip install pytorch-crf
"""

import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
from pathlib import Path
import transformers


"""
概述：从指定路径加载预训练的BERT模型，并在加载过程中临时设置日志级别为错误以减少输出信息。
参数：
bert_path：字符串类型，表示预训练BERT模型的路径。
返回值：
返回一个BertModel类型的实例，表示加载好的预训练BERT模型。
"""
def _load_bert(bert_path: str) -> BertModel:
    prev = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    bert = BertModel.from_pretrained(bert_path)
    transformers.logging.set_verbosity(prev)
    return bert
class BertNER(nn.Module):
    """BERT + 线性分类头，逐 token 独立预测 BIO 标签。
    前向过程：
      input_ids → BertModel → last_hidden_state (B, L, 768)
               → Dropout → Linear(768, num_labels) → logits (B, L, num_labels)
    损失：CrossEntropy，ignore_index=-100 跳过特殊token和非首子词
    预测：argmax(logits, dim=-1)
    """
    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
    def forward(self, input_ids, attention_mask, token_type_ids,labels=None):
        #input_ids:输入学历的整数索引张量。形状为(Batch_size, Sequence_length)
        #attention_mask:一个整数张量，用于指示输入序列中哪些位置是填充的。形状为(Batch_size, Sequence_length)
        #token_type_ids:一个整数张量，用于指示输入序列中每个位置属于第一个句子还是第二个句子。形状为(Batch_size, Sequence_length)
        #labels:一个整数张量，表示每个位置的标签。形状为(Batch_size, Sequence_length)
        #return_dict:一个布尔值，表示是否返回一个字典，包含所有输出。默认为True。
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,return_dict=True)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)       
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))           
        return logits, loss
class BertCRFNER(nn.Module):
    """BERT + CRF 层，全局最优序列解码。

    与 BertNER 的区别：
      - Linear 输出称为 emissions（发射分数），不直接 argmax
      - CRF 在 emissions 上叠加转移矩阵，用 Viterbi 找全局最优序列
      - 损失：负对数似然（CRF 内部计算前向-后向）
      - 解码：self.crf.decode() 返回保证合法的标签序列

    CRF 的约束（自动学习）：
      - 初始只能以 O 或 B-X 开头
      - B-X 之后只能是 I-X 或 B-Y 或 O
      - I-X 之后只能是 I-X 或 B-Y 或 O
    """
    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels
    def _get_emissions(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        return emissions
    def forward(self, input_ids, attention_mask, token_type_ids,labels=None):
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)

        mask = attention_mask.bool()
        loss = None
        if labels is not None:
            labels_crf = labels.clone()
            labels_crf[labels == -100] = 0
            loss = -self.crf(emissions, labels_crf, mask=mask, reduction="mean")
        return emissions, loss
    def decode(self, input_ids, attention_mask, token_type_ids):
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask)
def build_model(use_crf: bool, bert_path: str, num_labels: int, dropout: float = 0.1) -> nn.Module:
    model_cls = BertCRFNER if use_crf else BertNER
    model = model_cls(bert_path=bert_path, num_labels=num_labels, dropout=dropout)
    #计算模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    #计算模型可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #模型名称
    model_name = "BERT + CRF" if use_crf else "BERT + Linear"
    print(f"模型：{model_name}")
    print(f"  标签数：{num_labels}")
    print(f"  参数总量：{total_params / 1e6:.1f}M")
    print(f"  可训练参数：{trainable_params / 1e6:.1f}M")
    return model