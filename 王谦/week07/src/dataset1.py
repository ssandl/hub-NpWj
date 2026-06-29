"""
文本匹配数据集类

教学重点：
  1. PairDataset — 句对数据集，用于 CosineEmbeddingLoss 训练和评估
  2. TripletDataset — 三元组数据集，用于 TripletLoss 训练
     离线构建方式：从正样本对出发，为每个 anchor 查找负样本
  3. CrossEncoderDataset — 交互型数据集，句对拼接为单序列送入 BERT

使用方式：
  from dataset import PairDataset, TripletDataset, CrossEncoderDataset, build_pair_loaders

依赖：
  pip install torch transformers
"""
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import random
random.seed(42)
#从制定路径加载数据集
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
'''
概述：该函数使用指定的tokenizer对输入文本进行编码，返回包含input_ids、attention_mask和token_type_ids的字典。

参数：

tokenizer：用于文本编码的tokenizer对象
text：需要编码的输入文本
max_length：编码的最大长度，超过此长度的文本会被截断
返回值：

一个包含以下键的字典：
input_ids：编码后的输入ID，形状为(max_length,)的张量
attention_mask：注意力掩码，形状为(max_length,)的张量
token_type_ids：令牌类型ID，形状为(max_length,)的张量
'''
def encode_text(tokenizer, text, max_length):
    #该函数只编码单个句子（sentence1 /anchor/positive 这种单文本，不处理句对），专门给 BiEncoder（PairDataset、TripletDataset）使用。
    #参数含义：text文本，max_length最大长度;truncation=True 表示如果文本长度超过 max_length，则截断文本；
    # padding="max_length" 表示将文本填充到 max_length 的长度；return_tensors="pt" 表示返回 PyTorch 张量。
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    #打印enc["input_ids"] 的形状
    # print(f"Input IDs shape: {enc['input_ids'].shape}")
    # print(f"Input IDs.squeeze(0) shape: {enc['input_ids'].squeeze(0).shape}") # 打印enc["input_ids"].squeeze(0) 的形状
    # print(f"Attention Mask shape: {enc['attention_mask'].shape}")
    # print(f"Token Type IDs shape: {enc['token_type_ids'].shape}")
    #tokenizer() 执行后返回 BatchEncoding，自带三个关键张量，初始 shape 都是 [1, max_length]（第 0 维是虚拟 batch 维度，因为只输入 1 条文本）
    return {
        "input_ids":      enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "token_type_ids": enc["token_type_ids"].squeeze(0),
    } 
'''
概述：这是一个用于处理句对数据集的类，用于训练和评估模型，支持余弦相似度计算和标签处理。每个样本包含两个句子和一个标签，数据从JSONL文件中加载。

参数：

data_path: JSONL文件路径，文件中包含sentence1、sentence2和label字段
tokenizer: HuggingFace的tokenizer对象，用于文本编码
max_length: 单句的最大token数，默认为64
返回值： 返回一个包含以下键的字典：

input_ids_a: 第一句的token IDs
attention_mask_a: 第一句的注意力掩码
token_type_ids_a: 第一句的token类型ID
input_ids_b: 第二句的token IDs
attention_mask_b: 第二句的注意力掩码
'''
class PairDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=64):
        #加载JSONL文件中的数据
        rows = load_jsonl(data_path)
        #将数据存储在self中
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows = rows

    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        #获取指定索引的样本
        row = self.rows[index]
        #对句子1进行编码
        enc_a = encode_text(self.tokenizer, row["sentence1"], self.max_length)
        #对句子2进行编码
        enc_b = encode_text(self.tokenizer, row["sentence2"], self.max_length)
        #返回编码后的结果和标签
        return {
            "input_ids_a":      enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "token_type_ids_a": enc_a["token_type_ids"],
            "input_ids_b":      enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
            "token_type_ids_b": enc_b["token_type_ids"],
            "label": torch.tensor(row["label"], dtype=torch.long),
        }
#TripletDataset
class TripletDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=64):
        #将数据存储在self中
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.triplets = self._build_triplets(load_jsonl(data_path))
    def _build_triplets(self, rows):
        # 1. defaultdict：key=句子，value=该句子对应的所有负样本文本列表
        neg_by_sent = defaultdict(list)
        all_sents = set() # 2. 全局句子池：收集全部句子
        #第一次遍历全部数据：建立【句子→自身负样本】索引 + 收集全部句子
        for row in rows:
            all_sents.append(row["sentence1"])
            all_sents.append(row["sentence2"])
            if row["label"] == 0:
                neg_by_sent[row["sentence1"]].append(row["sentence2"])
                neg_by_sent[row["sentence2"]].append(row["sentence1"])
        #把全部句子集合转为列表，作为兜底全局句子池
        global_pool = list(all_sents)

        triplets = []
        #第二次遍历全部数据：只拿正样本对构建三元组
        for row in rows:
            if row["label"] == 1:
                anchor = row["sentence1"]
                positive = row["sentence2"]
                #从自身负样本列表中随机选一个负样本
                neg_candidates = neg_by_sent.get(anchor, [])
                if neg_candidates:
                    negative = random.choice(neg_candidates)
                else:
                    # 该锚句没有匹配过的负样本，走全局随机兜底采样
                    negative = anchor
                    # 循环保证：负样本不能等于anchor、也不能等于positive
                    while negative in (anchor, positive):
                        negative = random.choice(global_pool)

                # 组装三元组存入列表
            triplets.append((anchor, positive, negative))
        print(f"  TripletDataset: 构建 {len(triplets):,} 个三元组")
        return triplets
    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        anchor, positive, negative = self.triplets[index]
        enc_a = encode_text(self.tokenizer, anchor, self.max_length)
        enc_p = encode_text(self.tokenizer, positive, self.max_length)
        enc_n = encode_text(self.tokenizer, negative, self.max_length)
        return {
            "input_ids_a":      enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "token_type_ids_a": enc_a["token_type_ids"],
            "input_ids_p":      enc_p["input_ids"],
            "attention_mask_p": enc_p["attention_mask"],
            "token_type_ids_p": enc_p["token_type_ids"],
            "input_ids_n":      enc_n["input_ids"],
            "attention_mask_n": enc_n["attention_mask"],
            "token_type_ids_n": enc_n["token_type_ids"],
        }
#CrossEncoderDataset
class CrossEncoderDataset(Dataset):
    """
    交互型数据集：sentence1 与 sentence2 拼接为单序列

    BERT tokenizer 自动生成：
      [CLS] sentence1 [SEP] sentence2 [SEP]
      token_type_ids: 0000...0  1111...1

    教学对比：
      相比 BiEncoder（两路独立编码），CrossEncoder 让两句在每一层都交互，
      表达能力更强，但无法预计算句向量，推理时每对都要过一次 BERT。

    参数：
      data_path  : JSONL 文件路径
      tokenizer  : HuggingFace tokenizer
      max_length : 句对总最大 token 数（两句拼接后）
    """
    def __init__(self, data_path, tokenizer, max_length=64):
        #加载JSONL文件中的数据
        rows = load_jsonl(data_path)
        #将数据存储在self中
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        #获取指定索引的样本
        row = self.rows[index]
        #对句子1进行编码
        enc = self.tokenizer(
            row["sentence1"],
            row["sentence2"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long),
        }
    
#工程函数
def build_pair_loaders(data_dir, tokenizer, max_length=64, batch_size=32):
    """
    构建 PairDataset 的 DataLoader 工厂函数

    参数：
      train_path : 训练集 JSONL 文件路径
      dev_path   : 验证集 JSONL 文件路径
      tokenizer  : HuggingFace tokenizer
      max_length : 单句最大 token 数
      batch_size : 批大小
    返回值：
      train_loader : 训练集 DataLoader  
      dev_loader   : 验证集 DataLoader
    """
    data_dir = Path(data_dir)
    train_ds = PairDataset(data_dir / "train.jsonl",      tokenizer, max_length)
    val_ds   = PairDataset(data_dir / "validation.jsonl", tokenizer, max_length)
    test_ds  = PairDataset(data_dir / "test.jsonl",       tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  train : {len(train_ds):>7,} 条, {len(train_loader):>5} batch")
    print(f"  val   : {len(val_ds):>7,} 条, {len(val_loader):>5} batch")
    print(f"  test  : {len(test_ds):>7,} 条, {len(test_loader):>5} batch  (AFQMC test 无正样本，仅供参考)")
    return train_loader, val_loader, test_loader
def build_triplet_loader(data_dir, tokenizer, max_length=64, batch_size=32):
    """为 TripletLoss 训练构建 DataLoader，val/test 仍用 PairDataset。"""
    data_dir = Path(data_dir)
    train_ds = TripletDataset(data_dir / "train.jsonl", tokenizer, max_length)
    val_ds   = PairDataset(data_dir / "validation.jsonl", tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  triplet train : {len(train_ds):>7,} 三元组, {len(train_loader):>5} batch")
    print(f"  val (pair)    : {len(val_ds):>7,} 对,     {len(val_loader):>5} batch")
    return train_loader, val_loader


def build_crossencoder_loaders(data_dir, tokenizer, max_length=128, batch_size=32):
    """为 CrossEncoder 构建 train/val/test DataLoader。"""
    data_dir = Path(data_dir)
    train_ds = CrossEncoderDataset(data_dir / "train.jsonl",      tokenizer, max_length)
    val_ds   = CrossEncoderDataset(data_dir / "validation.jsonl", tokenizer, max_length)
    test_ds  = CrossEncoderDataset(data_dir / "test.jsonl",       tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  train : {len(train_ds):>7,} 条, {len(train_loader):>5} batch")
    print(f"  val   : {len(val_ds):>7,} 条, {len(val_loader):>5} batch")
    print(f"  test  : {len(test_ds):>7,} 条, {len(test_loader):>5} batch")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from transformers import AutoTokenizer
    BERT_PATH  = "/Users/wangqian/Downloads/java/八斗学院/AI训练营/2026直播/每周作业/week5/pretrain_models/bert-base-chinese"
    ROOT       = Path(__file__).parent.parent
    DATA_DIR   = ROOT / "data" / "bq_corpus"
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    train_loader, val_loader, test_loader = build_pair_loaders(str(DATA_DIR), tokenizer, max_length=64, batch_size=32)
    #测试__getitem__方法
    for batch in train_loader:
        print(batch)
        break