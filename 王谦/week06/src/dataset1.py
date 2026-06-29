"""
NER 数据集类：span 标注→BIO 转换 + BERT 子词对齐

教学重点：
  1. cluener2020 的 span 格式转为 BIO 格式
     - span: {"name": {"叶老桂": [[9, 11]]}}
     - BIO:  ['O','O',...,'B-name','I-name','I-name',...]
  2. BERT 子词对齐（word_ids 策略）
     - 中文字符通常一字一token，但 [UNK] 和特殊字符可能例外
     - 非首子词标记为 -100，在 loss 计算中被忽略
  3. DataLoader 工厂函数统一封装

使用方式：
  from dataset import build_label_schema, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "peoples_daily"

ENTITY_TYPES = ["LOC", "PER", "ORG"]

def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 BIO 标签体系，返回 (labels, label2id, id2label)。"""
    labels = ["O"]
    for etype in ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label

# a,b,c = build_label_schema()
# print(f"labels: {a}\nlabel2id: {b}\nid2label: {c}")

class PeoplesDailyDataset(Dataset):
    """peoples_daily 的 PyTorch Dataset。
    
    数据集已提供 tokens 和 ner_tags，直接进行 BERT 子词对齐即可。
    """
    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizer,
        label2id: dict,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        tokens: list = row["tokens"]
        ner_tags: list = row["ner_tags"]

        # 1. 将字符级别的 ner_tags 转换为对应的 id 列表
        char_labels = [self.label2id.get(tag, 0) for tag in ner_tags]

        # 2. 将文本拆为字符列表，传入 tokenizer
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. 子词对齐：取每个 token 对应的字符索引
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                # 特殊字符（如 [CLS], [SEP], [PAD]）标记为 -100
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                # 首次出现这个字符索引：使用 BIO 标签
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                # 同一字符的后续子词标记为 -100，在 loss 计算中被忽略
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }

def load_records(split: str, data_dir: Optional[Path] = None) -> list: 
    """加载指定 split 的数据集，返回 records 列表。"""
    if data_dir is None:
        data_dir = DATA_DIR

    with open(data_dir / f"{split}.json", "r", encoding="utf-8") as f:
        records = json.load(f)
    return records

def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。"""
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    train_dataset = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    val_dataset = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    test_dataset = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

#test build_dataloaders
BERT_PATH = "/Users/wangqian/Downloads/java/八斗学院/AI训练营/2026直播/每周作业/week5/pretrain_models/bert-base-chinese" 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"设备：{device}")
# # 标签体系
# labels, label2id, id2label = build_label_schema()
# num_labels = len(labels)
# print(f"BIO 标签数：{num_labels}（O + {len(labels) - 1} 个实体标签）")

# # Tokenizer
# tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

# # DataLoader
# train_loader, val_loader, _ = build_dataloaders(
#     tokenizer=tokenizer,
#     label2id=label2id,
#     batch_size=32,
#     max_length=128,
#     data_dir=DATA_DIR,
# )
if __name__ == "__main__":
    # 1. 测试标签字典构建
    labels, label2id, id2label = build_label_schema()
    print(f"标签列表: {labels}")
    print(f"标签数量: {len(labels)}")
    print(f"标签到ID映射: {label2id}\n")

    # 2. 准备 Tokenizer (请将路径替换为您本地实际的 BERT 模型路径)
    # BERT_PATH = "bert-base-chinese" 
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    
    print(f"Tokenizer: {tokenizer.vocab_size} 个词汇")
    # 3. 测试 __getitem__ (单条数据测试)
    print("--- 测试单条数据获取 ---")
    train_records = load_records("train")
    if train_records:
        single_dataset = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length=128)
        sample = single_dataset[0]  # 触发 __getitem__
        for key, value in sample.items():
            print(f"{key} 形状: {value.shape}, 数据类型: {value.dtype}")
    else:
        print("训练数据加载失败，请检查文件路径。")

    # 4. 测试 build_dataloaders (Batch 数据测试)
    print("\n--- 测试 DataLoader Batch 输出 ---")
    train_loader, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=2,
        max_length=128,
    )
    # 取出一个 Batch 查看形状
    for batch in train_loader:
        print("Batch 数据形状:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")
        break  # 只取一个 Batch 观察即可
