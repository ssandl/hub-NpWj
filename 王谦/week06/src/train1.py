"""
BERT NER 训练脚本

教学重点：
  1. --use_crf 参数：一套脚本同时支持两种模型
  2. 分层学习率：BERT 层用 2e-5，分类头用 1e-4（加速头部收敛）
  3. Linear Warmup：防止训练初期大梯度破坏预训练参数
  4. seqeval 评估：entity-level F1（不是 token-level accuracy）

使用方式：
  python train.py                        # 训练 BERT+Linear（基线）
  python train.py --use_crf              # 训练 BERT+CRF
  python train.py --epochs 5 --lr 3e-5  # 自定义超参数

依赖：
  pip install torch transformers seqeval pytorch-crf tqdm
  export DASHSCOPE_API_KEY="sk-xxx"   （LLM对比时用）
"""
import os
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from dataset1  import build_label_schema, build_dataloaders
from model1 import BertForNER, BertForNERCRF, build_model

ROOT = Path(__file__).parent.parent
BERT_PATH = "/Users/wangqian/Downloads/java/八斗学院/AI训练营/2026直播/每周作业/week5/pretrain_models/bert-base-chinese" 
DATA_DIR = ROOT / "data" / "peoples_daily"
CKPT_DIR = ROOT / "outputs" / "checkpoints1"
LOG_DIR = ROOT / "outputs" / "logs1"

"""
概述：在给定的数据加载器上评估模型性能，计算平均损失和实体级别的F1分数。
参数：
model: 要评估的神经网络模型
loader: 数据加载器，提供用于评估的数据批次
id2label: 字典，将标签ID映射到标签名称
device: torch设备，指定计算设备
use_crf: 布尔值，指示模型是否使用CRF层
返回值：
返回一个元组，包含两个浮点数：
avg_loss: 在所有数据批次上的平均损失
entity_f1: 实体级别的F1分数
"""
def evaluate_epoch(
    model: nn.Module,
    loader,
    id2label: dict,
    device: torch.device,
    use_crf: bool,
) -> tuple[float, float]:
    """在 loader 上评估，返回 (avg_loss, entity_f1)。"""
    from seqeval.metrics import f1_score as seqeval_f1

    model.eval()
    total_loss = 0.0    # 累计损失
    all_preds: list[list[str]] = []  # 存储所有预测的标签序列
    all_golds: list[list[str]] = []  # 存储所有真实的标签序列
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                emissions, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            total_loss += loss.item()
            labels_np = labels.cpu().tolist()
            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                token_labels = labels_np[i]
                if use_crf:
                    pred_ids = pred_ids_list[i]
                else:
                    pred_ids = pred_ids_list[i]

                for j, gold_id in enumerate(token_labels):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    if use_crf:
                        if j < len(pred_ids):
                            pred_seq.append(id2label.get(pred_ids[j], "O"))
                        else:
                            pred_seq.append("O")
                    else:
                        pred_seq.append(id2label.get(pred_ids[j], "O"))

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    avg_loss = total_loss / len(loader)
    entity_f1 = seqeval_f1(all_golds, all_preds)
    return avg_loss, entity_f1

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_accum: int,
) -> float:
    model.train()
    total_loss = 0.0
    # 清零梯度
    optimizer.zero_grad()
    #用 tqdm 生成训练进度条，直观展示每个 epoch 的训练进度。
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)

    for step, batch in enumerate(pbar): # 遍历数据加载器中的每个批次
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        _,loss = model(input_ids, attention_mask, token_type_ids, labels)
        (loss / grad_accum).backward()
        total_loss += loss.item()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix_str(f"loss={loss.item():.4f}")
          # 处理最后不足 grad_accum 的批次
        remainder = len(loader) % grad_accum
        if remainder != 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / len(loader)

def parse_args():
    parser = argparse.ArgumentParser(description="训练 BERT NER 模型")
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层（否则使用线性头）")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头学习率倍数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()
def main():
    args = parse_args()