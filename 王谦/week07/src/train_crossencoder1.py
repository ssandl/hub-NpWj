"""
CrossEncoder 训练脚本（交互型文本匹配）

教学重点：
  1. CrossEncoder vs BiEncoder 的关键差异：
       CrossEncoder 让两句在 BERT 每一层都充分交互（Self-Attention 跨句），
       表达能力更强但无法预计算，不适合向量检索；
       BiEncoder 两句独立编码，可预计算向量，适合大规模检索（如 RAG Recall）
  2. 输入格式：[CLS] s1 [SEP] s2 [SEP]，token_type_ids 区分两段
       这是 BERT 原始预训练任务（NSP，Next Sentence Prediction）的格式
  3. CrossEncoder 评估与普通分类完全相同，无需阈值搜索（argmax 即为预测）

使用方式：
  # 默认参数（4 层 BERT，3 epoch）
  python train_crossencoder.py

  # 自定义参数
  python train_crossencoder.py --num_hidden_layers 6 --epochs 5 --batch_size 16

依赖：
  pip install torch transformers scikit-learn tqdm
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from dataset1 import build_crossencoder_loaders
from evaluate1 import eval_crossencoder
from model1 import build_crossencoder

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "bq_corpus"
BERT_PATH  = "/Users/wangqian/Downloads/java/八斗学院/AI训练营/2026直播/每周作业/week5/pretrain_models/bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints1"
LOG_DIR    = OUTPUT_DIR / "logs1"
#训练，参数含义：model:模型，loader:数据集，optimizer:优化器，scheduler:学习率调度器，criterion:损失函数，device:设备，epoch:当前轮数，total_epochs:总轮数，grad_accum:梯度累积
def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    device, epoch, total_epochs, grad_accum):
    model.train()
    optimizer.zero_grad()
    #含义：total_loss:总损失，total_correct:总正确数，total_samples:总样本数
    total_loss, total_correct, total_samples = 0.0, 0, 0
    #含义：tqdm:进度条
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{total_epochs}")
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, token_type_ids, attention_mask)
        loss = criterion(logits, labels)

        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        #含义：preds:预测结果
        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * len(labels)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)
        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return total_loss / total_samples, total_correct / total_samples

def parse_args():
    parser = argparse.ArgumentParser(description="CrossEncoder 训练（交互型文本匹配）")
    parser.add_argument("--bert_path",         default=str(BERT_PATH),   type=str)
    parser.add_argument("--data_dir",          default=str(DATA_DIR),    type=str)
    parser.add_argument("--num_hidden_layers", default=4,    type=int,
                        help="BERT Transformer 层数（默认 4 层；全量 12 层留给学生自行实验）")
    parser.add_argument("--epochs",            default=3,    type=int)
    parser.add_argument("--batch_size",        default=32,   type=int)
    parser.add_argument("--max_length",        default=128,  type=int,
                        help="句对总最大 token 数（两句拼接，建议 128）")
    parser.add_argument("--lr",                default=2e-5, type=float, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult",      default=5.0,  type=float, help="分类头学习率倍数")
    parser.add_argument("--warmup_ratio",      default=0.1,  type=float)
    parser.add_argument("--grad_accum",        default=10,    type=int)
    return parser.parse_args()

def main():
    args = parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"BERT 层数: {args.num_hidden_layers}  Epochs: {args.epochs}  "
          f"Batch size: {args.batch_size}")

    # ── Tokenizer & DataLoader ────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    print("\nDataLoader 构建中...")

    train_loader, dev_loader, test_loader = build_crossencoder_loaders(
        tokenizer=tokenizer, data_dir=args.data_dir, batch_size=args.batch_size, max_length=args.max_length
    )
    print("DataLoader 构建完成！")

    # ── Model & Optimizer ──────────────────────────────────────────────────
    model = build_crossencoder(
        bert_path=args.bert_path, num_hidden_layers=args.num_hidden_layers
    ).to(device)
    #分层学习率
    bert_params = list(model.bert.named_parameters())
    head_params = (list(model.classifier.named_parameters()) + list(model.dropout.named_parameters()))
    
    optimizer = AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ],weight_decay=0.01)
    #total_steps:总步数，warmup_steps:预热步数
    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    #学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"总训练步数: {total_steps}  Warmup 步数: {warmup_steps}")

    criterion = nn.CrossEntropyLoss()

       # ── 训练循环 ──────────────────────────────────────────────────────────
    ckpt_path   = CKPT_DIR / "crossencoder_best.pt"
    best_val_f1 = 0.0
    log_records = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.epochs, args.grad_accum,
        )

        val_metrics = eval_crossencoder(model, val_loader, device)
        elapsed = time.time() - t0

        val_acc = val_metrics["accuracy"]
        val_f1  = val_metrics["f1"]
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
              f"{elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "val_f1": val_f1, "elapsed_s": elapsed,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_acc":    val_acc,
                "val_f1":     val_f1,
                "args":       vars(args),
            }, ckpt_path)
            print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_f1={val_f1:.4f})")

    # ── 训练完成，保存日志 ────────────────────────────────────────────────
    log_path = LOG_DIR / "crossencoder_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"\n训练完成。最优 val_f1={best_val_f1:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"最优 checkpoint → {ckpt_path}")
    print(f"\n运行评估：python evaluate.py --model_type crossencoder --ckpt {ckpt_path}")


# ── 参数解析 ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CrossEncoder 训练（交互型文本匹配）")
    parser.add_argument("--bert_path",         default=str(BERT_PATH),   type=str)
    parser.add_argument("--data_dir",          default=str(DATA_DIR),    type=str)
    parser.add_argument("--num_hidden_layers", default=4,    type=int,
                        help="BERT Transformer 层数（默认 4 层；全量 12 层留给学生自行实验）")
    parser.add_argument("--epochs",            default=3,    type=int)
    parser.add_argument("--batch_size",        default=32,   type=int)
    parser.add_argument("--max_length",        default=128,  type=int,
                        help="句对总最大 token 数（两句拼接，建议 128）")
    parser.add_argument("--lr",                default=2e-5, type=float, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult",      default=5.0,  type=float, help="分类头学习率倍数")
    parser.add_argument("--warmup_ratio",      default=0.1,  type=float)
    parser.add_argument("--grad_accum",        default=1,    type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()

