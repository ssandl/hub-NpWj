"""
BiEncoder 训练脚本（表示型文本匹配，Sentence-BERT 架构）

教学重点：
  1. CosineEmbeddingLoss — 直接优化余弦相似度：
       正例对 (label=1) → cosine_sim 趋向 +1
       负例对 (label=0) → cosine_sim 低于 margin（默认 0.3），超出才计入 loss
  2. TripletLoss — 约束三角关系：
       sim(anchor, positive) > sim(anchor, negative) + margin
       无需标签，只需正/负样本的相对关系
  3. 评估时的阈值搜索 — BiEncoder 输出连续相似度，需在 val 上搜最优分类阈值
  4. num_hidden_layers 限层 — 4 层约为全量的 1/3 时间，适合课堂快速演示

Loss 对比总结（供学生参考）：
  CosineEmbeddingLoss：
    - 直接用已有 (s1, s2, label) 对，无需额外构造
    - 负样本到一定距离后梯度归零（margin 起到边界作用）
  TripletLoss：
    - 需构造 (anchor, positive, negative) 三元组
    - 更明确地告诉模型"相对远近"关系，适合检索/排序场景
    - 负样本质量影响训练效果（随机 vs 难负样本）

使用方式：
  # CosineEmbeddingLoss（默认）
  python train_biencoder.py

  # TripletLoss
  python train_biencoder.py --loss triplet

  # 自定义参数
  python train_biencoder.py --loss cosine --pool mean --num_hidden_layers 4 --epochs 3

依赖：
  pip install torch transformers scikit-learn tqdm
"""
import json
import time
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

from dataset1 import build_pair_loaders,build_crossencoder_loaders, build_triplet_loader
from model1 import build_biencoder
from evaluate1 import eval_biencoder


# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "bq_corpus"
BERT_PATH  = "/Users/wangqian/Downloads/java/八斗学院/AI训练营/2026直播/每周作业/week5/pretrain_models/bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints1"
LOG_DIR    = OUTPUT_DIR / "logs1"

#训练一个 epoch 
#参数含义:模型，数据加载器，优化器，学习率调度器，设备，当前 epoch，总 epoch 数，margin，梯度累积步数
def train_one_epoch_cosine(model,loader,optimizer,scheduler,device,epoch,total_epochs,margin,grad_accum):
    model.train()
    #累加所有样本的总损失，总样本数
    total_loss ,total_samples = 0.0,0
    optimizer.zero_grad()
    #tqdm 生成可视化进度条：
    pbar = tqdm(loader,desc=f"Epoch {epoch+1}/{total_epochs}[CosineEmbeddingLoss]]",leave = False,unit="batch")
    for step, batch in enumerate(pbar):
        batch_a = {
            "input_ids":      batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_b = {
            "input_ids":      batch["input_ids_b"].to(device),
            "attention_mask": batch["attention_mask_b"].to(device),
            "token_type_ids": batch["token_type_ids_b"].to(device),
        }
        label = batch["label"].to(device)
        #模型输出
        emb_a,emb_b = model(batch_a,batch_b)
        #关键标签转换
        cos_target = (label * 2) - 1
        #计算损失
        loss = F.cosine_embedding_loss(emb_a,emb_b,cos_target,margin=margin)
        (loss/grad_accum).backward()
        if(step +1) % grad_accum == 0:
            #限制模型所有参数梯度的最大范数为 1，防止训练出现梯度爆炸。
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        print(f"labes长度：{len(label)}，“label.size(0)”长度：{label.size(0)}")
        total_loss += loss.item() * label.size(0)
        total_samples += label.size(0)
        #
        pbar.set_postfix_str(f"loss: {loss.item():.4f}")
    return total_loss / total_samples

def train_one_epoch_triplet(model,loader,optimizer,scheduler,device,epoch,total_epochs,margin,grad_accum):
    model.train()
    total_loss ,total_samples = 0.0,0
    optimizer.zero_grad()
    pbar = tqdm(loader,desc=f"Epoch {epoch+1}/{total_epochs}[TripletLoss]]",leave = False,unit="batch")
    #这个遍历的dataset 中对应方法中的loader
    for step, batch in enumerate(pbar):
        enc_a = {
            "input_ids": batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        } 
        enc_p = {
            "input_ids": batch["input_ids_p"].to(device),
            "attention_mask": batch["attention_mask_p"].to(device),
            "token_type_ids": batch["token_type_ids_p"].to(device),
        }
        enc_n = {
            "input_ids": batch["input_ids_n"].to(device),
            "attention_mask": batch["attention_mask_n"].to(device),
            "token_type_ids": batch["token_type_ids_n"].to(device),
        }
        #模型输出 **enc_a：字典解包语法
        emb_a = model.encode(**enc_a)
        emb_p = model.encode(**enc_p)
        emb_n = model.encode(**enc_n)
        # triplet_margin_loss 默认用欧氏距离；
        # 由于 encode() 已 L2 归一化，向量在单位球上，欧氏距离与余弦距离单调相关
        loss = F.triplet_margin_loss(emb_a,emb_p,emb_n,margin=margin)
        (loss/grad_accum).backward()
        if(step +1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        bs = emb_a.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        pbar.set_postfix_str(f"loss: {loss.item():.4f}")
    return total_loss / total_samples
#参数
def parse_args():
    parser = argparse.ArgumentParser(description="BiEncoder 训练（表示型文本匹配）")
    parser.add_argument("--bert_path",         default=str(BERT_PATH),   type=str)
    parser.add_argument("--data_dir",          default=str(DATA_DIR),    type=str)
    parser.add_argument("--loss",              default="cosine",
                        choices=["cosine", "triplet"],
                        help="训练损失类型：cosine（CosineEmbeddingLoss）或 triplet（TripletLoss）")
    parser.add_argument("--pool",              default="mean",
                        choices=["cls", "mean", "max"],
                        help="句向量池化策略（Sentence-BERT 论文推荐 mean）")
    parser.add_argument("--num_hidden_layers", default=4,    type=int,
                        help="BERT Transformer 层数（默认 4 层快速验证；全量 12 层留给学生）")
    parser.add_argument("--epochs",            default=50,    type=int)
    parser.add_argument("--batch_size",        default=32,   type=int)
    parser.add_argument("--max_length",        default=48,   type=int,  help="单句最大 token 数")
    parser.add_argument("--lr",                default=2e-5, type=float, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult",      default=5.0,  type=float, help="dropout 层学习率倍数")
    parser.add_argument("--warmup_ratio",      default=0.1,  type=float)
    parser.add_argument("--grad_accum",        default=50,    type=int)
    parser.add_argument("--margin",            default=0.3,  type=float,
                        help="CosineEmbeddingLoss 的 margin；TripletLoss 的 margin 同参数")
    return parser.parse_args()

#主函数
def main():
    args = parse_args()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"Loss 类型: {args.loss}  池化策略: {args.pool}  "
          f"BERT 层数: {args.num_hidden_layers}  Epochs: {args.epochs}")
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    print("\nDataLoader 构建中...")
    if args.loss == "cosine":
        train_loader, val_loader, test_loader = build_pair_loaders(args.data_dir, tokenizer, args.batch_size, args.max_length)
    elif args.loss == "triplet":
        train_loader, val_loader = build_triplet_loader(args.data_dir, tokenizer, args.batch_size, args.max_length)
    #构建模型
    print("\n模型构建中...")
    model = build_biencoder(
        bert_path=args.bert_path,
        pool=args.pool,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)
    # ── 分层学习率 ────────────────────────────────────────────────────────
    # BERT 骨干用较小 lr，防止预训练知识被过度破坏
    bert_params = list(model.bert.parameters())
    head_params = list(model.dropout.parameters())
    optimizer = AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)
    #计算全程总更新步数 total_steps
    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    #计算预热步数 warmup_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    #创建线性预热 + 线性衰减的学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"总训练步数: {total_steps}  Warmup 步数: {warmup_steps}")
    #训练
    # ── 训练循环 ──────────────────────────────────────────────────────────
    ckpt_name = f"biencoder_{args.loss}_best.pt"
    ckpt_path = CKPT_DIR / ckpt_name
    best_val_f1 = 0.0
    log_records = []
    print("\n训练开始...")
    for epoch in range(1,args.epochs+1):
        t0 = time.time()
        if args.loss == "cosine":
            train_loss = train_one_epoch_cosine(
                model, train_loader, optimizer, scheduler, device,
                epoch, args.epochs, args.margin, args.grad_accum,
            )
        else:
            train_loss = train_one_epoch_triplet(
                model, train_loader, optimizer, scheduler, device,
                epoch, args.epochs, args.margin, args.grad_accum,
            )
        # 每个 epoch 末评估 val（BiEncoder：相似度 + 阈值搜索）
        val_metrics = eval_biencoder(model, val_loader, device)
        elapsed = time.time() - t0

        val_acc = val_metrics["accuracy"]
        val_f1  = val_metrics["f1"]
        val_thr = val_metrics["threshold"]
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} threshold={val_thr:.2f} | "
              f"{elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_acc": val_acc, "val_f1": val_f1,
            "threshold": val_thr, "elapsed_s": elapsed,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "threshold":  val_thr,
                "val_acc":    val_acc,
                "val_f1":     val_f1,
                "args":       vars(args),
            }, ckpt_path)
            print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_f1={val_f1:.4f})")

    # ── 训练完成，保存日志 ────────────────────────────────────────────────
    log_path = LOG_DIR / f"biencoder_{args.loss}_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"\n训练完成。最优 val_f1={best_val_f1:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"最优 checkpoint → {ckpt_path}")
    print(f"\n运行评估：python evaluate.py --model_type biencoder --ckpt {ckpt_path}")


# ── 参数解析 ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="BiEncoder 训练（表示型文本匹配）")
    parser.add_argument("--bert_path",         default=str(BERT_PATH),   type=str)
    parser.add_argument("--data_dir",          default=str(DATA_DIR),    type=str)
    parser.add_argument("--loss",              default="cosine",
                        choices=["cosine", "triplet"],
                        help="训练损失类型：cosine（CosineEmbeddingLoss）或 triplet（TripletLoss）")
    parser.add_argument("--pool",              default="mean",
                        choices=["cls", "mean", "max"],
                        help="句向量池化策略（Sentence-BERT 论文推荐 mean）")
    parser.add_argument("--num_hidden_layers", default=4,    type=int,
                        help="BERT Transformer 层数（默认 4 层快速验证；全量 12 层留给学生）")
    parser.add_argument("--epochs",            default=10,    type=int)
    parser.add_argument("--batch_size",        default=32,   type=int)
    parser.add_argument("--max_length",        default=96,   type=int,  help="单句最大 token 数")
    parser.add_argument("--lr",                default=2e-5, type=float, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult",      default=5.0,  type=float, help="dropout 层学习率倍数")
    parser.add_argument("--warmup_ratio",      default=0.1,  type=float)
    parser.add_argument("--grad_accum",        default=1,    type=int)
    parser.add_argument("--margin",            default=0.3,  type=float,
                        help="CosineEmbeddingLoss 的 margin；TripletLoss 的 margin 同参数")
    return parser.parse_args()


if __name__ == "__main__":
    main()
