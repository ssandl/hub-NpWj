"""
GPT 预训练主循环

教学重点：
  1. 自回归语言模型的训练目标：预测下一个 token，交叉熵 loss
     input  = tokens[0..T-1]，target = tokens[1..T]（右移一位）
  2. PPL（困惑度）= exp(avg_cross_entropy_loss)
     PPL 越低 → 模型对验证集越"不困惑" → 语言建模能力越强
  3. AdamW + 余弦学习率调度：预训练标准配置
  4. Gradient Clipping：防止梯度爆炸，预训练必备
  5. Checkpoint 保存策略：每 epoch 保存，同时记录最优 val PPL

使用方式：
  python train.py                        # 默认配置
  python train.py --epochs 5             # 训练轮数
  python train.py --batch_size 16        # 显存不足时减小
  python train.py --lr 3e-4              # 学习率

依赖：
  pip install torch transformers
"""
import os
import math
import torch
import torch.nn as nn
import argparse
import logging
import json


from torch.utils.data import DataLoader,Dataset
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from modle import build_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_PATH = OUTPUT_DIR / "training_log.jsonl"

class TokenDataset(Dataset):
    """
    概述：这是一个用于从.pt文件加载token数据集的类。它将每个样本处理为一个长度为seq_len+1的token序列，并将其分割为输入序列和目标序列。
    输入序列是样本的前seq_len个token，目标序列是样本的后seq_len个token。
    参数：pt_path (Path) - 包含token数据集的.pt文件的路径。
    返回值：
    __len__方法返回数据集中样本的数量。
    __getitem__方法返回一个元组，包含两个序列：输入序列（样本的前seq_len个token）和目标序列（样本的后seq_len个token）。
    """
    def __init__(self, pt_path: Path):
        #weights_only 是 PyTorch 中 torch.load() 函数的一个安全参数，主要作用是防止反序列化攻击，提升加载模型或数据时的安全性。当 weights_only=True 时，只会加载模型参数，而不会加载模型结构。这可以防止攻击者通过加载恶意构造的模型文件来执行恶意代码。
        # ckpt = torch.load(pt_path, weights_only=False)
         # 关闭 weights_only 以正确加载包含字典结构的完整数据，消除索引警告
        ckpt = torch.load(pt_path, weights_only=False)
        
        # 防御性检查：确保 ckpt 是字典且包含所需的键
        if not isinstance(ckpt, dict) or 'data' not in ckpt:
            raise ValueError(f"加载的文件 {pt_path} 格式不正确，未找到 'data' 键。实际类型: {type(ckpt)}")
          
        self.data = ckpt['data']
        self.seq_len = ckpt['seq_len']
        self.vocab_size = ckpt['vocab_size']
        logger.info(f"Loaded dataset from {pt_path}, num_samples={len(self.data)}, seq_len={self.seq_len}, vocab_size={self.vocab_size}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = self.data[idx]  # 长度为 seq_len + 1 的 token 序列
        # x = sample[:self.seq_len]  # 输入序列：前 seq_len 个 token
        # y = sample[1:self.seq_len+1]  # 目标序列：后 seq_len 个 token（右移一位）
        # return x, y
         # 核心修复：强制取连续 seq_len+1 个token，绝对是1维张量
        chunk = self.data[idx : idx + self.seq_len + 1]
        
        # 双重保险：如果长度不够，直接跳过（防止极端情况）
        if len(chunk) < self.seq_len + 1:
            return self.__getitem__(0)
        
        # 输入x=前256个，标签y=后256个
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
    
'''
概述：计算给定模型在验证集上的困惑度（Perplexity, PPL）。困惑度是语言模型性能评估的重要指标，表示模型对验证数据的平均预测难度。
参数：
model：要评估的神经网络模型
loader：提供验证数据的DataLoader
device：计算设备（CPU或GPU）
返回值：浮点数，表示模型在验证集上的困惑度值
'''
def compute_ppl(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()  # 切换到评估模式
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 假设0是padding token ID

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)  # (B, T, vocab_size)
            B, T, V = logits.shape
            loss = criterion(logits.view(B*T, V), y.view(B*T))  # 展平后计算交叉熵
            total_loss += loss.item() * (y != 0).sum().item()  # 累加非padding token的损失
            total_tokens += (y != 0).sum().item()  # 累加非padding token的数量

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_loss) if avg_loss < float('inf') else float('inf')
    return ppl

'''
概述：该函数用于训练模型，包括设置训练参数、数据加载、模型训练和验证、模型保存等过程。
参数：
epochs: 训练轮数，默认为3
batch_size: 批次大小，默认为32
lr: 学习率，默认为3e-4
weight_decay: 权重衰减，默认为0.1
grad_clip: 梯度裁剪阈值，默认为1.0
seq_len: 序列长度，默认为256
num_workers: 数据加载的工作进程数，默认为0
返回值：无返回值
'''
def train(epochs=3, batch_size=32, lr=3e-4, weight_decay=0.1, grad_clip=1.0, seq_len=256, num_workers=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 加载数据集
    train_dataset = TokenDataset(DATA_DIR / 'train_data.pt')
    val_dataset = TokenDataset(DATA_DIR / 'val_data.pt')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    vocab_size = train_dataset.vocab_size
    # 构建模型（seq_len 使用数据集中实际的值，避免位置编码越界）
    config = {
        'vocab_size': vocab_size,
        'seq_len': seq_len,
        'd_model': 384,
        'n_heads': 6,
        'n_layers': 6,
        'd_ff': 1536,
        'dropout': 0.1
    }
    model = build_model(config).to(device)

    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    vocab_size = train_dataset.vocab_size
    # ── 初始基线 PPL ──────────────────────────────────────────────────────────
    logger.info("计算训练前基线 PPL（随机初始化）...")
    baseline_ppl = compute_ppl(model, val_loader, device)
    logger.info(f"基线 val PPL：{baseline_ppl:.1f}（随机猜测约等于 vocab_size={vocab_size}）")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_ppl = float("inf")
    global_step = 0
    
     # ── 训练循环 ──────────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)  # (B, T, vocab_size)
            B, T, V = logits.shape

            loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # 假设0是padding token ID
            loss = loss_fn(logits.view(B*T, V), y.view(B*T))  # 展平后计算交叉熵
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # 梯度裁剪
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if step % 100 == 0:
                avg_loss = epoch_loss / n_batches
                ppl = math.exp(avg_loss) if avg_loss < float('inf') else float('inf')
                cur_lr = scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch}, Step {step}, Loss: {avg_loss:.4f}, PPL: {ppl:.1f}, LR: {cur_lr:.6f}")

        # ── Epoch 结束：计算验证集 PPL ─────────────────────────────────────────  
        val_ppl = compute_ppl(model, val_loader, device)
        train_ppl = math.exp(epoch_loss / n_batches) if n_batches > 0 else float('inf')
        cur_lr = scheduler.get_last_lr()[0]    # 获取当前学习率
        logger.info(
            f"\n{'='*60}\n"
            f"Epoch {epoch} 完成 | "
            f"train PPL={train_ppl:.1f} | val PPL={val_ppl:.1f} | lr={cur_lr:.2e}\n"
            f"{'='*60}"
        )
        # 记录训练日志
        log_entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_loss / n_batches,
            "train_ppl": train_ppl,
            "val_ppl": val_ppl,
            "lr": cur_lr,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # 保存每个 epoch 的 checkpoint
        ckpt_path = CKPT_DIR / f"epoch{epoch}_ppl{val_ppl:.1f}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_ppl": val_ppl,
            "vocab_size": vocab_size,
            "seq_len": seq_len,
        }, ckpt_path)
        logger.info(f"Checkpoint 已保存：{ckpt_path.name}")

        # 保存最优模型
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_path = CKPT_DIR / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_ppl": val_ppl,
                "vocab_size": vocab_size,
                "seq_len": seq_len,
            }, best_path)
            logger.info(f"最优模型已更新 → val PPL={best_val_ppl:.1f}")

    logger.info(f"\n训练完成！最优 val PPL = {best_val_ppl:.1f}")
    logger.info(f"训练日志：{LOG_PATH}")
    logger.info(f"最优模型：{CKPT_DIR / 'best_model.pt'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

'''
2026-05-27 00:27:29,162 - INFO - __main__ - Loaded dataset from /Users/wangqian/Downloads/vscode/hub-NpWj/王谦/week05/data/train_data.pt, num_samples=211457, seq_len=256, vocab_size=21128
2026-05-27 00:27:29,163 - INFO - __main__ - Loaded dataset from /Users/wangqian/Downloads/vscode/hub-NpWj/王谦/week05/data/val_data.pt, num_samples=11009, seq_len=256, vocab_size=21128
2026-05-27 00:27:29,543 - INFO - modle - Model built with 18.86M parameters
2026-05-27 00:27:30,051 - INFO - __main__ - 计算训练前基线 PPL（随机初始化）...
2026-05-27 00:31:36,349 - INFO - __main__ - 基线 val PPL：23192.8（随机猜测约等于 vocab_size=21128）
2026-05-27 00:36:34,620 - INFO - __main__ - Epoch 1, Step 100, Loss: 6.4914, PPL: 659.5, LR: 0.000300
2026-05-27 00:41:33,238 - INFO - __main__ - Epoch 1, Step 200, Loss: 5.6244, PPL: 277.1, LR: 0.000300
2026-05-27 00:46:39,436 - INFO - __main__ - Epoch 1, Step 300, Loss: 5.1433, PPL: 171.3, LR: 0.000300
2026-05-27 00:51:33,629 - INFO - __main__ - Epoch 1, Step 400, Loss: 4.8108, PPL: 122.8, LR: 0.000300
2026-05-27 00:56:11,576 - INFO - __main__ - Epoch 1, Step 500, Loss: 4.5514, PPL: 94.8, LR: 0.000300
2026-05-27 01:00:51,941 - INFO - __main__ - Epoch 1, Step 600, Loss: 4.3241, PPL: 75.5, LR: 0.000299
2026-05-27 01:05:39,179 - INFO - __main__ - Epoch 1, Step 700, Loss: 4.1168, PPL: 61.4, LR: 0.000299
2026-05-27 01:10:46,775 - INFO - __main__ - Epoch 1, Step 800, Loss: 3.9217, PPL: 50.5, LR: 0.000299
2026-05-27 01:15:23,935 - INFO - __main__ - Epoch 1, Step 900, Loss: 3.7365, PPL: 42.0, LR: 0.000298
2026-05-27 01:20:09,861 - INFO - __main__ - Epoch 1, Step 1000, Loss: 3.5599, PPL: 35.2, LR: 0.000298
2026-05-27 01:24:55,607 - INFO - __main__ - Epoch 1, Step 1100, Loss: 3.3929, PPL: 29.8, LR: 0.000298
2026-05-27 01:29:46,881 - INFO - __main__ - Epoch 1, Step 1200, Loss: 3.2342, PPL: 25.4, LR: 0.000297
2026-05-27 01:34:27,815 - INFO - __main__ - Epoch 1, Step 1300, Loss: 3.0846, PPL: 21.9, LR: 0.000297
2026-05-27 01:39:16,227 - INFO - __main__ - Epoch 1, Step 1400, Loss: 2.9446, PPL: 19.0, LR: 0.000296
2026-05-27 01:43:46,313 - INFO - __main__ - Epoch 1, Step 1500, Loss: 2.8142, PPL: 16.7, LR: 0.000296
2026-05-27 01:48:29,641 - INFO - __main__ - Epoch 1, Step 1600, Loss: 2.6926, PPL: 14.8, LR: 0.000295
2026-05-27 01:53:34,290 - INFO - __main__ - Epoch 1, Step 1700, Loss: 2.5801, PPL: 13.2, LR: 0.000295
2026-05-27 01:58:58,224 - INFO - __main__ - Epoch 1, Step 1800, Loss: 2.4756, PPL: 11.9, LR: 0.000294
2026-05-27 02:04:36,737 - INFO - __main__ - Epoch 1, Step 1900, Loss: 2.3788, PPL: 10.8, LR: 0.000293
2026-05-27 02:09:44,236 - INFO - __main__ - Epoch 1, Step 2000, Loss: 2.2887, PPL: 9.9, LR: 0.000293
2026-05-27 02:19:47,253 - INFO - __main__ - Epoch 1, Step 2100, Loss: 2.2053, PPL: 9.1, LR: 0.000292
2026-05-27 02:23:58,277 - INFO - __main__ - Epoch 1, Step 2200, Loss: 2.1276, PPL: 8.4, LR: 0.000291
2026-05-27 02:28:39,907 - INFO - __main__ - Epoch 1, Step 2300, Loss: 2.0554, PPL: 7.8, LR: 0.000290
2026-05-27 02:33:35,516 - INFO - __main__ - Epoch 1, Step 2400, Loss: 1.9880, PPL: 7.3, LR: 0.000289
2026-05-27 02:38:51,945 - INFO - __main__ - Epoch 1, Step 2500, Loss: 1.9249, PPL: 6.9, LR: 0.000288
2026-05-27 02:44:08,060 - INFO - __main__ - Epoch 1, Step 2600, Loss: 1.8660, PPL: 6.5, LR: 0.000287
2026-05-27 02:48:39,811 - INFO - __main__ - Epoch 1, Step 2700, Loss: 1.8109, PPL: 6.1, LR: 0.000286
2026-05-27 02:53:22,121 - INFO - __main__ - Epoch 1, Step 2800, Loss: 1.7591, PPL: 5.8, LR: 0.000285
2026-05-27 02:57:52,585 - INFO - __main__ - Epoch 1, Step 2900, Loss: 1.7103, PPL: 5.5, LR: 0.000284
2026-05-27 03:02:50,633 - INFO - __main__ - Epoch 1, Step 3000, Loss: 1.6643, PPL: 5.3, LR: 0.000283
2026-05-27 03:24:20,082 - INFO - __main__ - Epoch 1, Step 3100, Loss: 1.6210, PPL: 5.1, LR: 0.000282
2026-05-27 05:00:19,908 - INFO - __main__ - Epoch 1, Step 3200, Loss: 1.5799, PPL: 4.9, LR: 0.000281
2026-05-27 06:26:10,775 - INFO - __main__ - Epoch 1, Step 3300, Loss: 1.5411, PPL: 4.7, LR: 0.000280
2026-05-27 08:04:08,835 - INFO - __main__ - Epoch 1, Step 3400, Loss: 1.5043, PPL: 4.5, LR: 0.000279
2026-05-27 10:06:29,144 - INFO - __main__ - Epoch 1, Step 3500, Loss: 1.4694, PPL: 4.3, LR: 0.000278
2026-05-27 10:12:51,874 - INFO - __main__ - Epoch 1, Step 3600, Loss: 1.4362, PPL: 4.2, LR: 0.000276
'''