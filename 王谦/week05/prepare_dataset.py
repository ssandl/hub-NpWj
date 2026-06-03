"""
将原始文本 data/*.txt，拼接成固定长度块，存为 PyTorch 张量文件

教学重点：
  1. 预训练数据处理：文本 → token id → 连续长序列 → 切块
  2. "拼接后切块" vs "按句截断"：预训练用前者，充分利用每个 token
  3. 训练/验证集划分：按 token 数量划分，而非按文章数量
  4. 数据规模估算：1亿 token / seq_len=256 = ~390k 个训练样本

使用方式：
  python prepare_dataset.py                        # 默认参数
  python prepare_dataset.py --seq_len 512          # 更长上下文
  python prepare_dataset.py --max_tokens 20000000  # 快速验证（2000万token）

依赖：
  pip install transformers
"""
import os
import torch
from transformers import BertTokenizer
import logging
from pathlib import Path
import argparse
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
SEP_TOKEN_ID = 102  # [SEP] 用于文章间分隔

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'

def get_tokenizer():
    """
    加载 BERT 分词器
    如果本地路径 /pretrained_models/bert-base-chinese 存在，则从本地加载；否则从 Hugging Face 下载
    """
    tokenizer_path = BASE_DIR / 'pretrain_models' / 'bert-base-chinese'
    if os.path.exists(tokenizer_path):
        logger.info(f'Loading tokenizer from {tokenizer_path}')
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    else:
        logger.info('Loading tokenizer from Hugging Face')
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    return tokenizer
def build_dataset(seq_len=256,val_ratio:float = 0.5, max_tokens :int = None):
    """
    将原始文本 data/*.txt，拼接成固定长度块，存为 PyTorch 张量文件
    """
    text_paths = list(DATA_DIR.glob('*.txt'))
    if not text_paths:
        logger.warning(f'No text files found in {DATA_DIR}')
        return
    logger.info(f'Found {len(text_paths)} text files in {DATA_DIR}')
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    logger.info(f'Tokenizer vocab size: {vocab_size}')
     # ── 第一步：读取文章，tokenize，拼接成一条长 token 流 ──────────────────────
    # 核心思路：把所有文章首尾相接，文章之间插入 [SEP] 作为边界标记
    # 好处：没有任何 token 被浪费，每个训练样本都是完整的 seq_len 长度
    logger.info('Tokenizing and concatenating texts...')
    token_ids = []
    for text_path in text_paths:
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        ids = tokenizer.encode(text, add_special_tokens=False)  # 不添加 [CLS] [SEP]
        token_ids.extend(ids)
        token_ids.append(SEP_TOKEN_ID)  # 文章间插入 [SEP]
    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]
        logger.info(f'Using only the first {max_tokens} tokens for quick validation')
    total_tokens = len(token_ids)
    logger.info(f'Total tokens after concatenation: {total_tokens}')
    # ── 第二步：切成固定长度的 batch ───────────────────────────────
    # 核心思路：按照 seq_len 切块，前 val_ratio 的块作为验证集，剩余的作为训练集
    # 注意：这里是按 token 数量划分，而非按文章数量，确保训练/验证集都充分利用数据
    logger.info('Splitting into training and validation sets...')
    n_chunks = (total_tokens-1)// seq_len  # 向下取整，最后剩余的 token 会被丢弃
    logger.info(f'Total chunks (seq_len={seq_len}): {n_chunks}')
    #将所有id转换成torch 张量
    ids_tensor = torch.tensor(token_ids[:n_chunks * seq_len+1], dtype=torch.long)
    # ——第三步：将数据集切分为训练集和验证集──────────────────────────────
    val_size = max(1, int(n_chunks * val_ratio))  # 验证集至少包含1个块
    train_size = n_chunks - val_size
    logger.info(f'Training chunks: {train_size}, Validation chunks: {val_size}')
    # 切分训练集和验证集
    train_data = ids_tensor[:train_size * seq_len+1]
    val_data = ids_tensor[train_size * seq_len: (train_size + val_size) * seq_len+1]
    # 保存为 PyTorch 张量文件
    train_path = DATA_DIR / 'train_data.pt'
    val_path = DATA_DIR / 'val_data.pt'
    logger.info(f"Training data shape: {train_data.shape}, Validation data shape: {val_data.shape}")
    torch.save({"data": train_data, "vocab_size": vocab_size, "seq_len": seq_len}, train_path)
    torch.save({"data": val_data, "vocab_size": vocab_size, "seq_len": seq_len}, val_path)
    logger.info(f"训练集保存：{train_path},{train_data.shape}，验证集保存：{val_path},{val_data.shape}")
    return train_path, val_path
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=256, help="序列长度，默认 256")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例，默认 0.05")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="限制最大 token 数，如 20000000 表示 2000万（快速验证用）")
    args = parser.parse_args()
    build_dataset(args.seq_len, args.val_ratio, args.max_tokens)


if __name__ == "__main__":
    main()