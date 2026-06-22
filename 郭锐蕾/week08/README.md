# 文本匹配训练脚本 - Week08 作业

本文件夹包含基于 BQ Corpus 和 LCQMC 数据集的文本匹配训练代码，支持 BiEncoder（表示型）、CrossEncoder（交互型）和 LLM SFT（生成式）三种架构。

## 文件结构

```
week08作业/
├── dataset.py                    # 数据集加载工具
├── model.py                      # 模型定义（BiEncoder、CrossEncoder）
├── evaluate.py                   # BERT 模型评估工具
├── train_biencoder_bq.py         # BiEncoder 训练脚本（BQ Corpus）
├── train_biencoder_lcqmc.py      # BiEncoder 训练脚本（LCQMC）
├── train_crossencoder_bq.py      # CrossEncoder 训练脚本（BQ Corpus）
├── train_crossencoder_lcqmc.py   # CrossEncoder 训练脚本（LCQMC）
├── src_llm/                      # LLM SFT 训练脚本
│   ├── train_sft_bq.py           # LLM SFT 训练脚本（BQ Corpus）
│   ├── train_sft_lcqmc.py        # LLM SFT 训练脚本（LCQMC）
│   ├── evaluate_sft.py           # LLM SFT 评估脚本
│   └── llm_compare.py            # LLM API 对比脚本
└── outputs/                      # 输出目录（自动创建）
    ├── checkpoints/              # BERT 模型权重
    ├── sft_bq_adapter/           # BQ Corpus SFT LoRA adapter
    ├── sft_lcqmc_adapter/        # LCQMC SFT LoRA adapter
    ├── sft_bq_full_ckpt/         # BQ Corpus SFT 全量微调模型
    ├── sft_lcqmc_full_ckpt/      # LCQMC SFT 全量微调模型
    ├── logs/                     # 训练日志
    └── figures/                  # 评估图表
```

## 数据集

### BQ Corpus
- 银行客服领域问答匹配数据集
- 数据格式：`{"sentence1": "...", "sentence2": "...", "label": 0/1}`
- 数据路径：`../data/bq_corpus/`

### LCQMC
- 通用中文问答匹配数据集
- 数据格式：`{"sentence1": "...", "sentence2": "...", "label": 0/1}`
- 数据路径：`../data/lcqmc/`

## 一、BERT 模型训练（BiEncoder & CrossEncoder）

### 1. BiEncoder 训练（表示型）

BiEncoder 使用 Siamese 架构，分别编码两句话，然后计算余弦相似度。

#### BQ Corpus 数据集
```bash
# 使用 CosineEmbeddingLoss（默认）
python train_biencoder_bq.py

# 使用 TripletLoss
python train_biencoder_bq.py --loss triplet

# 自定义参数
python train_biencoder_bq.py --loss cosine --pool mean --num_hidden_layers 4 --epochs 3 --batch_size 32
```

#### LCQMC 数据集
```bash
# 使用 CosineEmbeddingLoss（默认）
python train_biencoder_lcqmc.py

# 使用 TripletLoss
python train_biencoder_lcqmc.py --loss triplet

# 自定义参数
python train_biencoder_lcqmc.py --loss cosine --pool mean --num_hidden_layers 4 --epochs 3 --batch_size 32
```

### 2. CrossEncoder 训练（交互型）

CrossEncoder 将两句话拼接后整体送入 BERT，直接输出匹配概率。

#### BQ Corpus 数据集
```bash
# 默认参数
python train_crossencoder_bq.py

# 自定义参数
python train_crossencoder_bq.py --num_hidden_layers 6 --epochs 5 --batch_size 16
```

#### LCQMC 数据集
```bash
# 默认参数
python train_crossencoder_lcqmc.py

# 自定义参数
python train_crossencoder_lcqmc.py --num_hidden_layers 6 --epochs 5 --batch_size 16
```

### 3. BERT 模型评估

训练完成后，使用 evaluate.py 评估模型性能：

```bash
# 评估 BiEncoder 模型（BQ Corpus）
python evaluate.py --model_type biencoder --ckpt outputs/checkpoints/biencoder_bq_cosine_best.pt --data_dir ../data/bq_corpus

# 评估 BiEncoder 模型（LCQMC）
python evaluate.py --model_type biencoder --ckpt outputs/checkpoints/biencoder_lcqmc_cosine_best.pt --data_dir ../data/lcqmc

# 评估 CrossEncoder 模型（BQ Corpus）
python evaluate.py --model_type crossencoder --ckpt outputs/checkpoints/crossencoder_bq_best.pt --data_dir ../data/bq_corpus

# 评估 CrossEncoder 模型（LCQMC）
python evaluate.py --model_type crossencoder --ckpt outputs/checkpoints/crossencoder_lcqmc_best.pt --data_dir ../data/lcqmc
```

## 二、LLM SFT 训练（生成式）

### 1. LLM SFT 训练

基于 Qwen2-0.5B-Instruct 模型，使用 LoRA 高效微调进行文本匹配训练。

#### BQ Corpus 数据集
```bash
# LoRA 微调（默认，5000 条快速演示）
cd src_llm
python train_sft_bq.py

# LoRA 微调，全部数据
python train_sft_bq.py --num_train -1

# 全量微调（需显存 ≥ 16GB）
python train_sft_bq.py --full_ft --lr 2e-5

# 自定义参数
python train_sft_bq.py --epochs 3 --batch_size 4 --grad_accum 4 --lr 2e-4
```

#### LCQMC 数据集
```bash
# LoRA 微调（默认，5000 条快速演示）
cd src_llm
python train_sft_lcqmc.py

# LoRA 微调，全部数据
python train_sft_lcqmc.py --num_train -1

# 全量微调（需显存 ≥ 16GB）
python train_sft_lcqmc.py --full_ft --lr 2e-5

# 自定义参数
python train_sft_lcqmc.py --epochs 3 --batch_size 4 --grad_accum 4 --lr 2e-4
```

### 2. LLM SFT 评估

训练完成后，使用 evaluate_sft.py 评估模型性能：

```bash
cd src_llm

# 评估 BQ Corpus SFT 模型（LoRA）
python evaluate_sft.py --ckpt_dir ../outputs/sft_bq_adapter --data_dir ../data/bq_corpus

# 评估 BQ Corpus SFT 模型（全量微调）
python evaluate_sft.py --ckpt_dir ../outputs/sft_bq_full_ckpt --data_dir ../data/bq_corpus

# 评估 LCQMC SFT 模型（LoRA）
python evaluate_sft.py --ckpt_dir ../outputs/sft_lcqmc_adapter --data_dir ../data/lcqmc

# 评估 LCQMC SFT 模型（全量微调）
python evaluate_sft.py --ckpt_dir ../outputs/sft_lcqmc_full_ckpt --data_dir ../data/lcqmc

# 快速演示（5 条样本）
python evaluate_sft.py --ckpt_dir ../outputs/sft_bq_adapter --data_dir ../data/bq_corpus --demo
```

### 3. LLM API 对比

使用 DashScope API 进行 zero-shot 文本匹配，与训练模型对比：

```bash
cd src_llm

# 设置 API Key
export DASHSCOPE_API_KEY="sk-xxx"

# BQ Corpus 数据集
python llm_compare.py --data_dir ../data/bq_corpus

# LCQMC 数据集
python llm_compare.py --data_dir ../data/lcqmc

# 自定义参数
python llm_compare.py --data_dir ../data/bq_corpus --num_samples 50 --model qwen-plus
```

## 主要参数说明

### BERT 模型通用参数
- `--bert_path`: BERT 预训练模型路径（默认：`../../pretrain_models/bert-base-chinese`）
- `--data_dir`: 数据集目录路径
- `--num_hidden_layers`: BERT Transformer 层数（默认：4，可选：12）
- `--epochs`: 训练轮数（默认：3）
- `--batch_size`: 批次大小（默认：32）
- `--lr`: BERT 层学习率（默认：2e-5）
- `--warmup_ratio`: Warmup 比例（默认：0.1）

### BiEncoder 专用参数
- `--loss`: 损失函数类型（cosine/triplet）
- `--pool`: 句向量池化策略（cls/mean/max，默认：mean）
- `--max_length`: 单句最大 token 数（默认：64）
- `--margin`: Margin 参数（默认：0.3）

### CrossEncoder 专用参数
- `--max_length`: 句对总最大 token 数（默认：128）

### LLM SFT 专用参数
- `--model_path`: 基座模型路径（默认：`../../pretrain_models/Qwen2-0.5B-Instruct`）
- `--num_train`: 训练样本数（默认：5000，-1 使用全部数据）
- `--full_ft`: 全量微调开关（默认：LoRA）
- `--lora_r`: LoRA rank（默认：8）
- `--lora_alpha`: LoRA alpha（默认：16）
- `--max_length`: 最大序列长度（默认：128）

## 模型对比

| 特性 | BiEncoder | CrossEncoder | LLM SFT |
|------|-----------|--------------|---------|
| 架构 | Siamese 双塔 | 单塔交互 | 生成式 |
| 输入 | 两句独立编码 | 两句拼接 | 指令格式 |
| 输出 | 余弦相似度 | 分类概率 | 文本生成 |
| 预计算 | 可预计算向量 | 不可预计算 | 不可预计算 |
| 适用场景 | 大规模检索（RAG Recall） | 精排序（Reranker） | 复杂语义理解 |
| 训练速度 | 较快 | 较慢 | 中等 |
| 精度 | 较低 | 较高 | 最高 |
| 推理速度 | 毫秒级 | 秒级 | 秒级 |

## 损失函数对比（BiEncoder）

### CosineEmbeddingLoss
- 直接用已有 (s1, s2, label) 对，无需额外构造
- 负样本到一定距离后梯度归零（margin 起到边界作用）
- 适合有标签的句对数据

### TripletLoss
- 需构造 (anchor, positive, negative) 三元组
- 更明确地告诉模型"相对远近"关系
- 适合检索/排序场景
- 负样本质量影响训练效果

## 输出文件

训练完成后，会在 `outputs/` 目录下生成：

- `checkpoints/`: BERT 模型权重文件（.pt 格式）
- `sft_bq_adapter/`: BQ Corpus SFT LoRA adapter
- `sft_lcqmc_adapter/`: LCQMC SFT LoRA adapter
- `sft_bq_full_ckpt/`: BQ Corpus SFT 全量微调模型
- `sft_lcqmc_full_ckpt/`: LCQMC SFT 全量微调模型
- `logs/`: 训练日志（JSON 格式）
- `figures/`: 评估图表（相似度分布图等）

## 依赖安装

### BERT 模型依赖
```bash
pip install torch transformers scikit-learn tqdm matplotlib
```

### LLM SFT 依赖
```bash
# LoRA 模式
pip install torch transformers peft tqdm

# 全量微调模式
pip install torch transformers tqdm

# LLM API 对比
pip install openai
```

## 注意事项

1. 确保 BERT 预训练模型路径正确（`../../pretrain_models/bert-base-chinese`）
2. 确保 Qwen2 预训练模型路径正确（`../../pretrain_models/Qwen2-0.5B-Instruct`）
3. 数据集文件格式为 JSONL，包含 `sentence1`、`sentence2`、`label` 字段
4. 训练过程中会自动创建输出目录
5. 建议使用 GPU 加速训练
6. 不同数据集可能需要调整 `max_length` 参数
7. LLM API 对比需要设置 `DASHSCOPE_API_KEY` 环境变量

## 教学重点

### BERT 模型
1. **BiEncoder vs CrossEncoder**: 理解两种架构的差异和适用场景
2. **损失函数**: CosineEmbeddingLoss 和 TripletLoss 的区别
3. **阈值搜索**: BiEncoder 需要在验证集上搜索最优分类阈值
4. **分层学习率**: BERT 骨干和分类头使用不同学习率
5. **模型层数**: 限制 BERT 层数可以加速训练（4 层约为全量的 1/3 时间）

### LLM SFT
1. **生成式 vs 判别式**: 理解生成式方法的优势和劣势
2. **指令微调格式**: 如何将文本匹配转换为指令格式
3. **Loss masking**: 只在目标 token 上计算 loss
4. **LoRA 高效微调**: 如何用少量参数实现高效训练
5. **Zero-shot vs Fine-tuned**: LLM API 与训练模型的对比

## 完整训练流程示例

### BQ Corpus 数据集完整流程
```bash
# 1. BiEncoder 训练
python train_biencoder_bq.py

# 2. CrossEncoder 训练
python train_crossencoder_bq.py

# 3. LLM SFT 训练
cd src_llm
python train_sft_bq.py

# 4. 评估所有模型
cd ..
python evaluate.py --model_type biencoder --ckpt outputs/checkpoints/biencoder_bq_cosine_best.pt --data_dir ../data/bq_corpus
python evaluate.py --model_type crossencoder --ckpt outputs/checkpoints/crossencoder_bq_best.pt --data_dir ../data/bq_corpus

cd src_llm
python evaluate_sft.py --ckpt_dir ../outputs/sft_bq_adapter --data_dir ../data/bq_corpus

# 5. LLM API 对比（可选）
export DASHSCOPE_API_KEY="sk-xxx"
python llm_compare.py --data_dir ../data/bq_corpus
```

### LCQMC 数据集完整流程
```bash
# 1. BiEncoder 训练
python train_biencoder_lcqmc.py

# 2. CrossEncoder 训练
python train_crossencoder_lcqmc.py

# 3. LLM SFT 训练
cd src_llm
python train_sft_lcqmc.py

# 4. 评估所有模型
cd ..
python evaluate.py --model_type biencoder --ckpt outputs/checkpoints/biencoder_lcqmc_cosine_best.pt --data_dir ../data/lcqmc
python evaluate.py --model_type crossencoder --ckpt outputs/checkpoints/crossencoder_lcqmc_best.pt --data_dir ../data/lcqmc

cd src_llm
python evaluate_sft.py --ckpt_dir ../outputs/sft_lcqmc_adapter --data_dir ../data/lcqmc

# 5. LLM API 对比（可选）
export DASHSCOPE_API_KEY="sk-xxx"
python llm_compare.py --data_dir ../data/lcqmc
```
