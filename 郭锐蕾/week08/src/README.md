# 文本匹配训练脚本 - Week08 作业

本文件夹包含基于 BQ Corpus 和 LCQMC 数据集的文本匹配训练代码，支持 BiEncoder（表示型）和 CrossEncoder（交互型）两种架构。

## 文件结构

```
week08作业/
├── dataset.py                    # 数据集加载工具
├── model.py                      # 模型定义（BiEncoder、CrossEncoder）
├── evaluate.py                   # 评估工具
├── train_biencoder_bq.py         # BiEncoder 训练脚本（BQ Corpus）
├── train_biencoder_lcqmc.py      # BiEncoder 训练脚本（LCQMC）
├── train_crossencoder_bq.py      # CrossEncoder 训练脚本（BQ Corpus）
├── train_crossencoder_lcqmc.py   # CrossEncoder 训练脚本（LCQMC）
└── outputs/                      # 输出目录（自动创建）
    ├── checkpoints/              # 模型权重
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

## 使用方法

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

### 3. 模型评估

训练完成后，使用 evaluate.py 评估模型性能：

```bash
# 评估 BiEncoder 模型
python evaluate.py --model_type biencoder --ckpt outputs/checkpoints/biencoder_bq_cosine_best.pt --data_dir ../data/bq_corpus

# 评估 CrossEncoder 模型
python evaluate.py --model_type crossencoder --ckpt outputs/checkpoints/crossencoder_bq_best.pt --data_dir ../data/bq_corpus

# 评估测试集
python evaluate.py --model_type biencoder --ckpt outputs/checkpoints/biencoder_lcqmc_cosine_best.pt --data_dir ../data/lcqmc --split test
```

## 主要参数说明

### 通用参数
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

## 模型对比

| 特性 | BiEncoder | CrossEncoder |
|------|-----------|--------------|
| 架构 | Siamese 双塔 | 单塔交互 |
| 输入 | 两句独立编码 | 两句拼接 |
| 输出 | 余弦相似度 | 分类概率 |
| 预计算 | 可预计算向量 | 不可预计算 |
| 适用场景 | 大规模检索（RAG Recall） | 精排序（Reranker） |
| 训练速度 | 较快 | 较慢 |
| 精度 | 较低 | 较高 |

## 损失函数对比

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

- `checkpoints/`: 模型权重文件（.pt 格式）
- `logs/`: 训练日志（JSON 格式）
- `figures/`: 评估图表（相似度分布图等）

## 依赖安装

```bash
pip install torch transformers scikit-learn tqdm matplotlib
```

## 注意事项

1. 确保 BERT 预训练模型路径正确
2. 数据集文件格式为 JSONL，包含 `sentence1`、`sentence2`、`label` 字段
3. 训练过程中会自动创建输出目录
4. 建议使用 GPU 加速训练
5. 不同数据集可能需要调整 `max_length` 参数

## 教学重点

1. **BiEncoder vs CrossEncoder**: 理解两种架构的差异和适用场景
2. **损失函数**: CosineEmbeddingLoss 和 TripletLoss 的区别
3. **阈值搜索**: BiEncoder 需要在验证集上搜索最优分类阈值
4. **分层学习率**: BERT 骨干和分类头使用不同学习率
5. **模型层数**: 限制 BERT 层数可以加速训练（4 层约为全量的 1/3 时间）
