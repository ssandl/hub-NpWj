# NER Sequence Labeling Baseline

本项目实现一个标准的 **命名实体识别（NER）序列标注模型训练、验证、测试与结果保存流程**。  
它和 LLM SFT 生成 JSON 的做法不同，这里使用 `AutoModelForTokenClassification`，属于经典 token-level sequence labeling 方案，更适合作为“在另一个数据集上实现序列标注模型训练”的作业版本。

## 1. 项目结构

```text
ner_sequence_labeling_project/
├── README.md
├── requirements.txt
├── src/
│   └── train_ner_token_cls.py
└── scripts/
    └── run_train.sh
```

## 2. 数据格式

默认读取如下目录：

```text
data/your_dataset/
├── train.json
├── dev.json
└── test.json
```

每个 JSON 文件是一个 list，每条样本格式如下：

```json
{
  "tokens": ["我", "爱", "北", "京", "天", "安", "门"],
  "ner_tags": ["O", "O", "B-LOC", "I-LOC", "I-LOC", "I-LOC", "I-LOC"]
}
```

也兼容字段名 `labels`：

```json
{
  "tokens": ["张", "三", "在", "北", "京"],
  "labels": ["B-PER", "I-PER", "O", "B-LOC", "I-LOC"]
}
```

标签采用 BIO 格式，例如：

```text
O
B-PER
I-PER
B-ORG
I-ORG
B-LOC
I-LOC
```

## 3. 安装依赖

建议使用 Python 3.9+。

```bash
pip install -r requirements.txt
```

## 4. 训练命令

使用 Hugging Face 上的中文 BERT：

```bash
bash scripts/run_train.sh
```

或者手动运行：

```bash
python src/train_ner_token_cls.py \
  --data_dir data/your_dataset \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --output_dir outputs/ner_token_cls \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --max_length 256
```

如果你的服务器不能联网，可以把 `--model_name_or_path` 改成本地预训练模型路径，例如：

```bash
python src/train_ner_token_cls.py \
  --data_dir data/your_dataset \
  --model_name_or_path /path/to/pretrain_models/chinese-roberta-wwm-ext \
  --output_dir outputs/ner_token_cls
```

## 5. 输出结果

训练完成后，输出目录中会生成：

```text
outputs/ner_token_cls/
├── best_model/
│   ├── config.json
│   ├── model.safetensors / pytorch_model.bin
│   └── tokenizer files
├── label2id.json
├── id2label.json
├── train_log.jsonl
├── dev_results.json
├── test_results.json
└── test_predictions.json
```

其中：

- `best_model/`：验证集 F1 最优的模型；
- `test_results.json`：测试集 Precision / Recall / F1；
- `test_predictions.json`：每条样本的预测 BIO 标签、金标准 BIO 标签、预测实体、金标准实体。

## 6. 和 LLM SFT NER 的区别

你同学的代码属于：

```text
文本输入 → LLM 生成 JSON → 解析实体 → 转 BIO → seqeval 评估
```

本项目属于：

```text
tokens → tokenizer 对齐 subword → token classification → BIO 标签预测 → seqeval 评估
```

两者评价指标可以统一使用 seqeval 的 entity-level Precision / Recall / F1，但建模方式不同。

## 7. 常见问题

### 7.1 tokens 和 ner_tags 长度不一致怎么办？

脚本会直接报错。请先清洗数据，保证：

```python
len(tokens) == len(ner_tags)
```

### 7.2 中文数据集应该按字还是按词？

两种都可以，但必须保证标签和 tokens 对齐。  
如果你的数据是按字标注，`tokens` 就应该是单字列表；如果按词标注，`tokens` 就应该是词列表。

### 7.3 subword 怎么处理？

默认只在每个原始 token 的第一个 subword 上计算 loss，其余 subword label 设为 `-100`，不会参与 loss 和指标计算。  
如果希望所有 subword 都参与训练，可以加：

```bash
--label_all_tokens
```

### 7.4 可以换数据集吗？

可以。只要新数据集整理成 `train.json/dev.json/test.json`，并满足 `tokens + ner_tags` 格式即可。

## 8. 建议提交说明

如果用于课程作业，建议你在报告中说明：

1. 使用了预训练语言模型进行 token-level 序列标注；
2. 使用 BIO 标签体系；
3. 使用 `seqeval` 计算实体级 Precision、Recall、F1；
4. 和 LLM 生成式 NER 相比，本方法更稳定、可控，输出天然满足 BIO 格式；
5. 局限是需要 token-level 标注数据，且对跨领域数据仍存在泛化问题。
```
