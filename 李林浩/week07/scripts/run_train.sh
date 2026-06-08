#!/usr/bin/env bash

set -e

python src/train_ner_token_cls.py \
  --data_dir data/your_dataset \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --output_dir outputs/ner_token_cls \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --max_length 256 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01
