结果表

| 方法 | 关键参数 | 是否训练 | 训练数据量 | 最优 Epoch | Accuracy | Val Macro F1 | 无法解析率 | 说明           |
|---|---|---:|---:|---:|---:|---:|---:|--------------|
| BERT fine-tune | `pool=cls` | 是 | 53360 | 2 | 0.5669 | 0.5550953946774189 | - | cls          |
| BERT fine-tune | `pool=mean` | 是 | 53360 | 2 | 0.5688 | 0.5615171106912856 | - | mean         |
| BERT fine-tune | `pool=max` | 是 | 53360 | 2 | 0.5670 | 0.559728604602726 | - | max          |
| BERT + class weight | `pool=cls` | 是 | 53360 | 3 | 0.5590 | 0.5541797592950678 | - | class weight |
| Qwen zero-shot | `num_samples=200, seed=42` | 否 | 0 | - | 0.3600 | - | 29.0% | error，后续重跑   |
| Qwen LoRA SFT | `num_train=5000, epochs=3` | 是 | 5000 | 2（val_loss 最低） | 0.5800 | - | 1.0% | error， 后续重跑  |

