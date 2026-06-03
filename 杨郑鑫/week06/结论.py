"""
cls
    accuracy                           0.57     10000
   macro avg       0.57      0.56      0.56     10000
weighted avg       0.57      0.57      0.57     10000

mean
    accuracy                           0.57     10000
   macro avg       0.56      0.55      0.56     10000
weighted avg       0.57      0.57      0.57     10000

max
    accuracy                           0.56     10000
   macro avg       0.56      0.57      0.56     10000
weighted avg       0.56      0.56      0.56     10000

通过对比全局准率来看 cls和mean最好
在平衡方面 max最好

今天成都天气还不错不是特别热，而且今天我下班很早
max推测出来是：
预测：汽车 (置信度 0.5982)
Top-3：
  [ 6] 汽车    0.5982
  [10] 旅游    0.1791
  [ 3] 体育    0.0319
cls推测出来是：
预测：旅游 (置信度 0.5358)
Top-3：
  [10] 旅游    0.5358
  [ 6] 汽车    0.2390
  [ 0] 故事    0.0893
mean推测出来是：
预测：旅游 (置信度 0.5326)
Top-3：
  [10] 旅游    0.5326
  [ 6] 汽车    0.2207
  [ 5] 房产    0.0365
我这句话有一定误导性，目前来看最合理的是 cls和mean


python train.py --pool cls --use_class_weight --epochs 3
    accuracy                           0.56     10000
   macro avg       0.54      0.59      0.56     10000
weighted avg       0.57      0.56      0.56     10000
python train.py --pool mean --use_class_weight --epochs 3
    accuracy                           0.56     10000
   macro avg       0.54      0.58      0.56     10000
weighted avg       0.57      0.56      0.56     10000
python train.py --pool max --use_class_weight --epochs 3
    accuracy                           0.56     10000
   macro avg       0.54      0.59      0.55     10000
weighted avg       0.56      0.56      0.56     10000
加权之后cls表现最好

千问0.5b
 样本数   : 200
  准确率   : 72/200 = 0.3600
  无法解析 : 58 条 (29.0%)
  总耗时   : 15.8s, 均值 0.08s/条

对比参考（典型结果）：
  BERT fine-tune (3 epochs, cls)   val accuracy ≈ 0.56
  Qwen2-0.5B zero-shot             val accuracy ≈ 0.36
千问准确率很低，受限于模型太小了

样本数   : 2000
  准确率   : 687/2000 = 0.3435
  无法解析 : 507 条 (25.4%)
  总耗时   : 183.8s, 均值 0.09s/条

对比参考（典型结果）：
  BERT fine-tune (3 epochs, cls)   val accuracy ≈ 0.57 ~ 0.62
  Qwen2-0.5B zero-shot             val accuracy ≈ 0.34
样本多了准确率还在下降

python train_sft.py
三次最优：val_loss=0.6523

三次对比
BERT ：0.56
Qwen2-0.5B zero-shot ：0.3435
Qwen2-0.5B SFT（LoRA，200 条样本） ： 0.58
Qwen2-0.5B SFT（LoRA，2000 条样本）： 0.5595
sft对比zero-shot提升百分之80，sft样本多就下降了
"""
