"""
AFQMC 数据集探索与可视化

教学重点：
  1. 文本匹配数据的结构——sentence pair + binary label，与分类任务的本质区别
  2. 类别不均衡（~31% 正例）——实际业务中"不同义"的问题对比"同义"更多见
  3. 句子长度分布——BERT max_length 截断阈值的选择依据
  4. 正/负样本的长度差异——是否存在"长句倾向于不相似"的捷径（shortcut）
  5. Token 数 vs 字符数——BERT 中文字节对编码的粒度

使用方式：
  python explore_data1.py
  python explore_data.py --data_dir ../data/afqmc --output_dir ../outputs/figures

依赖：
  pip install matplotlib transformers
"""

from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import os
import numpy as np
from collections import Counter

import matplotlib
# 无桌面服务器必备
matplotlib.use("Agg")
# 全局中文设置
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "PingFang_SC", "Microsoft YaHei"]
# 解决负号显示方框
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "bq_corpus"
BERT_PATH  = "/Users/wangqian/Downloads/java/八斗学院/AI训练营/2026直播/每周作业/week5/pretrain_models/bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs" / "figures1"

#中文
_CN_FONT = None

def _get_font():
    global _CN_FONT
    if _CN_FONT is None:
        from matplotlib.font_manager import FontProperties, findSystemFonts
        try:
            # 自动匹配系统中文黑体/宋体
            font_path = next(p for p in findSystemFonts() if any(k in p.lower() for k in ("PingFang", "msyh", "simsun")))
            _CN_FONT = FontProperties(fname=font_path)
        except:
            _CN_FONT = FontProperties()
    return _CN_FONT
# 1. 加载数据
def load_data(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if  line:
                row = json.loads(line)
                rows.append(row)
    return rows
    #标签分布
def plot_label_distribution(splits_data, output_dir):
    output_dir = Path(output_dir)
    #三个参数含义：行数，列数，图表大小,返回一个figure对象和一个axes对象列表
    fig, axes = plt.subplots(1, len(splits_data), figsize=(10, 4))
    if len(splits_data) == 1:
        axes = [axes]  # 保持axes为列表，方便统一处理

    fp = _get_font()#中文字体
    for ax, (split_name, rows) in zip(axes, splits_data.items()):
        labels = [row["label"] for row in rows]
        cnt = Counter(labels)
        counts = [cnt.get(0, 0), cnt.get(1, 0)]
        #绘制柱状图 参数含义：x轴标签，y轴数值，颜色，柱宽
        bars = ax.bar(["不相似 (0)", "相似 (1)"], counts,
                      color=["#F44336", "#2196F3"], width=0.5)
        for bar, label in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + max(counts)*0.01,
                    f"{label:,}", ha="center", va="bottom", fontsize=10, fontproperties=fp)
        #设置标题，使用中文字体
        ax.set_title(f"{split_name.capitalize()} 集", fontproperties=fp)
        #设置x轴标签和y轴标签，使用中文字体
        ax.set_ylabel("数量", fontproperties=fp)
        #设置x轴标签字体大小，y轴范围
        ax.tick_params(axis="x", labelsize=9)
        #设置y轴范围
        ax.set_ylim(0, max(counts)*1.2)
    plt.tight_layout()
    plt.savefig(output_dir / "label_distribution.png")
    plt.close(fig)
    print(f"标签分布图已保存到: {output_dir / 'label_distribution.png'}")

#2：句子字符长度分布
def plot_char_length_distribution(rows, output_dir): 
    output_dir = Path(output_dir)
    # 空数据防护
    if not rows:
        print("train数据集为空，跳过长度分布图生成")
        return

    pos_rows = [r for r in rows if r["label"] == 1]
    neg_rows = [r for r in rows if r["label"] == 0]

    def lengths(data):
        return [len(r["sentence1"]) for r in data] + [len(r["sentence2"]) for r in data]
    pos_lengths = lengths(pos_rows)
    neg_lengths = lengths(neg_rows)
    all_lens = pos_lengths + neg_lengths  # 只遍历一次，复用数据

    fp = _get_font()
    fig, ax = plt.subplots(figsize=(15, 10))

    # 方案1：count计数直方图（真实样本量，推荐，去掉density=True）
    ax.hist(pos_lengths, bins=50, alpha=0.7, label="正样本（相似）", color="#2196F3")
    ax.hist(neg_lengths, bins=50, alpha=0.4, label="负样本（不相似）", color="#F44336")

    # 常用截断阈值竖线，简化图例避免重叠
    ax.axvline(32, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(64, color="dimgray", linestyle="--", linewidth=1.2)
    # 标注文字替代图例，更清晰
    ax.text(32, ax.get_ylim()[1]*0.9, "max_len=32", fontproperties=fp, fontsize=8)
    ax.text(64, ax.get_ylim()[1]*0.9, "max_len=64", fontproperties=fp, fontsize=8)

    # 绘制统计参考线：均值、95分位
    mean_len = np.mean(all_lens)
    p95_len = np.percentile(all_lens, 95)
    ax.axvline(mean_len, color="orange", linestyle="-", linewidth=1.2)
    ax.text(mean_len, ax.get_ylim()[1]*0.8, f"均值={mean_len:.1f}", fontproperties=fp, fontsize=8)
    ax.axvline(p95_len, color="green", linestyle="-", linewidth=1.2)
    ax.text(p95_len, ax.get_ylim()[1]*0.8, f"P95={p95_len:.0f}", fontproperties=fp, fontsize=8)

    # 限制X轴，过滤超长异常值，防止图像挤在左侧
    clip_x = int(p95_len * 1.3)
    ax.set_xlim(0, clip_x)

    ax.set_xlabel("句子原始字符长度", fontproperties=fp)
    ax.set_ylabel("样本数量（单句）", fontproperties=fp)
    ax.set_title("正/负样本单句字符长度分布（Train集）", fontproperties=fp)
    ax.legend(prop=fp)
    fig.tight_layout()

    save_path = output_dir / "char_length_distribution.png"
    # 关键修复：bbox_inches 防止文字截断
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")

    print(f"  字符长度统计（train 全部句子）：")
    print(f"    均值={mean_len:.1f}  中位数={np.median(all_lens):.0f}  "
          f"P95={p95_len:.0f}  最长={max(all_lens)}")
    # 计算覆盖率
    for threshold in [32, 48, 64, 96]:
        cover = sum(1 for l in all_lens if l <= threshold) / len(all_lens) * 100
        print(f"    max_length={threshold:3d} 字符覆盖率: {cover:.1f}%")
    print("  ⚠️ 提示：字符覆盖率仅作参考，BERT输入限制以Token数量为准！")

#Token 数分布（BERT Tokenizer）    
def plot_token_length(rows, tokenizer, output_dir):
    output_dir = Path(output_dir)
    print("  计算 Token 长度（需要 tokenize，稍慢...）")
    token_lens = []
    for r in rows[:500]:  # 取前 500 条避免太慢
        t1 = len(tokenizer.tokenize(r["sentence1"]))
        t2 = len(tokenizer.tokenize(r["sentence2"]))
        token_lens.extend([t1, t2])

    fp = _get_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    # 绘制直方图，density=True 显示概率密度，便于观察分布形态
    ax.hist(token_lens, bins=40, color="#4CAF50", alpha=0.8, density=True)
    ax.axvline(np.mean(token_lens), color="red", linestyle="-",
               label=f"均值={np.mean(token_lens):.1f}")
    ax.axvline(np.percentile(token_lens, 95), color="orange", linestyle="--",
               label=f"P95={np.percentile(token_lens, 95):.0f}")
    ax.set_xlabel("单句 Token 数（不含 [CLS]/[SEP]）", fontproperties=fp)
    ax.set_ylabel("密度", fontproperties=fp)
    ax.set_title("单句 Token 数分布（train 前 5000 条）", fontproperties=fp)
    ax.legend(prop=fp)
    fig.tight_layout()

    save_path = output_dir / "token_length_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")
    print(f"  Token 长度：均值={np.mean(token_lens):.1f}  "
          f"P95={np.percentile(token_lens, 95):.0f}  最长={max(token_lens)}")

#参数
def parse_args():
    parser = argparse.ArgumentParser(description="AFQMC 数据集探索与可视化")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="数据文件所在目录")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="输出图表的保存目录")
    parser.add_argument("--bert_path",  default=str(BERT_PATH),  type=str)
    parser.add_argument("--skip_token", action="store_true", help="跳过 Token 长度分析（较慢）")
    return parser.parse_args()

#控制台输出
def print_stats(name, rows):
    labels = [row["label"] for row in rows]
    label_counts = Counter(labels)
    s1_lengths = [len(row["sentence1"]) for row in rows]
    s2_lengths = [len(row["sentence2"]) for row in rows]
    all_lengths = s1_lengths + s2_lengths
    print(f"数据集: {name},共 {len(rows)} 条样本")

    n_pos = label_counts.get(1, 0)
    n_neg = label_counts.get(0, 0)
    #不是0，1 的数据数量
    n_unknown = sum(count for label, count in label_counts.items() if label not in (0, 1))
    print(f"正例数: {n_pos}")
    print(f"负例数: {n_neg}")
    print(f"未知标签数: {n_unknown}")
    if n_unknown:
        print(f"未知标签分布: { {label: count for label, count in label_counts.items() if label not in (0, 1)} }")
    else:
        print(f"  正样本（相似）  : {n_pos:>6,} ({n_pos/len(rows)*100:.1f}%)")
        print(f"  负样本（不相似）: {n_neg:>6,} ({n_neg/len(rows)*100:.1f}%)")
        print(f"  不均衡比 (neg/pos): {n_neg/max(n_pos, 1):.1f}x")
    print(f"  句子字符长度 — 均值={np.mean(all_lengths):.1f}  中位数={np.median(all_lengths):.0f}  "
    f"P95={np.percentile(all_lengths, 95):.0f}  最长={max(all_lengths)}")
    print(f"  示例正样本：")
    for r in [r for r in rows if r["label"] == 1][:2]:
        print(f"    ✓  {r['sentence1']!r}  ||  {r['sentence2']!r}")
    print(f"  示例负样本：")
    for r in [r for r in rows if r["label"] == 0][:2]:
        print(f"    ✗  {r['sentence1']!r}  ||  {r['sentence2']!r}")
    # 查找超长文本（阈值200，超过正常P95=25）
    # threshold = 200
    # long_samples = []
    # for r in rows:
    #     len1 = len(r["sentence1"])
    #     len2 = len(r["sentence2"])
    #     if len1 > threshold or len2 > threshold:
    #         long_samples.append((len1, len2, r["sentence1"], r["sentence2"], r["label"]))

    # if long_samples:
    #     print(f"\n⚠️ 检测到超长异常句子（>{threshold}字符），共 {len(long_samples)} 条：")
    #     for l1, l2, s1, s2, lab in long_samples[:3]: # 只打印前3条避免刷屏
    #         print(f"sentence1长度:{l1}, sentence2长度:{l2}, label:{lab}")
    #         print(f"S1：{s1[:100]}......") # 只打印前100字符，防止输出爆炸
    #         print(f"S2：{s2[:100]}......\n") 
    #         print(f"    ✓  {long_samples[0][2]!r}  ||  {long_samples[0][3]!r}")
# 2. 数据集统计 
def main(): 
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split in ["train", "validation", "test"]:
        path = Path(args.data_dir)/ f"{split}.jsonl"
        if path.exists():
            splits[split] = load_data(path)

    for name, rows in splits.items():
        print_stats(name, rows)
    train_rows = splits.get("train", [])
    print(f"\n{'='*50}")
    print("生成可视化图表...")

    plot_label_distribution(splits, args.output_dir)
    plot_char_length_distribution(train_rows, args.output_dir)
    if not args.skip_token:
        tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        plot_token_length(train_rows, tokenizer, args.output_dir)

if __name__ == "__main__":
    main()