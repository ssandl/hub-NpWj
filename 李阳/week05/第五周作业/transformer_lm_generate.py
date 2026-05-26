"""
作者：深衷浅貌
日期：2026年05月21日--22:04
项目：NLP
文件名：transformer_lm_generate.py
"""

"""
使用训练好的 Transformer 模型生成文本

用法:
    python transformer_lm_generate.py --model_path transformer_lm.pt --prompt "今天"
    python transformer_lm_generate.py --prompt "你好" --max_new_tokens 100 --temperature 0.9
"""

import argparse
import torch

from transformer_lm_model import TransformerLanguageModel


def load_model(model_path, device="cpu"):
    """加载训练好的模型"""
    ckpt = torch.load(model_path, map_location=device)

    config = ckpt["model_config"]
    model = TransformerLanguageModel(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_attention_heads=config["num_heads"],
        max_seq_len=config["max_seq_len"],
        dropout=0  # 推理时不需要 dropout
    ).to(device)

    model.load_state_dict(ckpt["model_state"],  strict=False)
    model.eval()

    return model, ckpt["char2idx"], ckpt["idx2char"]


def main():
    parser = argparse.ArgumentParser(description="使用 Transformer 模型生成文本")

    # 模型参数
    parser.add_argument("--model_path", default="transformer_lm.pt", help="模型路径")

    # 生成参数
    parser.add_argument("--prompt", default="爱你", help="起始提示词")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=1.2, help="温度参数 (越高越随机)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K 采样 (限制候选数量)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载模型
    print(f"加载模型: {args.model_path}")
    model, char2idx, idx2char = load_model(args.model_path, device)
    print(f"模型加载成功！词表大小: {len(char2idx)}")

    # 生成文本
    print(f"\n提示词: {args.prompt}")
    print(f"温度: {args.temperature}, Top-K: {args.top_k}")
    print("-" * 50)

    generated = model.generate(
        prompt=args.prompt,
        char2idx=char2idx,
        idx2char=idx2char,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )

    print(f"生成结果:\n{generated}")
    print("=" * 50)

    # 可选：保存生成结果
    # if args.save_output:
    #     with open(args.save_output, "w", encoding="utf-8") as f:
    #         f.write(generated)
    #     print(f"结果已保存至 {args.save_output}")


if __name__ == "__main__":
    main()