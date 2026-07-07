"""Generate a speedup comparison chart from benchmark_results.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/benchmark_results.json")
    parser.add_argument("--output", default="outputs/speedup_comparison.png")
    args = parser.parse_args()

    input_path = Path(args.input)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    rows = data["summary_table"]
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.bar(df["engine"], df["throughput_tokens_per_s"])
    ax.set_title("Week09 vLLM Speedup Verification")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_xlabel("Inference Engine")

    for index, value in enumerate(df["throughput_tokens_per_s"]):
        ax.text(index, value, f"{value:.2f}", ha="center", va="bottom")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Chart saved to: {output_path}")


if __name__ == "__main__":
    main()
