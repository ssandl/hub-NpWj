"""
运行 week08 文本匹配多数据集实验。

默认采用正式 benchmark 口径：AFQMC / LCQMC / BQ Corpus 三个数据集、
三种 BERT 文本匹配方法、全量数据、4 层 BERT、3 epoch。
如需只调试训练链路，使用 --smoke 生成小样本均衡子集。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiment_utils import (
    METHOD_LABELS,
    METHOD_LOG_FILES,
    archive_method_outputs,
    build_training_command,
    extract_best_epoch,
    prepare_balanced_subset,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEEK6_BERT = (
    Path("/Users/skywalker124/workspace/nlp_learn/视频和课件/week6文本分类问题")
    / "pretrain_models"
    / "bert-base-chinese"
)


def resolve_bert_path(explicit: str | None) -> str:
    if explicit:
        return str(Path(explicit).expanduser().resolve())

    candidates = [
        ROOT.parent.parent / "pretrain_models" / "bert-base-chinese",
        DEFAULT_WEEK6_BERT,
        Path("bert-base-chinese"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return "bert-base-chinese"


def resolve_python_bin(explicit: str | None) -> str:
    if explicit:
        return str(Path(explicit).expanduser().resolve())

    candidates = [
        Path(sys.executable),
        Path("/opt/anaconda3/bin/python"),
        Path("/opt/anaconda3/bin/python3"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        result = subprocess.run(
            [str(candidate), "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return str(candidate.resolve())
    return sys.executable


def build_arg_parser():
    parser = argparse.ArgumentParser(description="运行 AFQMC / LCQMC / BQ Corpus 多方法文本匹配实验")
    parser.add_argument("--datasets", nargs="+", default=["afqmc", "lcqmc", "bq_corpus"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["biencoder_cosine", "biencoder_triplet", "crossencoder"],
        choices=list(METHOD_LABELS.keys()),
    )
    parser.add_argument("--bert_path", default=None)
    parser.add_argument("--python_bin", default=None, help="用于训练的 Python，需能 import torch")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_hidden_layers", default=4, type=int)
    parser.add_argument("--train_per_label", default=32, type=int)
    parser.add_argument("--eval_per_label", default=32, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--smoke", action="store_true", help="只抽小样本，用于调试训练链路，不作为正式结论")
    parser.add_argument("--full_data", action="store_true", help="兼容旧参数；当前默认已经使用完整数据集")
    parser.add_argument("--experiments_dir", default=str(ROOT / "outputs" / "experiments"))
    return parser


def parse_args():
    return build_arg_parser().parse_args()


def _run_command(command: list[str], cwd: Path, stdout_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        subprocess.run(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    args = parse_args()
    bert_path = resolve_bert_path(args.bert_path)
    python_bin = resolve_python_bin(args.python_bin)
    experiments_dir = Path(args.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    run_mode = "smoke" if args.smoke else "full"
    print(f"运行模式: {run_mode}")
    print(f"BERT 路径: {bert_path}")
    print(f"Python: {python_bin}")
    print(f"数据集: {', '.join(args.datasets)}")
    print(f"方法: {', '.join(args.methods)}")

    results = []

    for dataset_name in args.datasets:
        source_data_dir = ROOT / "data" / dataset_name
        if not source_data_dir.exists():
            raise FileNotFoundError(f"数据集不存在: {source_data_dir}")

        if args.smoke:
            train_data_dir = (
                experiments_dir
                / "_subsets"
                / f"{dataset_name}_train{args.train_per_label}_eval{args.eval_per_label}_seed{args.seed}"
            )
            prepare_balanced_subset(
                source_data_dir,
                train_data_dir,
                per_label={
                    "train": args.train_per_label,
                    "validation": args.eval_per_label,
                    "test": args.eval_per_label,
                },
                seed=args.seed,
            )
        else:
            train_data_dir = source_data_dir

        for method_key in args.methods:
            method_dir = experiments_dir / dataset_name / method_key
            stdout_path = method_dir / "train_stdout.log.tmp"
            method_dir.mkdir(parents=True, exist_ok=True)

            command = build_training_command(
                method_key=method_key,
                bert_path=bert_path,
                data_dir=train_data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_hidden_layers=args.num_hidden_layers,
                project_root=ROOT,
                python_executable=python_bin,
            )
            print(f"\n[{dataset_name}] {METHOD_LABELS[method_key]}")
            print(" ".join(command))
            _run_command(command, cwd=ROOT / "src", stdout_path=stdout_path)

            copied = archive_method_outputs(ROOT, method_key, method_dir, stdout_log=stdout_path)
            if stdout_path.exists():
                stdout_path.unlink()
            if "log_path" not in copied:
                raise FileNotFoundError(f"训练完成但未找到日志: {METHOD_LOG_FILES[method_key]}")

            best = extract_best_epoch(copied["log_path"])
            record = {
                "dataset": dataset_name,
                "method": method_key,
                "method_label": METHOD_LABELS[method_key],
                "log_path": str(Path(copied["log_path"]).relative_to(ROOT)),
                **best,
            }
            if "ckpt_path" in copied:
                record["ckpt_path"] = str(Path(copied["ckpt_path"]).relative_to(ROOT))
            results.append(record)

            print(
                f"  最优 epoch={record['epoch']} "
                f"val_acc={record['val_acc']:.4f} val_f1={record['val_f1']:.4f}"
            )

    results_path = experiments_dir / "results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n结果 JSON: {results_path}")


if __name__ == "__main__":
    main()
