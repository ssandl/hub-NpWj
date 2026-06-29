"""
训练设备选择工具。

优先级：CUDA > Apple MPS > CPU。MPS 开启 fallback，遇到个别不支持的算子时回退 CPU。
"""

from __future__ import annotations

import os


def select_device(torch_module=None):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if torch_module is None:
        import torch as torch_module

    if torch_module.cuda.is_available():
        return torch_module.device("cuda")

    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch_module.device("mps")

    return torch_module.device("cpu")
