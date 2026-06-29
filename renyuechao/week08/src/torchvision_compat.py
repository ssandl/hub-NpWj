"""
兼容本机 Anaconda 中 torchvision::nms 算子缺失导致的 transformers 导入失败。

部分 torch / torchvision 版本组合会在导入 torchvision 时注册 fake nms，
但当前运行时没有声明 torchvision::nms 算子，进而拖垮 transformers 的 BERT 导入链。
文本匹配训练不使用 torchvision，这里只补一个算子声明，让导入链通过。
"""

from __future__ import annotations

_STUB_LIBRARY = None


def ensure_torchvision_nms_stub() -> bool:
    """
    若运行时缺少 torchvision::nms，则注册一个最小 schema。

    返回 True 表示本函数已确认或完成兼容处理；返回 False 表示 torch 不可用。
    """
    global _STUB_LIBRARY
    try:
        import torch
    except Exception:
        return False

    # transformers 的 FP8 集成在部分版本中会引用较新的 torch dtype。
    # 当前训练不使用 FP8；若旧 torch 没有该属性，用 uint8 作为导入期占位。
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = torch.uint8

    try:
        torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta")
        return True
    except RuntimeError as exc:
        if "does not exist" not in str(exc):
            return True

    try:
        lib = torch.library.Library("torchvision", "DEF")
        lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        _STUB_LIBRARY = lib
    except Exception:
        # 若另一个导入路径已经注册过，继续即可；目标只是避免 transformers 导入失败。
        pass
    return True
