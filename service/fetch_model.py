#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型自动下载:检测模型目录缺失时,用 ModelScope 下载。
推理只需两个文件:model.safetensors(权重)+ vocab.txt(词表)。
"""
import os
import sys
from typing import Optional

# 推理必需的文件(缺一个就认为模型不完整)
REQUIRED_FILES = ("model.safetensors", "vocab.txt")

# 默认 model_id(ModelScope);BAAI 官方源
DEFAULT_MODEL_ID = "BAAI/bge-small-zh-v1.5"


def model_ready(model_dir: str) -> bool:
    """模型目录是否已就绪(必需文件都在)。"""
    return all(os.path.isfile(os.path.join(model_dir, f)) for f in REQUIRED_FILES)


def ensure_model(model_dir: str,
                 model_id: Optional[str] = None,
                 force: bool = False) -> str:
    """
    确保模型可用:
      - 若 model_dir 里已有必需文件,直接返回(不下载)。
      - 否则用 ModelScope 下载 model_id 到 model_dir。
    返回实际使用的 model_dir。
    下载失败会抛异常(调用方可捕获处理)。
    """
    model_id = model_id or os.environ.get("TINYEMB_MODEL_ID", DEFAULT_MODEL_ID)

    if not force and model_ready(model_dir):
        return model_dir

    os.makedirs(model_dir, exist_ok=True)
    print(f"[tinyemb] 模型不完整,从 ModelScope 下载 {model_id} → {model_dir} ...")

    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "缺少 modelscope,无法自动下载模型。请先 pip install modelscope,"
            "或手动将模型放到 " + model_dir
        ) from e

    # ModelScope 默认在 ~/.cache/modelscope 存锁/缓存;某些受限环境不可写。
    # 若默认缓存目录不可写,回退到模型目录旁的 .ms_cache,保证下载可用。
    _default_cache = os.path.expanduser("~/.cache/modelscope")
    if not os.access(os.path.dirname(_default_cache) or ".", os.W_OK):
        _fallback = os.path.join(os.path.dirname(os.path.abspath(model_dir)) or ".", ".ms_cache")
        os.environ.setdefault("MODELSCOPE_CACHE", _fallback)
        os.makedirs(_fallback, exist_ok=True)

    try:
        snapshot_download(
            model_id,
            local_dir=model_dir,       # 直接下载到目标目录(平铺)
        )
    except Exception as e:
        raise RuntimeError(f"ModelScope 下载 {model_id} 失败: {e}") from e

    if not model_ready(model_dir):
        missing = [f for f in REQUIRED_FILES
                   if not os.path.isfile(os.path.join(model_dir, f))]
        raise RuntimeError(
            f"下载完成但仍缺文件: {missing}。请检查 {model_id} 是否包含它们。")

    print(f"[tinyemb] 模型就绪: {model_dir}")
    return model_dir


if __name__ == "__main__":
    # 命令行:python3 service/fetch_model.py [model_dir] [model_id]
    d = sys.argv[1] if len(sys.argv) > 1 else "models/bge-small-zh-v1.5"
    mid = sys.argv[2] if len(sys.argv) > 2 else None
    ensure_model(d, mid)
    print("OK")
