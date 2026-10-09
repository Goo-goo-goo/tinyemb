#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tinyemb 的 Python ctypes 绑定:把 C 推理引擎包成 Python 对象。"""

import ctypes
import ctypes.util

# ---- 平台无关的库名/路径探测 ----
# Linux/NAS: libtinyemb.so;macOS: libtinyemb.dylib;版本化名如 libtinyemb.so.0 也兼容。
import glob
import os
import sys
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.platform == "darwin":
    _LIB_NAMES = ["libtinyemb.dylib"]
elif sys.platform.startswith("win"):
    _LIB_NAMES = ["tinyemb.dll", "libtinyemb.dll"]
else:
    _LIB_NAMES = ["libtinyemb.so", "libtinyemb.so.0"]

# 查找顺序:环境变量 TINYEMB_LIB > 构建目录 > 同目录 > 系统 ld 路径
_env_lib = os.environ.get("TINYEMB_LIB")
_search_dirs = [
    os.path.join(_HERE, "..", "build-release"),
    os.path.join(_HERE, "..", "build"),
    _HERE,
    "/usr/local/lib",
    "/usr/lib",
]
_LIB_PATH = None
if _env_lib and os.path.exists(_env_lib):
    _LIB_PATH = os.path.abspath(_env_lib)
else:
    for _d in _search_dirs:
        for _nm in _LIB_NAMES:
            _cand = os.path.join(_d, _nm)
            if os.path.exists(_cand):
                _LIB_PATH = os.path.abspath(_cand)
                break
        if _LIB_PATH:
            break
    if not _LIB_PATH:
        # 最后靠系统 loader 找(已 ldconfig 的情况)
        _found = ctypes.util.find_library("tinyemb")
        if _found:
            _LIB_PATH = _found
if not _LIB_PATH:
    raise FileNotFoundError(
        "找不到 libtinyemb。请先编译:cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "
        "&& cmake --build build --target tinyemb;或设 TINYEMB_LIB 环境变量指向 .so/.dylib"
    )

_lib = ctypes.CDLL(_LIB_PATH)

# 声明函数签名
_lib.tinyemb_load.restype = ctypes.c_void_p
_lib.tinyemb_load.argtypes = [ctypes.c_char_p]
_lib.tinyemb_free.restype = None
_lib.tinyemb_free.argtypes = [ctypes.c_void_p]
_lib.tinyemb_dim.restype = ctypes.c_int
_lib.tinyemb_dim.argtypes = [ctypes.c_void_p]
_lib.tinyemb_encode_text.restype = ctypes.c_int
_lib.tinyemb_encode_text.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]
_lib.tinyemb_token_count.restype = ctypes.c_int
_lib.tinyemb_token_count.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.tinyemb_encode_texts.restype = ctypes.c_int
_lib.tinyemb_encode_texts.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]
_lib.tinyemb_set_threads.restype = ctypes.c_int
_lib.tinyemb_set_threads.argtypes = [ctypes.c_void_p, ctypes.c_int]


class TinyEmb:
    """BERT embedding 模型(C 引擎的 Python 封装)。"""

    def __init__(self, model_dir: str):
        self._h = _lib.tinyemb_load(model_dir.encode("utf-8"))
        if not self._h:
            raise RuntimeError(f"加载模型失败: {model_dir}")
        self.dim = _lib.tinyemb_dim(self._h)

    def encode(self, text: str) -> List[float]:
        """把一段文本编码成 embedding(已 L2 归一化)。"""
        out = (ctypes.c_float * self.dim)()
        rc = _lib.tinyemb_encode_text(self._h, text.encode("utf-8"), out, self.dim)
        if rc != 0:
            raise RuntimeError(f"编码失败: {text!r}")
        return list(out)

    def encode_batch(self, texts: List[str], with_tokens: bool = False):
        """批量编码(共享矩阵乘,吞吐远高于逐句)。
        with_tokens=True 时返回 (向量列表, token 数列表),省去重复分词。"""
        if not texts:
            return ([], []) if with_tokens else []
        if len(texts) == 1:
            v = [self.encode(texts[0])]
            return (v, [self.token_count(texts[0])]) if with_tokens else v
        n = len(texts)
        arr = (ctypes.c_char_p * n)(*[t.encode("utf-8") for t in texts])
        out = (ctypes.c_float * (n * self.dim))()
        rc = _lib.tinyemb_encode_texts(self._h, arr, n, out, n * self.dim)
        if rc != 0:
            raise RuntimeError("批量编码失败")
        vecs = [[out[i * self.dim + d] for d in range(self.dim)] for i in range(n)]
        if with_tokens:
            toks = [self.token_count(t) for t in texts]
            return vecs, toks
        return vecs

    def set_threads(self, n: int):
        return _lib.tinyemb_set_threads(self._h, n)

    def token_count(self, text: str) -> int:
        return _lib.tinyemb_token_count(self._h, text.encode("utf-8"))

    def close(self):
        if self._h:
            _lib.tinyemb_free(self._h)
            self._h = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


if __name__ == "__main__":
    import sys

    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/bge-small-zh-v1.5"
    m = TinyEmb(model_dir)
    v = m.encode("北京的天气不错")
    print(f"dim = {m.dim}")
    print(f"token_count = {m.token_count('北京的天气不错')}")
    print("前 8 个分量:", [round(x, 8) for x in v[:8]])
    # 和已知基准对拍
    ref = [
        0.02531923,
        0.10003396,
        -0.02899555,
        0.04294751,
        -0.05089042,
        -0.05437123,
        0.07070439,
        0.04134032,
    ]
    max_err = max(abs(a - b) for a, b in zip(v[:8], ref))
    print(f"和基准最大误差 = {max_err:.8f}  {'[PASS]' if max_err < 1e-5 else '[FAIL]'}")
