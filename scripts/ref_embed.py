# 参考实现:用 transformers 官方管线输出基准向量,供 C++ 实现对拍
import json, sys, numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_DIR = "models/bge-small-zh-v1.5"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModel.from_pretrained(MODEL_DIR).eval()

texts = json.loads(sys.argv[1]) if len(sys.argv) > 1 else ["北京的天气不错", "如何学习编程"]

with torch.no_grad():
    batch = tok(texts, padding=True, truncation=True, return_tensors="pt")
    out = model(**batch).last_hidden_state[:, 0]          # CLS 池化
    out = torch.nn.functional.normalize(out, p=2, dim=1)  # L2 归一化

print(json.dumps({
    "input_ids": batch["input_ids"].tolist(),
    "dim": out.shape[1],
    "vectors": [[round(x, 8) for x in row] for row in out.tolist()],
}, ensure_ascii=False))
