#!/bin/bash
# 生成 HF transformers 基准向量,供 test_bert_model 对拍。
# 用法: ./scripts/gen_hf_ref.sh
# 产出 /tmp/ref_short.txt 和 /tmp/ref_long.txt(各 512 行 float)
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python3}
"$PY" - <<'PY'
import sys
from transformers import AutoTokenizer, AutoModel
import torch
MODEL_DIR = "models/bge-small-zh-v1.5"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModel.from_pretrained(MODEL_DIR).eval()
long_doc = "北京是中国的首都,有着悠久的历史和灿烂的文化,故宫长城颐和园闻名天下,吸引着全世界的游客前来参观游览。" * 80
short = "北京的天气不错"
with torch.no_grad():
    batch = tok([short, long_doc], padding=True, truncation=True, return_tensors="pt")
    out = model(**batch).last_hidden_state[:, 0]
    out = torch.nn.functional.normalize(out, p=2, dim=1)
for name, row in [("short", out[0]), ("long", out[1])]:
    open(f"/tmp/ref_{name}.txt","w").write("\n".join(f"{x:.8f}" for x in row.tolist()))
    print(f"ref_{name}.txt 生成完成,token_count={len(batch['input_ids'][0 if name=='short' else 1])}")
PY
echo "基准已生成。对拍:"
echo "  ./build/test_bert_model '北京的天气不错' /tmp/ref_short.txt"
echo "  ./build/test_bert_model '<超长文本>' /tmp/ref_long.txt"
