# 输出 "北京的天气不错" 的基准向量(每行一个分量),供 C 程序对拍
import sys, torch
from transformers import AutoTokenizer, AutoModel
text = sys.argv[1] if len(sys.argv) > 1 else "北京的天气不错"
tok = AutoTokenizer.from_pretrained("models/bge-small-zh-v1.5")
model = AutoModel.from_pretrained("models/bge-small-zh-v1.5").eval()
with torch.no_grad():
    b = tok([text], return_tensors="pt")
    v = model(**b).last_hidden_state[:, 0]
    v = torch.nn.functional.normalize(v, p=2, dim=1)[0]
for x in v.tolist():
    print(f"{x:.10f}")
