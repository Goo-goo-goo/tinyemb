#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 tinyemb 服务的 OpenAI 兼容性。
用法:
  .venv/bin/python3 service/test_client.py            # 用原生 HTTP 测试
  .venv/bin/python3 service/test_client.py --openai   # 用 openai SDK 测试(需 pip install openai)
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"


def http_post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


def test_http():
    print("=== 1. 单条输入 ===")
    r = http_post("/v1/embeddings", {"input": "北京的天气不错", "model": "bge-small-zh-v1.5"})
    emb = r["data"][0]["embedding"]
    print(f"  object={r['object']} model={r['model']} dim={len(emb)}")
    print(f"  usage={r['usage']}")
    print(f"  前4个分量={[round(x,6) for x in emb[:4]]}")
    assert r["object"] == "list"
    assert len(emb) == 512
    assert r["usage"]["prompt_tokens"] == 9

    print("=== 2. 批量输入 ===")
    r = http_post("/v1/embeddings", {"input": ["北京的天气不错", "如何学习编程"]})
    assert len(r["data"]) == 2
    assert r["data"][0]["index"] == 0 and r["data"][1]["index"] == 1
    print(f"  2 条向量,dim={len(r['data'][0]['embedding'])}, usage={r['usage']}")

    print("=== 3. /v1/models ===")
    req = urllib.request.Request(BASE + "/v1/models")
    with urllib.request.urlopen(req) as r:
        models = json.loads(r.read())
    print(f"  {models['data'][0]['id']}")

    print("\n[HTTP 测试全部通过]")


def test_openai():
    try:
        from openai import OpenAI
    except ImportError:
        print("未安装 openai SDK,跳过(可 pip install openai)")
        return
    client = OpenAI(base_url=BASE + "/v1", api_key="unused")
    print("=== OpenAI SDK 测试 ===")
    r = client.embeddings.create(input="北京的天气不错", model="bge-small-zh-v1.5")
    emb = r.data[0].embedding
    print(f"  dim={len(emb)} 前4={[round(x,6) for x in emb[:4]]}")
    print(f"  usage: prompt_tokens={r.usage.prompt_tokens}")
    print("[OpenAI SDK 测试通过]")


if __name__ == "__main__":
    if "--openai" in sys.argv:
        test_openai()
    else:
        test_http()
        test_openai()
