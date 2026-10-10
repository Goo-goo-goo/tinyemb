#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tinyemb 压测脚本
用法:
  python3 scripts/bench.py                        # 压 127.0.0.1:8000,默认配置
  python3 scripts/bench.py --host 127.0.0.1:8000  # 指定地址
  python3 scripts/bench.py --quick                # 快速模式(请求少,几秒出结果)
  python3 scripts/bench.py --full                 # 完整模式(更多数据,更准)

测试项:
  1. 单条延迟(不同文本长度)
  2. 并发吞吐(1/2/4/8/16/32 并发)
  3. 批量请求吞吐(input 数组)
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ---- 测试文本(不同长度)----
SHORT = "你好"
MEDIUM = "北京的天气不错"
LONG = "人工智能技术在自然语言处理领域取得了显著进展，" * 4
XLONG = "这是一段比较长的中文文本用来测试在较长输入下的推理延迟表现，" * 8

BASE = "http://127.0.0.1:8000"


def one_request(text, base):
    """发一次 embedding 请求,返回(延迟ms, 响应dict)"""
    payload = json.dumps({"input": text}).encode()
    req = urllib.request.Request(
        base + "/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    return (time.perf_counter() - t0) * 1000, d


def bench_latency(base, n=20):
    """单条延迟:不同文本长度"""
    print("\n=== 1. 单条延迟(不同文本长度)===")
    print(f"{'文本':<12} {'token':>6} {'p50(ms)':>10} {'p99(ms)':>10} {'mean(ms)':>10}")
    print("-" * 54)
    for text, label in [(SHORT, "短(2字)"), (MEDIUM, "中(7字)"),
                        (LONG, "长(~80字)"), (XLONG, "超长(~240字)")]:
        lat = []
        tokens = 0
        for _ in range(n):
            ms, d = one_request(text, base)
            lat.append(ms)
            tokens = d["usage"]["prompt_tokens"]
        lat.sort()
        p50 = lat[len(lat) // 2]
        p99 = lat[min(int(len(lat) * 0.99), len(lat) - 1)]
        mean = statistics.mean(lat)
        print(f"{label:<12} {tokens:>6} {p50:>10.1f} {p99:>10.1f} {mean:>10.1f}")


def bench_concurrency(base, quick=False):
    """并发吞吐:多线程压测"""
    print("\n=== 2. 并发吞吐(单条 input,多线程)===")
    print(f"{'并发':>6} {'请求数':>8} {'吞吐(req/s)':>14} {'p50(ms)':>10} {'p99(ms)':>10}")
    print("-" * 56)
    concurrencies = [1, 2, 4, 8, 16] if quick else [1, 2, 4, 8, 16, 32]
    for nc in concurrencies:
        nreq = nc * (5 if quick else 10)
        texts = [MEDIUM] * nreq
        # 预热
        for _ in range(3):
            one_request(MEDIUM, base)
        lat = []
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=nc) as ex:
            results = list(ex.map(lambda t: one_request(t, base), texts))
        total = time.perf_counter() - t0
        lat = sorted(r[0] for r in results)
        n = len(lat)
        throughput = n / total
        p50 = lat[n // 2]
        p99 = lat[min(int(n * 0.99), n - 1)]
        print(f"{nc:>6} {n:>8} {throughput:>14.1f} {p50:>10.1f} {p99:>10.1f}")


def bench_batch(base, quick=False):
    """批量吞吐:input 数组"""
    print("\n=== 3. 批量请求吞吐(input 数组)===")
    print(f"{'batch':>6} {'吞吐(句/s)':>12} {'每批延迟(ms)':>14}")
    print("-" * 36)
    batches = [1, 4, 8] if quick else [1, 4, 8, 16, 32]
    n_round = 5 if quick else 10
    for b in batches:
        texts = [f"批量测试句子{i}" for i in range(b)]
        payload = json.dumps({"input": texts}).encode()
        req = urllib.request.Request(
            base + "/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        # 预热
        urllib.request.urlopen(req, timeout=60).read()
        t0 = time.perf_counter()
        for _ in range(n_round):
            urllib.request.urlopen(req, timeout=60).read()
        total = time.perf_counter() - t0
        throughput = b * n_round / total
        per_batch = total / n_round * 1000
        print(f"{b:>6} {throughput:>12.1f} {per_batch:>14.1f}")


def main():
    global BASE
    ap = argparse.ArgumentParser(description="tinyemb 压测")
    ap.add_argument("--host", default="127.0.0.1:8000", help="服务地址(默认 127.0.0.1:8000)")
    ap.add_argument("--quick", action="store_true", help="快速模式(请求少)")
    ap.add_argument("--full", action="store_true", help="完整模式(更多数据)")
    args = ap.parse_args()

    BASE = "http://" + args.host
    quick = args.quick and not args.full

    # 健康检查
    print(f"压测目标: {BASE}")
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
            h = json.loads(r.read())
        print(f"服务状态: {h}")
    except Exception as e:
        print(f"✗ 无法连接服务: {e}")
        print("  确认服务已启动,或用 --host 指定地址")
        sys.exit(1)

    bench_latency(BASE, n=10 if quick else 20)
    bench_concurrency(BASE, quick)
    bench_batch(BASE, quick)

    print("\n=== 压测完成 ===")
    print("提示:吞吐峰值通常在并发 8~16;批量 input 8~16 吞吐最高")


if __name__ == "__main__":
    main()
