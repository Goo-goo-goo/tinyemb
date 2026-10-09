#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tinyemb OpenAI 兼容 embedding 服务(SOTA 版)
特性:
  - 动态批处理(dynamic batching):请求进队列,后台攒批,吞吐最大化
  - 请求队列:削峰填谷,背压控制
  - 多 worker:并发处理多个 batch
  - 启动预热:预热 ggml/缓存,消除冷启动延迟
启动:.venv/bin/python3 service/server.py
     (可设 TINYEMB_BATCH/TINYEMB_WAIT_MS/TINYEMB_WORKERS 环境变量)
"""
import asyncio
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future
from typing import List, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinyemb import TinyEmb  # noqa: E402
from fetch_model import ensure_model  # noqa: E402

# ---- 配置 ----
MODEL_DIR = os.environ.get("TINYEMB_MODEL", "models/bge-small-zh-v1.5")
MODEL_NAME = os.environ.get("TINYEMB_MODEL_NAME", "bge-small-zh-v1.5")
MAX_BATCH = int(os.environ.get("TINYEMB_BATCH", "8"))       # 单批上限(吞吐峰值在 8~16)
MAX_WAIT_MS = float(os.environ.get("TINYEMB_WAIT_MS", "5"))  # 攒批最大等待(毫秒)
N_WORKERS = int(os.environ.get("TINYEMB_WORKERS", "4"))      # 并发批处理线程数
N_THREADS = int(os.environ.get("TINYEMB_THREADS", "2"))      # 每个 encode 的 ggml 线程数
QUEUE_SIZE = int(os.environ.get("TINYEMB_QUEUE", "1024"))    # 请求队列上限
MAX_TEXT_CHARS = int(os.environ.get("TINYEMB_MAX_CHARS", "2048"))  # 单条文本最大字符数(防超长文本撑爆 seq² 内存)

# ---- 自动调优:按物理核数决定并发 worker 数(没显式设环境变量时)----
import multiprocessing
_phys = multiprocessing.cpu_count()
if "TINYEMB_WORKERS" not in os.environ:
    N_WORKERS = max(2, min(6, _phys // 2))   # 经验值:worker 数约为核数一半
if "TINYEMB_BATCH" not in os.environ:
    MAX_BATCH = 8                             # 实测吞吐峰值在 8~16

app = FastAPI(title="tinyemb", version="2.0.0",
              default_response_class=ORJSONResponse,
              description="OpenAI 兼容 embedding 服务(纯 C 引擎 + 动态批处理 + orjson 序列化)")

model: TinyEmb = None  # noqa

# ---- 动态批处理核心 ----
# 每个待处理请求:(text, Future)
_req_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=QUEUE_SIZE)


class _Request:
    __slots__ = ("text", "future", "enqueued_at")

    def __init__(self, text: str):
        self.text = text
        self.future: Future = Future()
        self.enqueued_at = time.perf_counter()


def _batcher_loop(worker_id: int):
    """后台批处理线程:从队列攒批,一次 encode_batch,分发结果。"""
    while True:
        try:
            # 1. 阻塞取第一个(等不到就一直等)
            first = _req_queue.get()
            if first is None:  # 毒丸:关闭
                break
            batch = [first]
            # 2. 攒批:最多 MAX_BATCH 条,最多等 MAX_WAIT_MS
            deadline = time.perf_counter() + MAX_WAIT_MS / 1000.0
            while len(batch) < MAX_BATCH:
                remain = deadline - time.perf_counter()
                if remain <= 0:
                    break
                try:
                    batch.append(_req_queue.get(timeout=remain))
                except queue.Empty:
                    break
            # 3. 一次批量编码(带 token 数,省去服务端重复分词)
            texts = [r.text for r in batch]
            try:
                vecs, toks = model.encode_batch(texts, with_tokens=True)
                for r, v, nt in zip(batch, vecs, toks):
                    if not r.future.set_running_or_notify_cancel():
                        continue
                    r.future.set_result((v, nt))
            except Exception as e:  # noqa
                for r in batch:
                    if not r.future.done():
                        r.future.set_exception(e)
        except Exception:  # noqa
            continue


@app.on_event("startup")
def _startup():
    global model
    t0 = time.perf_counter()
    try:
        print(f"[tinyemb] 模型目录: {os.path.abspath(MODEL_DIR)}")
        # 模型目录缺失时自动从 ModelScope 下载(model_id 可用 TINYEMB_MODEL_ID 覆盖)
        ensure_model(MODEL_DIR, os.environ.get("TINYEMB_MODEL_ID"))
        print(f"[tinyemb] 加载模型...")
        model = TinyEmb(MODEL_DIR)
        model.set_threads(N_THREADS)

        # 预热:跑几批,预热 ggml 内核/缓存/页表,消除冷启动
        warmup_texts = ["预热句子", "warmup", "北京的天气不错", "人工智能"]
        for _ in range(3):
            model.encode_batch(warmup_texts[:MAX_BATCH])
        print(f"[tinyemb] 模型加载+预热 {time.perf_counter()-t0:.2f}s | "
              f"batch={MAX_BATCH} wait={MAX_WAIT_MS}ms workers={N_WORKERS} threads={N_THREADS}")
    except Exception:
        import traceback
        traceback.print_exc()
        print("[tinyemb] 启动失败!详细错误见上方 traceback", file=sys.stderr, flush=True)
        raise

    # 启动批处理 worker 线程
    for i in range(N_WORKERS):
        threading.Thread(target=_batcher_loop, args=(i,), daemon=True).start()


# ---- OpenAI 兼容的请求/响应模型 ----

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="文本或文本列表")
    model: str = Field(default=MODEL_NAME, description="模型名(兼容字段)")


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage


def _submit(text: str) -> Future:
    """把一个请求放入队列,返回 Future。超长文本截断,保护小内存设备。"""
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    req = _Request(text)
    _req_queue.put(req)  # 队列满则阻塞(自然背压)
    return req.future


@app.post("/v1/embeddings", response_model=EmbeddingResponse, response_class=ORJSONResponse)
async def create_embedding(req: EmbeddingRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        raise HTTPException(status_code=400, detail="input 不能为空")

    # 入队,异步等批量结果(不阻塞事件循环,支持真并发)
    futures = [_submit(t) for t in texts]
    afuts = [asyncio.wrap_future(f) for f in futures]
    try:
        vecs = await asyncio.wait_for(asyncio.gather(*afuts), timeout=30)
    except Exception as e:  # noqa
        raise HTTPException(status_code=500, detail=f"编码失败: {e}")

    data: List[EmbeddingData] = []
    total_tokens = 0
    for i, (vec, ntok) in enumerate(vecs):
        data.append(EmbeddingData(index=i, embedding=vec))
        total_tokens += ntok

    return EmbeddingResponse(
        data=data,
        model=req.model,
        usage=Usage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "tinyemb",
        }],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "dim": model.dim if model else 0,
        "queue_depth": _req_queue.qsize(),
        "max_batch": MAX_BATCH,
        "workers": N_WORKERS,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)  # 单进程多线程批处理
