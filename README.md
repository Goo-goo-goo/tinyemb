# tinyemb

> **为 NAS 与轻量服务器而生的 CPU embedding 服务。**
> 纯 C 推理引擎 + OpenAI 兼容接口,只做一件事:把中文文本变成向量。极小、极简、零负担。

**Ultra-lightweight CPU embedding service for NAS & small servers. Pure C + ggml, OpenAI-compatible API. Lighter than TEI & Ollama.**

[![x86](https://img.shields.io/badge/CPU-x86__64%20only-blue)](#部署)
[![GPU](https://img.shields.io/badge/GPU-not%20required-green)](#部署)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

想在 NAS、旧笔记本、或一台不想被吃掉资源的小服务器上,快速拉起一个 embedding 服务?
tinyemb 就是干这个的——**不用 GPU、不用 PyTorch、不占几个 G 内存**,几十 MB 就能跑起来。

> ⚠️ **当前限制**
> - **仅支持 x86_64 CPU**(NAS、常见服务器;暂不支持 ARM/Apple Silicon)
> - **仅适配 `bge-small-zh-v1.5`**(其他 embedding 模型需改代码,见下)

---

## 为什么选 tinyemb

| | **tinyemb** | TEI (HuggingFace) | Ollama |
|---|---|---|---|
| 定位 | 只做中文 embedding | 高性能 embedding 服务 | 通用 LLM 运行时 |
| 语言 | 纯 C(+ ggml) | Rust | Go + llama.cpp |
| GPU | 不需要 | 可选(CUDA/ROCm) | 可选 |
| 深度学习框架 | **无**(不用 torch/transformers) | 无(ONNX/Candle) | 无 |
| 部署内存 | **~150 MB**(Q8_0) | 数百 MB ~ 数 GB | 数百 MB + 模型 |
| 上手 | `docker compose up` | `docker run` | `ollama serve` + pull |
| 接口 | OpenAI `/v1/embeddings` | OpenAI 兼容 | OpenAI 兼容 |
| 适合 | **NAS / 小服务器 / 资源敏感** | 高吞吐生产集群 | 已在用 Ollama 跑 LLM |

**一句话**:
- 已经在用 **Ollama** 跑大模型?那它够用,不必换。
- 要在**集群上榨取最高吞吐**?**TEI**(尤其有 GPU)更强。
- 想在 **NAS/小机器上零负担挂一个中文 embedding 服务**?**tinyemb** 就是为这个造的。

---

## 特性

- **纯 CPU,零 GPU 依赖** —— 任意 x86_64 都能跑,含 J3160 这类低功耗 NAS CPU
- **不碰 PyTorch / transformers** —— 推理引擎是 ~1300 行纯 C,底层用 ggml 做 CPU 算子
- **OpenAI 兼容** —— 任何 OpenAI SDK / LangChain 客户端换 `base_url` 即可接入
- **Q8_0 量化** —— 权重内存 89MB → 55MB,检索质量无损
- **模型自动下载** —— 缺模型时自动从 ModelScope 拉取,开箱即用
- **动态批处理** —— 请求自动攒批,小机器也有不错的吞吐
- **数值精确** —— F32 模式与 HuggingFace transformers 输出误差 < 1e-7

---

## 快速开始

### Docker(推荐,NAS 用户)

```bash
# 0. 获取源码
git clone https://github.com/Goo-goo-goo/tinyemb.git
cd tinyemb

# 1. 一键启动:构建 + 启动 + 首次自动下载模型
docker compose up -d

# 2. 健康检查
curl http://127.0.0.1:8000/health
```

想持久化模型(避免重建容器重复下载),编辑 `docker-compose.yaml` 取消 volumes 注释。

### 裸机部署

```bash
# 1. 获取源码(含依赖 ggml,无需 --recursive)
git clone https://github.com/Goo-goo-goo/tinyemb.git
cd tinyemb

# 2. 构建推理引擎(纯 CPU)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tinyemb -j

# 3. 装 Python 服务依赖
pip install fastapi uvicorn orjson modelscope

# 4. 启动(缺模型会自动从 ModelScope 下载)
python3 service/server.py
```

> 依赖的 ggml(CPU 算子库)源码已直接放在 `third_party/ggml/`,clone 即可编译,
> 不需要额外拉子模块。

---

## 使用

### 任何 OpenAI 客户端

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
vec = client.embeddings.create(
    input="北京的天气不错",
    model="bge-small-zh-v1.5",
).data[0].embedding   # 512 维,已 L2 归一化
```

### curl

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "北京的天气不错"}'
```

`input` 支持单条字符串或数组(批量):
```json
{"input": ["句子一", "句子二"]}
```

### LangChain

```python
from langchain_openai import OpenAIEmbeddings
emb = OpenAIEmbeddings(base_url="http://127.0.0.1:8000/v1", api_key="unused",
                       model="bge-small-zh-v1.5")
```

---

## 配置

### 环境变量

全部可选,不设则按 CPU 自适应。

**模型相关**

| 变量 | 默认 | 说明 |
|---|---|---|
| `TINYEMB_MODEL` | `models/bge-small-zh-v1.5` | 模型目录路径 |
| `TINYEMB_MODEL_ID` | `BAAI/bge-small-zh-v1.5` | ModelScope model_id,缺模型时自动下载 |
| `TINYEMB_MODEL_NAME` | `bge-small-zh-v1.5` | 对外显示的模型名(OpenAI 接口返回) |
| `TINYEMB_QUANT` | `0` | `1` 开 Q8_0 量化(降内存,推荐 NAS) |

**性能相关**

| 变量 | 默认 | 说明 |
|---|---|---|
| `TINYEMB_THREADS` | `2` | 每次推理的 ggml 线程数 |
| `TINYEMB_WORKERS` | `4`(自动) | 并发批处理线程数 |
| `TINYEMB_BATCH` | `8` | 单批条数(吞吐峰值 8~16) |
| `TINYEMB_WAIT_MS` | `5` | 动态攒批最大等待(毫秒) |
| `TINYEMB_QUEUE` | `1024` | 请求队列上限(背压) |
| `TINYEMB_MAX_CHARS` | `2048` | 单条文本最大字符(防 OOM) |

**路径/缓存相关**

| 变量 | 默认 | 说明 |
|---|---|---|
| `TINYEMB_LIB` | 自动探测 | 显式指定 `libtinyemb.so` 路径 |
| `MODELSCOPE_CACHE` | `~/.cache/modelscope` | 模型下载缓存目录 |
| `MODELSCOPE_DOMAIN` | `modelscope.cn` | ModelScope 下载源(国内) |

> 服务固定监听 `0.0.0.0:8000`。

### 模型挂载(Docker)

挂载模型目录,避免重建容器重复下载(91MB):

```bash
mkdir -p ./models
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  tinyemb
```

用 docker-compose 时,取消 `docker-compose.yaml` 里 `volumes` 那行注释即可。

### Docker 运行模式

**一键(compose,推荐)**
```bash
docker compose up -d      # 构建 + 启动 + 首次自动下载
docker compose logs -f    # 看日志
docker compose down       # 停止
```

**手动 docker run**
```bash
docker build -t tinyemb .
docker run -d --name tinyemb \
  -p 8000:8000 --memory=256m --cpus=2 \
  -v $(pwd)/models:/app/models \
  -e TINYEMB_QUANT=1 \
  tinyemb
```

**低功耗 NAS(J3160 等)**
```bash
docker run -d -p 8000:8000 --memory=256m --cpus=2 \
  -v $(pwd)/models:/app/models \
  -e TINYEMB_QUANT=1 -e TINYEMB_THREADS=2 -e TINYEMB_WORKERS=2 \
  tinyemb
```

**换模型**
```bash
docker run -d -p 8000:8000 \
  -e TINYEMB_MODEL_ID=BAAI/bge-small-en-v1.5 \
  -e TINYEMB_MODEL=/app/models/en \
  -v $(pwd)/models:/app/models \
  tinyemb
```

**常用 docker run 参数**

| 参数 | 作用 |
|---|---|
| `-d` | 后台运行 |
| `-p 8000:8000` | 端口映射(宿主机:容器) |
| `-v 宿主机:容器` | 挂载目录 |
| `-e 变量=值` | 设环境变量 |
| `--memory=256m` | 内存上限(NAS 建议设) |
| `--cpus=2` | CPU 上限 |
| `--restart=unless-stopped` | 崩溃自动重启 |

**裸机启动(低功耗 NAS)**
```bash
TINYEMB_QUANT=1 TINYEMB_THREADS=2 TINYEMB_WORKERS=2 python3 service/server.py
```

---

## 性能

基准数据来自开发期在 Apple M3(4P+4E)上的测试,仅供参考——**目标部署环境是 x86 NAS/服务器**,
实际数值会因 CPU 而异(小 CPU 绝对速度低,但相对轻量优势更明显)。

| 指标 | 数值 |
|---|---|
| 单条延迟(纯引擎) | ~7 ms |
| 纯引擎批量吞吐 | 312 句/s |
| 服务吞吐(含 HTTP) | 160~187 req/s(32 并发) |
| 模型内存(Q8_0) | ~55 MB 权重 + 运行时 |
| F32 vs transformers | 误差 < 1e-7 |
| Q8_0 检索质量 | 余弦一致(检索无感) |

> 小 CPU(J3160 等)绝对速度低得多,但**相对轻量优势更明显**——内存带宽受限的 CPU
> 从量化收益最大(权重少读 4 倍)。详见 [PERFORMANCE.md](PERFORMANCE.md)。

---

## 它是怎么做到这么轻的

1. **不加载 PyTorch** —— 没有 autograd、没有张量框架开销,权重直接读进 C 数组
2. **ggml 只做算子** —— 矩阵乘/softmax 走 ggml 的 CPU 优化内核(SSE 等),其余逻辑自己写
3. **Q8_0 量化** —— 权重压到 1.125 字节/元素,内存和内存带宽双降
4. **单二进制引擎** —— `libtinyemb.so` 约 60KB,其余都是标准库

代码规模(不含第三方 ggml):
- 推理引擎:~1300 行 C
- 服务层:~500 行 Python

---

## 项目结构

```
tinyemb/
├── src/                     # 纯 C 推理引擎
│   ├── safetensor_loader.c  #   safetensors 二进制解析
│   ├── tokenizer.c          #   WordPiece 分词
│   ├── bert_layer.c         #   Transformer Encoder + 多头注意力
│   ├── bert_model.c         #   完整 BERT:嵌入 + 4 层 + CLS 池化 + L2
│   └── tinyemb.c            #   对外 C API
├── include/io/              # 头文件
├── service/                 # 服务层
│   ├── server.py            #   FastAPI,OpenAI 兼容 /v1/embeddings
│   ├── tinyemb.py           #   Python ctypes 绑定
│   └── fetch_model.py       #   ModelScope 自动下载
├── third_party/ggml/        # ggml(CPU 算子库)
├── models/                  # 模型(不进镜像,自动下载/挂载)
├── Dockerfile               # 多阶段构建,换国内源
├── docker-compose.yaml      # 一键部署
├── DEPLOY.md                # 详细部署指南
└── PERFORMANCE.md           # 性能优化报告
```

---

## 支持的模型

**当前仅适配 `bge-small-zh-v1.5`** —— 4 层 BERT、hidden 512、CLS 池化 + L2 归一化的中文 embedding 模型。

推理代码按这个结构写死(层数、维度、CLS 池化)。要支持其他模型(不同层数/维度/池化方式),需修改 `src/bert_model.c` 中的配置和池化逻辑。

- ✅ `bge-small-zh-v1.5`(已验证)
- ❌ 其他 bge 尺寸 / 英文模型 / 非 CLS 池化模型(需改代码)

如果你需要适配别的模型,欢迎提 issue / PR。

---

## 更多

- **部署细节**(systemd、Docker、交叉编译、故障排查):[DEPLOY.md](DEPLOY.md)
- **性能优化与量化**:[PERFORMANCE.md](PERFORMANCE.md)

---

## 精度

| 模式 | vs transformers | 检索质量 | 内存 |
|---|---|---|---|
| F32(默认) | < 1e-7 | 无损 | 89 MB |
| Q8_0(`TINYEMB_QUANT=1`) | < 0.005 | 余弦一致,检索无感 | 55 MB |

Q8_0 的绝对误差是量化引入的,但向量间的**相对相似度**(检索真正依赖的)与 F32 一致——我们用句间余弦验证过。

---

## License

MIT
