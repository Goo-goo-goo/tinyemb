# tinyemb 部署指南(通用 x86)

目标:在任意 x86 服务器/NAS 上轻量部署 embedding 服务,开销小、开箱即用。
兼容 CPU:x86-64(2009+,含 Intel J3160/NAS、奔腾、酷睿、至强、AMD 全系)。

## 硬件要求

| 项目 | 最低 | 推荐 |
|---|---|---|
| CPU | x86-64,2 核 | 4 核+(J3160 即可) |
| 内存 | 256 MB | 512 MB |
| 磁盘 | 200 MB(模型 + 二进制) | — |
| GPU | **不需要** | 纯 CPU |

内存占用估算:
- 模型权重:F32 89 MB / Q8_0 量化 55 MB
- 运行时开销:~50 MB(含批量处理缓冲)
- **总计:Q8_0 模式约 100-150 MB**

## 构建(在目标机器或交叉编译)

```bash
# 1. 安装构建工具(Debian/Ubuntu NAS 为例)
sudo apt-get install -y build-essential cmake

# 2. 构建(纯 CPU,通用 x86)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tinyemb -j

# 产物:build/libtinyemb.so(Linux)/ libtinyemb.dylib(macOS)
```

构建系统自动:
- 纯 CPU(关 Metal/CUDA/BLAS,不依赖任何加速库)
- 目标 x86-64-v2 基线(SSE4.2,2009+ CPU 都支持)
- 不用 `-march=native`(避免换机器非法指令崩溃)

**想要极致性能**,在部署机上编译并加 `-march=native`:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-march=native"
```

## 获取模型(自动下载)

首次运行时,若模型目录里没有 `model.safetensors` + `vocab.txt`,服务会**自动从 ModelScope 下载**:

```bash
# 服务启动时自动下载(推荐,无需手动操作)
python3 service/server.py

# 或手动预下载到指定目录
python3 service/fetch_model.py models/bge-small-zh-v1.5
```

- 默认 model_id:`BAAI/bge-small-zh-v1.5`(ModelScope 官方源)
- 用 `TINYEMB_MODEL_ID` 覆盖 model_id,`TINYEMB_MODEL` 指定目标目录
- 只下载推理必需的文件;已就绪则跳过下载(不重复下载)
- 若 `~/.cache/modelscope` 不可写(受限环境),自动回退到模型目录旁的 `.ms_cache`

```bash
# 用别的模型(只要目录含 model.safetensors + vocab.txt)
TINYEMB_MODEL_ID=BAAI/bge-small-en-v1.5 TINYEMB_MODEL=models/bge-en python3 service/server.py
```

## 运行服务

```bash
# 安装 Python 依赖(FastAPI + orjson + uvicorn + modelscope,约 50MB)
pip install fastapi uvicorn orjson modelscope

# 启动(默认端口 8000)
python3 service/server.py
```

## 调用(OpenAI 兼容)

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "北京的天气不错"}'
```

Python(任何 OpenAI SDK 客户端直接换 base_url):
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
vec = client.embeddings.create(input="北京的天气不错", model="bge-small-zh-v1.5").data[0].embedding
```

## 配置(环境变量)

全部可选,不设则自动按 CPU 自适应:

| 变量 | 默认 | 说明 |
|---|---|---|
| `TINYEMB_MODEL` | `models/bge-small-zh-v1.5` | 模型目录 |
| `TINYEMB_MODEL_ID` | `BAAI/bge-small-zh-v1.5` | ModelScope model_id(缺失时自动下载) |
| `TINYEMB_QUANT` | `0` | `1` 开启 Q8_0 量化(降内存 34MB,小 CPU 提速) |
| `TINYEMB_WORKERS` | 自动(核数/2) | 并发批处理线程数 |
| `TINYEMB_BATCH` | `8` | 单批条数(吞吐峰值 8~16) |
| `TINYEMB_THREADS` | `2` | 每次推理的 ggml 线程数 |
| `TINYEMB_WAIT_MS` | `5` | 动态攒批最大等待 |
| `TINYEMB_QUEUE` | `1024` | 请求队列上限(背压) |
| `TINYEMB_MAX_CHARS` | `2048` | 单条文本最大字符(防 OOM) |
| `TINYEMB_LIB` | 自动探测 | 显式指定 `.so`/`.dylib` 路径 |

## 低功耗 NAS 优化建议(J3160 等)

```bash
# 用 Q8_0 量化(降内存 + 小 CPU 提速)
TINYEMB_QUANT=1 python3 service/server.py

# J3160 是 4 核低功耗,线程数保守设
TINYEMB_QUANT=1 TINYEMB_THREADS=2 TINYEMB_WORKERS=2 python3 service/server.py
```

**为什么小 CPU 要量化**:J3160 这类 CPU 是内存带宽瓶颈,矩阵乘的耗时主要在"从内存读权重"。Q8_0 让权重少读 4 倍,提速明显(M3 这类快 CPU 测不出差距)。

## 后台常驻(systemd,适用于 NAS)

```ini
# /etc/systemd/system/tinyemb.service
[Unit]
Description=tinyemb embedding service
After=network.target

[Service]
WorkingDirectory=/opt/tinyemb
Environment=TINYEMB_QUANT=1
Environment=TINYEMB_THREADS=2
ExecStart=/usr/bin/python3 service/server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now tinyemb
```

## Docker 部署(推荐,NAS 友好)

已换国内源:apt 阿里云、pip 清华、ModelScope 国内源。多阶段构建,最终镜像不含编译工具。

### 一键启动(docker compose)

```bash
docker compose up -d          # 构建 + 启动,首次自动下载模型
docker compose logs -f        # 看日志(含下载进度)
curl http://127.0.0.1:8000/health
```

### 或手动 docker build

```bash
docker build -t tinyemb .

# 运行(模型自动下载到容器内 /app/models)
docker run -d -p 8000:8000 --memory=256m --cpus=2 tinyemb

# 推荐:挂载宿主机模型目录(持久化,重建容器不重复下载)
mkdir -p ./models
docker run -d -p 8000:8000 --memory=256m --cpus=2 \
  -v $(pwd)/models:/app/models tinyemb
```

### 镜像说明

| 项 | 说明 |
|---|---|
| 基础镜像 | `python:3.11-slim`(amd64) |
| 构建方式 | 多阶段(build → runtime),最终镜像约 250MB |
| 换源 | apt 阿里云 + pip 清华 + ModelScope 国内 |
| 量化 | 默认 `TINYEMB_QUANT=1`(Q8_0,内存小) |
| 模型 | 不打进镜像,启动时自动下载或挂载 |
| 资源 | Q8_0 模式约 150MB 内存,2 核即可 |

### 换模型

```bash
docker run -d -p 8000:8000 \
  -e TINYEMB_MODEL_ID=BAAI/bge-small-en-v1.5 \
  -e TINYEMB_MODEL=/app/models/en \
  tinyemb
```


## 验证部署

```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 正确性(对比 HuggingFace transformers,误差应 < 1e-5 F32 / < 0.01 Q8_0)
curl http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" -d '{"input":"北京的天气不错"}' \
  | python3 -c "import sys,json; print('前3分量:', json.load(sys.stdin)['data'][0]['embedding'][:3])"
# 预期 F32: [0.0253, 0.1000, -0.0290]

# 性能压测
python3 service/test_client.py
```

## 故障排查

| 问题 | 原因 | 解决 |
|---|---|---|
| `找不到 libtinyemb` | 库没编译/路径错 | `cmake --build build --target tinyemb`,或设 `TINYEMB_LIB` |
| `symbol not found: tinyemb_encode_texts` | 加载了旧库 | 重新 `cmake --build build --target tinyemb` |
| `Illegal instruction` | `-march=native` 在别的 CPU 编译 | 用默认 `x86-64-v2` 基线重新编译 |
| 内存不足 OOM | 超长文本 seq² 爆炸 | 设 `TINYEMB_MAX_CHARS=512`、`TINYEMB_BATCH=4` |
| 速度慢 | 线程数不当 | J3160 用 `TINYEMB_THREADS=2`,高端 CPU 用 4 |

## 精度说明

| 模式 | vs transformers 误差 | 检索质量 | 内存 |
|---|---|---|---|
| F32(默认) | < 1e-7 | 无损 | 89 MB |
| Q8_0(`TINYEMB_QUANT=1`) | < 0.005 | 余弦一致(检索无感) | 55 MB |
