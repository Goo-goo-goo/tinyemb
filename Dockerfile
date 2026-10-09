# syntax=docker/dockerfile:1
# ============================================================
# tinyemb:纯 CPU embedding 推理服务(通用 x86)
# 多阶段构建:编译阶段 → 运行阶段,最终镜像不含编译工具。
# 已换国内源(apt 阿里云 / pip 清华),适合国内 NAS 部署。
#
# 构建:docker build -t tinyemb .
# 运行:docker run -d -p 8000:8000 --memory=256m tinyemb
# 模型:默认启动时从 ModelScope 自动下载;或挂载 -v /path/models:/app/models
# ============================================================

# 目标平台 x86_64(J3160 等 NAS 均为 x86_64);也可 buildx 构建多架构
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.11-slim AS build

# ---- 换 apt 源为阿里云(Debian bookworm)----
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
 && sed -i 's|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY third_party/ggml/ third_party/ggml/
COPY include/ include/
COPY src/ src/
COPY test/ test/
COPY CMakeLists.txt ./

# 编译推理引擎(纯 CPU,x86-64-v2 通用基线)
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build --target tinyemb -j"$(nproc)"

# ============================================================
# 运行阶段:不含编译工具,只有 Python + 依赖 + 产物
# ============================================================
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.11-slim AS runtime

# ---- 换 pip 源为清华(快,且或json等都有预编译 wheel)----
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
 && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 仅运行时依赖(orjson 走预编译 wheel,不需编译器)
COPY service/requirements.txt /app/service/requirements.txt
RUN pip install --no-cache-dir -r /app/service/requirements.txt

WORKDIR /app
COPY service/ service/
# 拷贝推理引擎 + 它依赖的 ggml 动态库(libggml / libggml-cpu / libggml-base)
COPY --from=build /src/build/libtinyemb.so* /app/build/
COPY --from=build /src/build/third_party/ggml/src/libggml*.so* /app/build/
# 模型不打进镜像(减小体积)。启动时自动从 ModelScope 下载到 /app/models。
RUN mkdir -p /app/models

# 环境变量:纯 CPU + Q8_0 量化(内存小)+ 自动下载 model_id
ENV PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/app/build \
    TINYEMB_LIB=/app/build/libtinyemb.so \
    TINYEMB_MODEL=/app/models/bge-small-zh-v1.5 \
    TINYEMB_MODEL_ID=BAAI/bge-small-zh-v1.5 \
    TINYEMB_QUANT=1 \
    TINYEMB_THREADS=2 \
    TINYEMB_WORKERS=2 \
    MODELSCOPE_CACHE=/app/.ms_cache \
    MODELSCOPE_DOMAIN=modelscope.cn

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200 else 1)"

CMD ["python3", "service/server.py"]
