FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 配置时区
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && apt-get update && apt-get install -y tzdata \
    && dpkg-reconfigure --frontend noninteractive tzdata

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3.11 \
    python3.11-distutils \
    python3.11-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安装 pip 并设置 Python 3.11 为默认
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# 安装 PyTorch（CUDA 12.1 版本）
RUN pip install --no-cache-dir \
    torch==2.3.0+cu121 \
    torchvision==0.18.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖（使用清华源加速）
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    transformers==4.40.0 \
    datasets==2.18.0 \
    accelerate==0.29.0

WORKDIR /llama_GRPO_project/workspace
