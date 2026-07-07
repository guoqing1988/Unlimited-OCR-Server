# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Unlimited-OCR 是百度推出的单次长文档解析（One-shot Long-horizon Parsing）OCR 模型，基于 DeepSeek-OCR 进一步优化。该仓库主要是模型发布与推理示例代码，模型权重托管在 Hugging Face (`baidu/Unlimited-OCR`)。

## 仓库结构

- `infer.py` — 基于 SGLang 的并发批量推理脚本，支持图片目录和 PDF 两种输入模式
- `wheel/` — 预构建的 SGLang wheel 包，用于 SGLang 推理方式
- `assets/` — README 用到的图片和演示 GIF

## 推理方式

支持三种推理后端：

1. **Transformers** — 使用 HuggingFace `transformers` 库直接加载模型进行推理
2. **vLLM** — 通过官方 vLLM recipe 部署
3. **SGLang** — 使用本仓库提供的 `wheel/sglang-*.whl` 启动服务端，支持 OpenAI 兼容 API

## 常用命令

### SGLang 环境配置

```shell
uv venv --python 3.12
source .venv/bin/activate
uv pip install wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl
uv pip install kernels==0.11.7
uv pip install pymupdf==1.27.2.2
```

### 启动 SGLang 服务端

```shell
python -m sglang.launch_server \
    --model baidu/Unlimited-OCR \
    --served-model-name Unlimited-OCR \
    --attention-backend fa3 \
    --page-size 1 \
    --mem-fraction-static 0.8 \
    --context-length 32768 \
    --enable-custom-logit-processor \
    --disable-overlap-schedule \
    --skip-server-warmup \
    --host 0.0.0.0 \
    --port 10000
```

### 使用 infer.py 批量推理

```shell
# 图片目录
python infer.py --image_dir ./examples/images --output_dir ./outputs --concurrency 8 --image_mode gundam

# PDF 文件
python infer.py --pdf ./examples/document.pdf --output_dir ./outputs --concurrency 8 --image_mode gundam
```

主要参数：`--model_dir`（模型路径或 HF ID）、`--gpu`（CUDA_VISIBLE_DEVICES 值）、`--server_log`（服务端日志路径）、`--image_mode`（`gundam` 或 `base`）。

## 编码规范

- 符合 PEP 8，4 空格缩进
- 推理后端扩展代码放在对应后端模块中，不混入通用模块
- 提交 PR 需附带单测，GitHub Actions 必须通过
