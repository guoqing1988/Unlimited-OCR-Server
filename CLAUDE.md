# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Unlimited-OCR 是百度推出的单次长文档解析（One-shot Long-horizon Parsing）OCR 模型，基于 DeepSeek-OCR 进一步优化。该仓库主要是模型发布与推理示例代码，模型权重托管在 Hugging Face (`baidu/Unlimited-OCR` / `baidu/Unlimited-OCR-2`)。本地部署当前使用 **llama.cpp GGUF（Unlimited-OCR-2 Q8_0）后端**（2026-11 由 vLLM FP8 迁移，社区补丁版 llama.cpp），详见下方推理方式。

## 仓库结构

- `server_gguf.py` — **核心（当前生产）**：OpenAI 兼容 FastAPI 适配层，API 与 server_vllm.py 完全一致；推理转发给下游 llama-server 子进程（端口 9706，补丁版 llama.cpp + Unlimited-OCR-2 Q8_0 GGUF）；空闲 900s 自动停止子进程释放显存，下次请求自动拉起。监听 0.0.0.0:18001（端口由 .env 的 PORT 控制）。
- `server_vllm.py` — **上一版生产（保留可回退）**：vLLM FP8 适配层，转发给 docker 容器 `vllm-ocr`，API 与 server_gguf.py 一致。
- `server.py` — **Transformers 版（已停用，保留可回退）**：原生产版推理服务。
- `infer.py` — 基于 SGLang 的并发批量推理脚本（保留，当前未使用）
- `wheel/` — 预构建的 SGLang wheel 包（保留，当前未使用）
- `assets/` — README 用到的图片和演示 GIF
- `SERVER.md` — 服务器完整使用文档（API、输入方式、配置、示例代码）
- `DEPLOY_GGUF.md` — 部署文档（方案 C：llama.cpp GGUF，**当前生产**）
- `DEPLOY.md` — 部署文档（方案 B：原始 Transformers 版，保留可回退）
- `DEPLOY_VLLM.md` — 部署文档（方案 A：vLLM FP8，保留可回退）
- `.env` — 环境配置（已 gitignore，含 API_KEY 等敏感信息）
- `requirements.txt` — Python 依赖
- `pyproject.toml` — 项目元数据
- `unlimited-ocr.service` — Transformers 版 systemd service 文件（方案 B，历史）
- `unlimited-ocr-vllm.service` — vLLM 适配层 systemd service 文件（方案 A，当前）
- `images/` — 运行时：OCR 提取的图片（按 req_id 子目录，已 gitignore）
- `log/` — 运行时：服务日志（已 gitignore）
- `docs/superpowers/` — 设计文档和实施计划

## 推理方式

1. **llama.cpp GGUF + Unlimited-OCR-2（当前生产方案）**：补丁版 llama.cpp（`/data1/www/llama-ocr2`，基于 b11913 + `unlimited-ocr2.patch`），模型 `/data1/models/Unlimited-OCR-2-GGUF/Unlimited-OCR-2-Q8_0.gguf` + `mmproj-Unlimited-OCR-2-F16.gguf`。实测单页 ~5s、生成 ~250-280 t/s、显存 6.0GB（4k ctx）/8.4GB（32k ctx）。注意：主线 llama.cpp **不支持** `unlimited-ocr2` 架构，必须用补丁版；补丁只支持单序列（`-np 1`），DASH 稀疏注意力以全 KV mask 实现（结果等价、无加速）。
2. **vLLM + FP8（2026-09 上一版生产，保留可回退）**：官方镜像 `vllm/vllm-openai:unlimited-ocr`（docker 容器 `vllm-ocr`），`--quantization fp8`，`server_vllm.py` 适配层转发。实测 FP8 显存 ~8GB、单页 3.3s。需要 v1 权重 `/data/www/models/Unlimited-OCR`（本机当前缺失）。
3. **Transformers（已停用，可回退）**：`server.py`，本机直接加载 bf16 权重。
4. **SGLang** — v2 官方推荐方案（定制 wheel + Python 3.10），长文档 SR-SWA 加速更完整，当前未部署。

## 常用命令

### 环境配置

```shell
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

### 启动 OCR 服务（当前生产 = llama.cpp GGUF 适配层）

> 2026-11：vLLM FP8 版（server_vllm.py）保留可回退，由 llama.cpp GGUF 适配层
> （server_gguf.py + Unlimited-OCR-2 Q8_0）对外提供 18001 端口（.env 的 PORT/HOST 可配），客户端零感知。

```shell
# 0) 首次准备补丁版 llama-server（已编译好则跳过）
cd /data1/www/llama-ocr2   # b11913 + unlimited-ocr2.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j 48 --target llama-server llama-mtmd-cli

# 1) Python 虚拟环境（适配层只依赖 FastAPI/Pillow/PyMuPDF，不加载 torch）
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python fastapi "uvicorn[standard]" pydantic \
  python-dotenv python-multipart requests pillow pymupdf

# 2) 适配层以 systemd 服务方式运行（unlimited-ocr-gguf.service）
sudo cp unlimited-ocr-gguf.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now unlimited-ocr-gguf   # 监听地址/端口取 .env 的 HOST/PORT（当前 0.0.0.0:18001）
sudo systemctl status unlimited-ocr-gguf
sudo journalctl -u unlimited-ocr-gguf -f
```

适配层空闲 900s（IDLE_UNLOAD_SECONDS）后自动停止 llama-server 释放显存；
下一次请求到达时自动拉起（冷启动约 5s，模型文件在页缓存中）。
手动卸载：`curl -X POST localhost:18001/admin/unload`。
引擎日志：`log/llama-server.log`（由 `LLAMA_LOG` 控制）。

关键参数（.env 可覆盖）：
- `HOST=0.0.0.0`、`PORT=18001`（unit 用 `${HOST}`/`${PORT}` 展开，.env 优先于 unit 默认值）
- `LLAMA_GPU=7`（使用 GPU 7）、`LLAMA_CTX=32768`（显存 ~8.4GB；8192 约 6.4GB）
- `DRY_MULTIPLIER=0.8` 等 DRY 参数（防短句循环）
- `LLAMA_MAX_IMAGES_PER_REQ=3`（batch 模式每批页数）

### 测试 API

```shell
# 健康检查
curl http://localhost:18001/health | python3 -m json.tool

# 单图 OCR
curl -s http://localhost:18001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <key>" \
  -d '{"model":"Unlimited-OCR","messages":[{"role":"user","content":[{"type":"text","text":"<image>document parsing."},{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}],"max_tokens":4096}'

# 手动卸载模型
curl -X POST http://localhost:18001/admin/unload -H "Authorization: Bearer <key>"

# 本地图片路径
curl ... -d '{"messages":[{...,"image_url":{"url":"/home/liu/page.png"}}]}'

# PDF 路径
curl ... -d '{"messages":[{...,"image_url":{"url":"/data/docs/report.pdf"}}]}'
```

## 服务架构（llama.cpp GGUF 版，当前生产）

```
Client ──► :18001 ──► FastAPI (server_gguf.py 适配层, systemd)
                        │
                        ├─ 懒加载: 请求到达 → subprocess 启动 llama-server（若已空闲卸载）
                        ├─ 空闲卸载: _watchdog() 900s 空闲后 terminate llama-server
                        ├─ 认证中间件: Bearer Token（/health & /images/ 白名单）
                        │
                        ├─ 单图: llama_chat() 只发图片（模板自动 <PAGE><image>Multi page parsing.）
                        └─ 多图 batch: llama_chat() 一次发多图（每批 ≤ LLAMA_MAX_IMAGES_PER_REQ）
                                  │          │
                                  └─ 下游 llama-server :9706
                                              (补丁版 llama.cpp + Q8_0 GGUF + DASH attention)
                                              └─ process_raw_output() 解析 v2 标签
                                                   ├─ <BLOCK>header/title/subtitle → #/##/###
                                                   ├─ <BLOCK>text / image_caption → 段落合并
                                                   ├─ <BLOCK>image → 坐标裁剪 + alt 文本
                                                   └─ <MERGE> → 跨页同段合并
```

> 旧 vLLM 架构参考 git 历史（server_vllm.py 仍在仓库，可随时回退）。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 模型状态、空闲时间、请求计数 |
| GET | `/v1/models` | 模型列表 |
| POST | `/v1/chat/completions` | OCR 推理（支持 stream） |
| POST | `/admin/unload` | 手动卸载模型 |
| GET | `/images/{req_id}/{file}` | 静态图片服务 |

### 扩展参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `max_pages` | None（全部） | PDF/多图时限制处理页数 |
| `page_mode` | `"batch"` | `"batch"`（infer_multi 跨页连贯）或 `"single"`（逐张 gundam 高质量更快） |

## 输入方式

| 来源 | 图片 | PDF |
|------|:---:|:---:|
| `data:image/...;base64,...` | ✅ | — |
| `/absolute/path/file.jpg` | ✅ | ✅ (自动转图 300 DPI) |
| `http(s)://host/file.jpg` | ✅ | ✅ (自动下载) |

## Markdown 输出规范

v2 模型输出 `<BLOCK>type[<LEVELn>][x1,y1,x2,y2]<FORK>文本` 标签，自动映射为 Markdown：
- `<BLOCK>header[...]<FORK>` → `#`
- `<BLOCK>title[...]<FORK>` → `##`（`<LEVEL2>` → `###`）
- `<BLOCK>subtitle[...]<FORK>` → `###`
- `<BLOCK>text[...]<FORK>` / `image_caption` → 纯文本（同类型段落合并，序号自动识别为 `- ` 列表）
- `<BLOCK>image[...]<FORK>` → `![alt](images/{req_id}/N.jpg)`（alt 取最近标题）
- `<MERGE>` → 跨页同段合并（后续文本块与上一块之间不加空行）
- 孤立行（无 BLOCK 标签）独立成块

坐标仍为 [0, 999] 归一化值（兼容全角逗号等模型噪声）。
注意：`image` 标签仅在实际裁剪保存成功后才输出链接；若裁剪失败（坐标无效、
原图损坏等），该图片不输出，避免生成 404 死链。

> v1 的 `<|det|>...<|/det|>` 解析逻辑保留在 server_vllm.py 中，回退时使用。

## 编码规范

- 符合 PEP 8，4 空格缩进
- 所有代码注释必须使用中文编写，注释要详细完整
- 每个函数/类必须有 docstring 说明用途、参数和返回值
- 关键逻辑段落用 `# ── 标题 ──` 分隔符标注，增强可读性
- 推理后端扩展代码放在对应后端模块中，不混入通用模块
- 提交 PR 需附带单测，GitHub Actions 必须通过
