# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Unlimited-OCR 是百度推出的单次长文档解析（One-shot Long-horizon Parsing）OCR 模型，基于 DeepSeek-OCR 进一步优化。该仓库主要是模型发布与推理示例代码，模型权重托管在 Hugging Face (`baidu/Unlimited-OCR`)。本地部署当前使用 **vLLM + FP8 后端**（2026-09 由 Transformers 版迁移），详见下方推理方式。

## 仓库结构

- `server.py` — **Transformers 版（已停用，保留可回退）**：原生产版推理服务，端口 9705 已被 server_vllm.py 接管。
- `server_vllm.py` — **核心（当前生产）**：OpenAI 兼容 FastAPI 适配层，API 与 server.py 完全一致；推理转发给下游 docker 容器内 vLLM 引擎（`vllm-ocr`，端口 9706）；空闲 900s 自动 `docker stop` 释放显存，下次请求自动拉起。监听 0.0.0.0:9705（原端口）。模型复用 `/data/www/models/Unlimited-OCR`。
- `infer.py` — 基于 SGLang 的并发批量推理脚本（保留，当前未使用）
- `wheel/` — 预构建的 SGLang wheel 包（保留，当前未使用）
- `assets/` — README 用到的图片和演示 GIF
- `SERVER.md` — 服务器完整使用文档（API、输入方式、配置、示例代码）
- `DEPLOY.md` — 部署文档（方案 B：原始 Transformers 版，保留）
- `DEPLOY_VLLM.md` — 部署文档（方案 A：vLLM FP8，当前生产）
- `.env` — 环境配置（已 gitignore，含 API_KEY 等敏感信息）
- `requirements.txt` — Python 依赖
- `pyproject.toml` — 项目元数据
- `unlimited-ocr.service` — Transformers 版 systemd service 文件（方案 B，历史）
- `unlimited-ocr-vllm.service` — vLLM 适配层 systemd service 文件（方案 A，当前）
- `images/` — 运行时：OCR 提取的图片（按 req_id 子目录，已 gitignore）
- `log/` — 运行时：服务日志（已 gitignore）
- `docs/superpowers/` — 设计文档和实施计划

## 推理方式

1. **vLLM + FP8（当前生产方案）**：官方镜像 `vllm/vllm-openai:unlimited-ocr`（docker 容器 `vllm-ocr`），`--quantization fp8` 动态量化（无需预转权重），`server_vllm.py` 适配层转发。实测 FP8 显存 ~8GB（`--gpu-memory-utilization 0.2`）、单页 3.3s（Transformers 版 9s，快 2.7 倍）、质量与 bf16 一致；**支持完整 R-SWA**（llama.cpp 尚未合入）。驱动要求 CUDA 12.9+（本机 580.17 / CUDA 13.0 满足）。
2. **Transformers（已停用，可回退）**：`server.py`，本机直接加载 bf16 权重（~8.6GB 显存、单页 9s）。保留在仓库中，若 vLLM 出问题可 `systemctl enable --now unlimited-ocr.service` 回退（需先将端口让回）。
3. **SGLang** — `infer.py` 和 wheel 包提供（当前系统 torch 在 Python 3.12.9 上 segfault，未使用）。
4. **llama.cpp GGUF**（已实测，不建议）：Q4_K_M 量化质量劣化明显（正文幻觉/重复），且 R-SWA 未合入 master，长文档行为不等价。

## 常用命令

### 环境配置

```shell
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

### 启动 OCR 服务（当前生产 = vLLM 适配层）

> 2026-09：Transformers 版（server.py）已停用并禁用 systemd（unlimited-ocr.service），
> 由 vLLM FP8 适配层（server_vllm.py）接管原 9705 端口，客户端零感知。

```shell
# 1) 首次手动启动下游 vLLM 引擎容器（之后由适配层自动管理启停）
docker run -d --name vllm-ocr \
  --gpus '"device=0"' --ipc host -p 9706:8000 \
  -v /data/www/models/Unlimited-OCR:/models/Unlimited-OCR:ro \
  vllm/vllm-openai:unlimited-ocr \
  /models/Unlimited-OCR --served-model-name Unlimited-OCR \
  --trust-remote-code \
  --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --quantization fp8 --gpu-memory-utilization 0.2 \
  --max-model-len 32768 --host 0.0.0.0 --port 8000

# 2) 适配层以 systemd 服务方式运行（unlimited-ocr-vllm.service）
sudo systemctl enable --now unlimited-ocr-vllm   # 端口 0.0.0.0:9705（原端口）
sudo systemctl status unlimited-ocr-vllm
sudo journalctl -u unlimited-ocr-vllm -f
```

适配层空闲 900s（IDLE_UNLOAD_SECONDS）后自动 `docker stop vllm-ocr` 释放显存；
下一次请求到达时自动 `docker start` + 等待引擎就绪（冷启动约 30-40s）再推理。
手动卸载：`curl -X POST localhost:9705/admin/unload`。

注意：service 中 ExecStart 端口为硬编码 9705（不用 ${PORT} 变量），
避免与 .env 里遗留的 PORT 值产生变量展开歧义。

### 测试 API

```shell
# 健康检查
curl http://localhost:9705/health | python3 -m json.tool

# 单图 OCR
curl -s http://localhost:9705/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <key>" \
  -d '{"model":"Unlimited-OCR","messages":[{"role":"user","content":[{"type":"text","text":"<image>document parsing."},{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}],"max_tokens":4096}'

# 手动卸载模型
curl -X POST http://localhost:9705/admin/unload -H "Authorization: Bearer <key>"

# 本地图片路径
curl ... -d '{"messages":[{...,"image_url":{"url":"/home/liu/page.png"}}]}'

# PDF 路径
curl ... -d '{"messages":[{...,"image_url":{"url":"/data/docs/report.pdf"}}]}'
```

## 服务架构（vLLM FP8 版，当前生产）

```
Client ──► :9705 ──► FastAPI (server_vllm.py 适配层, systemd)
                        │
                        ├─ 懒加载: 请求到达 → docker start vllm-ocr（若已空闲卸载）
                        ├─ 空闲卸载: _watchdog() 900s 空闲后 docker stop vllm-ocr
                        ├─ 认证中间件: Bearer Token（/health & /images/ 白名单）
                        │
                        ├─ 单图: vllm_chat() 转发 1 图（vLLM 自动 crop gundam 640）
                        └─ 多图: vllm_chat() 转发多图（vLLM 自动 base 1024 + <PAGE>）
                                  │          │
                                  └─ 下游引擎 docker 容器 vllm-ocr :9706
                                              (vLLM FP8 + R-SWA, --gpu-memory-utilization 0.2)
                                              └─ process_raw_output() 解析 <|det|> 标签
                                                   ├─ header→#, title→##, subtitle→###
                                                   ├─ text 段落合并/列表识别
                                                   └─ image 坐标裁剪 + alt 文本
```

> 旧 Transformers 架构参考 git 历史（server.py 仍在仓库，可随时回退）。

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

模型输出 `<|det|>` 标签自动映射为 Markdown：
- `header` → `#`, `title` → `##`, `subtitle` → `###`
- `text` → 纯文本（同类型段落合并，序号自动识别为 `- ` 列表）
- `image` → `![alt](images/{req_id}/N.jpg)`（alt 取最近标题）
- 孤立行（无 det 标签）独立成块

注意：`image` 标签仅在实际裁剪保存成功后才输出链接；若裁剪失败（坐标无效、
原图损坏等），该图片不输出，避免生成 404 死链。

## 编码规范

- 符合 PEP 8，4 空格缩进
- 所有代码注释必须使用中文编写，注释要详细完整
- 每个函数/类必须有 docstring 说明用途、参数和返回值
- 关键逻辑段落用 `# ── 标题 ──` 分隔符标注，增强可读性
- 推理后端扩展代码放在对应后端模块中，不混入通用模块
- 提交 PR 需附带单测，GitHub Actions 必须通过
