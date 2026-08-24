# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Unlimited-OCR 是百度推出的单次长文档解析（One-shot Long-horizon Parsing）OCR 模型，基于 DeepSeek-OCR 进一步优化。该仓库主要是模型发布与推理示例代码，模型权重托管在 Hugging Face (`baidu/Unlimited-OCR`)。本地部署使用 HuggingFace Transformers 后端。

## 仓库结构

- `server.py` — **核心**：OpenAI 兼容的 FastAPI 推理服务，懒加载 + 空闲自动卸载 + Markdown 后处理 + 图片提取
- `infer.py` — 基于 SGLang 的并发批量推理脚本（保留，当前未使用）
- `wheel/` — 预构建的 SGLang wheel 包（保留，当前未使用）
- `assets/` — README 用到的图片和演示 GIF
- `SERVER.md` — 服务器完整使用文档（API、输入方式、配置、示例代码）
- `DEPLOY.md` — 部署快速参考
- `.env` — 环境配置（已 gitignore，含 API_KEY 等敏感信息）
- `requirements.txt` — Python 依赖
- `pyproject.toml` — 项目元数据
- `unlimited-ocr.service` — systemd service 文件
- `images/` — 运行时：OCR 提取的图片（按 req_id 子目录，已 gitignore）
- `log/` — 运行时：服务日志（已 gitignore）
- `docs/superpowers/` — 设计文档和实施计划

## 推理方式

当前使用 **Transformers 后端**（SGLang 和 vLLM 在当前系统上均不可用）：

1. **Transformers** — 当前生产方案，`server.py` 提供 OpenAI 兼容 API
2. **vLLM** — 通过官方 Docker 镜像部署（需 CUDA 12.9+ 驱动，当前 12.8 不支持）
3. **SGLang** — `infer.py` 和 wheel 包提供（当前系统 torch 2.9.1 在 Python 3.12.9 上 segfault）

## 常用命令

### 环境配置

```shell
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

### 启动服务

```shell
# 手动
source .venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 9705

# systemd（推荐）
sudo systemctl start unlimited-ocr
sudo systemctl status unlimited-ocr
sudo journalctl -u unlimited-ocr -f
```

### 测试 API

```shell
# 健康检查
curl http://localhost:9705/health | python3 -m json.tool

# 单图 OCR
curl -s http://localhost:9705/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <key>" \
  -d '{"model":"Unlimited-OCR","messages":[{"role":"user","content":[{"type":"text","text":"<image>\nFree OCR."},{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}],"max_tokens":4096}'

# 手动卸载模型
curl -X POST http://localhost:9705/admin/unload -H "Authorization: Bearer <key>"

# 本地图片路径
curl ... -d '{"messages":[{...,"image_url":{"url":"/home/liu/page.png"}}]}'

# PDF 路径
curl ... -d '{"messages":[{...,"image_url":{"url":"/data/docs/report.pdf"}}]}'
```

## 服务架构

```
Client ──► :9705 ──► FastAPI (server.py)
                        │
                        ├─ 懒加载: _ensure_loaded() 按需加载模型到 GPU
                        ├─ 空闲卸载: _watchdog() 900s 空闲后释放显存
                        ├─ 认证中间件: Bearer Token（/health & /images/ 白名单）
                        │
                        ├─ 单图: model.infer(eval_mode=True) gundam 640
                        └─ 多图: model.infer_multi() base 1024（或 page_mode=single 逐张）
                                  │
                                  └─ process_raw_output() 解析 <|det|> 标签
                                       ├─ header→#, title→##, subtitle→###
                                       ├─ text 段落合并/列表识别
                                       └─ image 坐标裁剪 + alt 文本
```

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
