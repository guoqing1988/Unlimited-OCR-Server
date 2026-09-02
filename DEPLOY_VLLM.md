# Unlimited-OCR vLLM FP8 部署文档（方案 A）

OpenAI 兼容的 Unlimited-OCR 推理服务（**vLLM FP8 后端**，当前生产方案）。

推理引擎为 docker 容器 `vllm-ocr`（官方镜像，端口 9706），前端为 `server_vllm.py`
适配层（systemd，监听 0.0.0.0:9705）。API 层面与原始 Transformers 版（方案 B，
见 `DEPLOY.md`）完全一致，客户端零感知切换。

## 架构总览

```
Client ──► :9705 (systemd unlimited-ocr-vllm, FastAPI server_vllm.py)
                │
                └─ HTTP ──► docker 容器 vllm-ocr :9706 (vLLM + FP8)
                              -v /data/www/models/Unlimited-OCR:ro
```

- 适配层（`server_vllm.py`）负责：认证 / PDF转图 / Markdown 后处理 / 图片裁剪 /
  引擎生命周期（空闲 900s 自动 `docker stop`，请求到达自动 `docker start`）
- 引擎（vLLM）负责：模型推理（FP8 动态量化，支持完整 R-SWA）

## 环境要求

- NVIDIA GPU，显存 >= 10 GB（引擎活跃 ~8 GB）
- NVIDIA Driver 570+/CUDA 12.9+（本机 580.17 / CUDA 13.0）
- Docker（含 GPU 支持，nvidia-container-toolkit）
- Python 3.12 + uv（仅适配层需要）

## 一、启动引擎容器（一次性初始化）

```bash
# 首次拉取镜像（约 18.6 GB）并创建容器；之后容器启停由适配层自动管理
docker run -d --name vllm-ocr \
  --gpus '"device=0"' \
  --ipc host \
  -p 9706:8000 \
  -v /data/www/models/Unlimited-OCR:/models/Unlimited-OCR:ro \
  vllm/vllm-openai:unlimited-ocr \
  /models/Unlimited-OCR \
  --served-model-name Unlimited-OCR \
  --trust-remote-code \
  --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --quantization fp8 \
  --gpu-memory-utilization 0.2 \
  --max-model-len 32768 \
  --host 0.0.0.0 --port 8000
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--quantization fp8` | FP8 动态量化（无需预转换权重，显存减半） |
| `--gpu-memory-utilization 0.2` | 显存池上限约 8GB（48GB 卡），KV cache 池按需 |
| `--max-model-len 32768` | 上下文上限；至少需 1.88GB KV cache，低于 0.15 利用率会失败 |
| `NGramPerReqLogitsProcessor` | 防长文重复（等价 no_repeat_ngram），必须注册 |
| `--no-enable-prefix-caching` | 官方 recipe 要求 |
| `--mm-processor-cache-gb 0` | 官方 recipe 要求（多图时按请求数重算） |

> 镜像 tag：默认 `unlimited-ocr` 用 CUDA 13.0；若驱动为 CUDA 12.9 用
> `vllm/vllm-openai:unlimited-ocr-cu129`。
> 模型必须是本地路径且为官方 bf16 safetensors（非 GGUF）。

## 二、安装适配层依赖

```bash
cd /data/www/wwwroot/Unlimited-OCR
uv venv .venv --python 3.12
source .venv/bin/activate

# 适配层仅转发推理，不需要 torch（但保留历史依赖），安装全套：
uv pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

> 若仅供适配层运行可省略 torch 安装（server_vllm.py 不 import torch）；
> 但安装后兼容性与旧版一致。

## 三、启动适配层（systemd，推荐）

```bash
sudo cp unlimited-ocr-vllm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now unlimited-ocr-vllm

# 管理
sudo systemctl status unlimited-ocr-vllm
sudo journalctl -u unlimited-ocr-vllm -f
```

关键点：service 的 ExecStart 硬编码 `--host 0.0.0.0 --port 9705`（不用 ${PORT} 变量），
避免与 .env 遗留的 PORT 冲突；HOST/PORT 由 Environment 显式指定。

### 手动启动（调试）

```bash
source .venv/bin/activate
uvicorn server_vllm:app --host 0.0.0.0 --port 9705
```

## 自动启停（懒加载 / 空闲卸载）

适配层内置引擎容器生命周期管理：

- 启动时不拉起引擎（对齐原版懒加载）；首次请求时自动 `docker start vllm-ocr`
  并轮询 `/v1/models` 等待就绪（冷启动约 30-40s），期间请求阻塞等待；
- 空闲 `IDLE_UNLOAD_SECONDS`（默认 900s）后 watchdog 自动 `docker stop vllm-ocr`
  释放 GPU 显存；
- 容器被误删时按内置 VLLM_START_ARGS 自动 `docker run` 重建；
- 手动卸载：`curl -X POST localhost:9705/admin/unload`。

```bash
# 查看状态
curl http://localhost:9705/health | python3 -m json.tool
```

`status` 字段：`ok`（引擎就绪）/ `stopped`（已空闲卸载）/ `degraded`（异常）。

## 适配层配置（.env / service 环境变量）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_URL` | `http://127.0.0.1:9706/v1` | 下游引擎地址 |
| `VLLM_CONTAINER` | `vllm-ocr` | 引擎容器名 |
| `VLLM_START_ARGS` | (内置) | 容器重建参数 |
| `VLLM_START_TIMEOUT` | `240` | 引擎就绪等待超时(s) |
| `VLLM_MAX_TOKENS` | `8192` | 生成 token 上限 |
| `VLLM_MAX_IMAGES_PER_REQ` | `3` | 单请求最大图片数 |
| `ENGINE_MODEL_NAME` | `Unlimited-OCR` | 引擎内模型名 |
| `API_KEY` | 空 | 认证密钥 |
| `IDLE_UNLOAD_SECONDS` | `900` | 空闲自动停容器(秒) |
| `WATCHDOG_POLL_SECONDS` | `10` | watchdog 间隔 |
| `MAX_PAGES_PER_BATCH` | `20` | PDF 每批页数 |
| `HOST`/`PORT` | `0.0.0.0`/`9705` | 由 service Environment 固定 |

## API

与方案 B（Transformers）完全一致：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（引擎状态/空闲时间） |
| GET | `/v1/models` | 模型列表 |
| POST | `/v1/chat/completions` | 多模态 OCR 推理（支持 stream） |
| POST | `/admin/unload` | 手动停引擎容器 |
| GET | `/images/{req_id}/{file}` | 静态图片服务 |

### Chat Completions 示例

```bash
curl -s http://localhost:9705/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{
    "model": "Unlimited-OCR",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "<image>document parsing."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }]
}' | python3 -m json.tool
```

> 完整 API / 输入方式 / Prompt 参考见 `SERVER.md`。

## 与 Transformers 版对比（实测）

| 项 | Transformers（方案 B） | vLLM FP8（方案 A） |
|----|:---:|:---:|
| 单页耗时 | ~9s | **~3.3s** |
| 活跃显存 | ~8.6GB | ~8GB |
| 空闲显存 | 0（进程内卸载） | 0（容器停止） |
| 冷启动 | ~5-10s | ~30-40s |
| 质量 | 基准 | 一致（仅同义词噪声） |
| R-SWA 长文档 | 完整 | 完整（官方实现） |

## 回退到 Transformers 版（方案 B）

```bash
# 1. 停 vLLM 适配层（让出 9705）
sudo systemctl stop unlimited-ocr-vllm
sudo systemctl disable unlimited-ocr-vllm
# 可选：停引擎容器释放显存
docker stop vllm-ocr

# 2. 启用 Transformers 版（unlimited-ocr.service，见 DEPLOY.md）
sudo cp unlimited-ocr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now unlimited-ocr
```

## 相关文件

```
/data/www/wwwroot/Unlimited-OCR/
├── .venv/                        # Python 3.12 虚拟环境
├── server_vllm.py                # 适配层 FastAPI（当前生产核心）
├── server.py                     # Transformers 版（方案 B，可回退）
├── unlimited-ocr-vllm.service    # 适配层 systemd（当前）
├── unlimited-ocr.service         # Transformers 版 systemd（历史）
├── .env                          # 环境配置（VLLM_URL 等见上表）
├── images/                       # 运行时：提取的图片（按 req_id 子目录）
├── log/                          # 运行时：服务日志
└── DEPLOY.md                     # 原始 Transformers 部署文档（方案 B）
```
