# Unlimited-OCR 推理服务部署设计

## 1. 目标

在现有系统中部署 Unlimited-OCR 推理服务，作为 DeepSeek-OCR-2 的替代方案。核心要求：

- 使用 SGLang 推理后端
- uv + Python 3.12 虚拟环境，不污染系统
- 空闲自动卸载模型释放显存（懒加载模式）
- OpenAI 兼容 API，与现有 DeepSeek-OCR-2 服务接口一致
- 输出标准 Markdown，带 alt 的图片引用
- 提供静态图片流量服务

## 2. 系统环境

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA RTX 5880 Ada, 48 GB VRAM, CC 8.9 |
| CUDA Driver | 570.86.16 / CUDA 12.8 |
| Python | /usr/bin/python3.12 (3.12.9) |
| 包管理器 | uv 0.6.12 |
| 模型路径 | /data/www/models/Unlimited-OCR |
| 模型大小 | 6.4 GB safetensors, ~3B 参数 (MoE + MLA) |

### 显存约束

- GPU 总量: 48 GB
- ComfyUI 常驻: ~12 GB
- 其他杂项: ~4 GB
- 可用: ~32 GB
- Unlimited-OCR 预算: 模型 6.4 GB + SGLang KV Cache，目标控制在 12-16 GB

## 3. 架构

```
                    +----------------------------------------------+
                    |        Unlimited-OCR Service (:10000)        |
                    |                                               |
Client ------------>|  +----------+   start/stop   +-------------+ |
                    |  |  Proxy   |<-------------->|   SGLang    | |
                    |  | (FastAPI)|                |  (:20000)   | |
                    |  |          |-----request--->|             | |
                    |  | lifecycle|<----raw text---|  inference  | |
                    |  | lazy load|                |  engine     | |
                    |  | idle unld|                +-------------+ |
                    |  | postproc |                                 |
                    |  | img serve|--> images/{req_id}/            |
                    |  +----------+   StaticFiles mount            |
                    +----------------------------------------------+
```

**Proxy (FastAPI)**：对外端口 10000，管理 SGLang 生命周期，代理请求，执行后处理管线，挂载静态图片目录。

**SGLang Server**：内部端口 20000（仅 localhost），由 Proxy 按需启停，执行实际推理。

## 4. 请求处理管线

```
POST /v1/chat/completions
  |
  +-- 1. 解析消息 -> prompt 文本 + 原图列表
  |
  +-- 2. 懒加载检查 -> SGLang 未就绪则启动，阻塞等待 health
  |     (并发请求共享同一个启动过程，不会重复拉起)
  |
  +-- 3. 保存原图副本到 images/{req_id}/original/
  |
  +-- 4. 转发请求到 SGLang /v1/chat/completions
  |     添加 images_config / custom_logit_processor / custom_params
  |     流式代理 SSE 响应，收集完整原始文本
  |
  +-- 5. 更新 _last_used 时间戳（看门狗据此判断空闲超时）
  |
  +-- 6. 后处理管线:
  |     +-- 解析 <|ref|> / <|det|> 标签
  |     |   格式: <|ref|>region_type<|/ref|><|det|>[x1,y1,x2,y2]<|/det|>
  |     |   坐标归一化到 [0,999]
  |     |
  |     +-- 图片提取: 按坐标从原图裁剪 -> images/{req_id}/N.jpg
  |     |
  |     +-- alt 生成: 向前搜索最近的 ## 标题
  |     |   格式: ![标题 - region_type](images/{req_id}/N.jpg)
  |     |
  |     +-- 标签替换: <|ref|>...<|/ref|><|det|>[...]<|/det|> -> ![...](...)
  |     |
  |     +-- 清理残留标签
  |
  +-- 7. 返回 Markdown
        stream=false: 直接返回完整 Markdown JSON
        stream=true:  收集完整文本后处理，再按词分割流式输出
```

## 5. API 端点

与 DeepSeek-OCR-2 完全对齐：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，含 `model_loaded`, `idle_seconds`, `total_requests` |
| GET | `/v1/models` | 模型列表 (`{ data: [{id: "Unlimited-OCR"}] }`) |
| POST | `/v1/chat/completions` | OCR 推理，支持 `stream: true/false` |
| POST | `/admin/unload` | 手动卸载模型 |
| GET | `/images/{req_id}/{file}` | 静态图片流量服务 |

### 请求格式 (与 DS-OCR-2 兼容)

```json
{
  "model": "Unlimited-OCR",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "<image>\nFree OCR."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }],
  "stream": true,
  "max_tokens": 4096
}
```

内部转发到 SGLang 时自动添加 `images_config`、`custom_logit_processor`、`custom_params`。

### 认证

与 DS-OCR-2 相同：`.env` 中 `API_KEY` 非空时校验 `Authorization: Bearer <key>`，`/health` 和 `/images/` 白名单免认证。

## 6. 生命周期

```
       first request
COLD ----------------> STARTING --health OK--> HOT --idle timeout--> SHUTDOWN --> COLD
(SGLang not running)   (launching...)          (serving)              (terminate)
(VRAM 0)               (waiting...)            (VRAM ~14GB)           (cuda empty)
```

- **COLD to STARTING**：收到请求，启动 SGLang 子进程。并发请求在 `threading.Lock` 上排队，共享同一个启动。
- **STARTING to HOT**：每 3 秒轮询 `http://127.0.0.1:20000/health`，就绪后放行所有等待请求。
- **HOT to SHUTDOWN**：看门狗线程每 10 秒检查 `_last_used`，超过 `IDLE_UNLOAD_SECONDS`（默认 900 秒）则 `terminate()` 子进程。
- **SHUTDOWN to COLD**：`torch.cuda.empty_cache()` + `gc.collect()` 确保显存完全释放。

并发安全：双重检查锁 (`threading.Lock`)，`_ensure_loaded()` 保证只启动一次。

## 7. 文件结构

```
/data/www/wwwroot/Unlimited-OCR/
+-- .venv/                    # uv + python3.12 venv
+-- server.py                 # NEW: FastAPI proxy service
+-- .env                      # NEW: environment config
+-- unlimited-ocr.service     # NEW: systemd unit
+-- images/                   # runtime: extracted images (req_id subdirs)
+-- log/                      # runtime: SGLang server logs
|
+-- infer.py                  # keep as-is
+-- wheel/                    # keep as-is
+-- assets/                   # keep as-is
+-- CLAUDE.md                 # keep as-is
+-- docs/superpowers/specs/   # design docs
```

## 8. 配置 (.env)

```bash
# Model
MODEL_PATH=/data/www/models/Unlimited-OCR
SERVED_MODEL_NAME=Unlimited-OCR

# Service
HOST=0.0.0.0
PORT=10000
SGLANG_PORT=20000

# Auth
API_KEY=

# Lifecycle
IDLE_UNLOAD_SECONDS=900
WATCHDOG_POLL_SECONDS=10

# GPU
GPU=0

# SGLang hyperparams
MEM_FRACTION_STATIC=0.25
CONTEXT_LENGTH=32768
```

## 9. SGLang 启动参数

```shell
python -m sglang.launch_server \
    --model ${MODEL_PATH} \
    --served-model-name ${SERVED_MODEL_NAME} \
    --attention-backend fa3 \
    --page-size 1 \
    --mem-fraction-static ${MEM_FRACTION_STATIC} \
    --context-length ${CONTEXT_LENGTH} \
    --enable-custom-logit-processor \
    --disable-overlap-schedule \
    --skip-server-warmup \
    --host 127.0.0.1 \
    --port ${SGLANG_PORT}
```

默认 `MEM_FRACTION_STATIC=0.25`（12 GB），部署时根据实际 OOM 情况调整。

## 10. 依赖

SGLang wheel 自带所有推理依赖。Proxy 层额外需要：

```
fastapi
uvicorn[standard]
python-dotenv
pydantic
requests
pillow
```

全部通过 `uv pip install` 安装到项目 `.venv`，不触碰系统 Python。

## 11. 部署步骤摘要

1. `uv venv .venv --python 3.12`
2. `uv pip install wheel/sglang-*.whl`
3. `uv pip install kernels==0.11.7 pymupdf==1.27.2.2 fastapi "uvicorn[standard]" python-dotenv`
4. 写入 `.env` 配置文件
5. 复制 `server.py` 到项目目录
6. 安装 systemd service 并 enable
7. 调优 `MEM_FRACTION_STATIC` 至稳定
