# Unlimited-OCR-2 GGUF 部署与使用文档

> 方案 C：llama.cpp GGUF 后端（**当前生产**）。
> 适配层 `server_gguf.py` 对外 API 与 `server_vllm.py` 完全一致；
> 下游引擎为**打了 Unlimited-OCR-2 补丁的 llama-server** + `Unlimited-OCR-2-Q8_0.gguf`。

---

## 1. 为什么必须用补丁版 llama.cpp

Unlimited-OCR-2 的 GGUF 架构名为 `unlimited-ocr2`，包含：

- page-aware 2D RoPE（页内偏移 + 页奇偶）
- SR-SWA / DASH 稀疏注意力（`indexer.*` 张量做块路由）

**主线 llama.cpp 不支持该架构**，直接加载会报：

```
error loading model: unknown model architecture: 'unlimited-ocr2'
```

官方模型仓库说明：

> These files need llama.cpp with Unlimited-OCR-2 support, which mainline llama.cpp does not have yet.
> Mainline llama.cpp and apps built on it cannot load these files.
> —— [AtomicChat/Unlimited-OCR-2-GGUF](https://huggingface.co/AtomicChat/Unlimited-OCR-2-GGUF)

补丁来源：[RX5950XT/Unlimited-OCR-2-GGUF](https://huggingface.co/RX5950XT/Unlimited-OCR-2-GGUF)
的 `llama.cpp-patch/unlimited-ocr2.patch`，基于 llama.cpp **b11913**（commit `7aca0d2ea`）。

> 注意：补丁把 DASH 路由实现为**全 KV 上的 mask**——结果与稀疏内核一致，
> 但没有论文里的 SR-SWA 加速；且页位置从 KV token 推导，**只支持单序列**（`-np 1`）。

---

## 2. 环境要求

| 项 | 要求 |
|---|---|
| GPU | NVIDIA，CUDA 12.8+（本机 RTX 4090 / CUDA 13.0） |
| 显存 | Q8_0 + 4k ctx ≈ 6.0GB；8k ≈ 6.4GB；32k ≈ 8.4GB |
| 系统内存 | 模型 mmap 页缓存，建议 ≥ 8GB 空闲（本机 1TB） |
| Python | 3.12（适配层不加载 torch） |
| 磁盘 | llama.cpp 构建 ~10GB + 模型 ~4GB |

---

## 3. 部署步骤

### 3.1 编译补丁版 llama-server

```bash
# 1) 克隆并切到补丁基线
git clone https://github.com/ggml-org/llama.cpp /data1/www/llama-ocr2
cd /data1/www/llama-ocr2
git checkout b11913

# 2) 下载并应用补丁
curl -L -o /tmp/unlimited-ocr2.patch \
  https://huggingface.co/RX5950XT/Unlimited-OCR-2-GGUF/raw/main/llama.cpp-patch/unlimited-ocr2.patch
git apply /tmp/unlimited-ocr2.patch

# 3) 编译（CUDA）
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j 48 --target llama-server llama-mtmd-cli

# 验证
./build/bin/llama-server --version
```

### 3.2 下载 GGUF 模型

```bash
mkdir -p /data1/models/Unlimited-OCR-2-GGUF
cd /data1/models/Unlimited-OCR-2-GGUF

# 推荐：Q8_0（与 BF16 差异中位 0.12%）
hf download AtomicChat/Unlimited-OCR-2-GGUF \
  Unlimited-OCR-2-Q8_0.gguf mmproj-Unlimited-OCR-2-F16.gguf --local-dir .

# 可选：更省显存的 AD-Q4_K（2.02GB，质量接近 Q8_0）
# hf download AtomicChat/Unlimited-OCR-2-GGUF \
#   Unlimited-OCR-2-AD-Q4_K.gguf mmproj-Unlimited-OCR-2-F16.gguf --local-dir .
```

| 文件 | 大小 | 说明 |
|---|---|---|
| `Unlimited-OCR-2-Q8_0.gguf` | 3.13 GB | 文本模型，推荐 |
| `mmproj-Unlimited-OCR-2-F16.gguf` | 0.83 GB | 视觉编码器（必需） |
| `Unlimited-OCR-2-AD-Q4_K.gguf` | 2.02 GB | 更省显存 |

### 3.3 创建 Python 虚拟环境

适配层只做 HTTP 转发与后处理，**不需要 torch/transformers**：

```bash
cd /data1/www/Unlimited-OCR
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python \
  fastapi "uvicorn[standard]" pydantic python-dotenv \
  python-multipart requests pillow pymupdf
```

### 3.4 配置 .env

```ini
# 监听地址/端口（systemd unit 通过 ${HOST}/${PORT} 引用）
HOST=0.0.0.0
PORT=18001

# API 认证（客户端 Authorization: Bearer <key>；留空则不校验）
API_KEY=ocr-xxxxxxxx

# 下游 llama-server
LLAMA_GPU=7
LLAMA_CTX=32768

# 可选：模型/引擎路径
# LLAMA_SERVER_BIN=/data1/www/llama-ocr2/build/bin/llama-server
# LLAMA_MODEL=/data1/models/Unlimited-OCR-2-GGUF/Unlimited-OCR-2-Q8_0.gguf
# LLAMA_MMPROJ=/data1/models/Unlimited-OCR-2-GGUF/mmproj-Unlimited-OCR-2-F16.gguf

# 可选：DRY 防重复（默认已开启，关闭会导致短句循环）
# DRY_MULTIPLIER=0.8
# DRY_BASE=1.75
# DRY_ALLOWED_LENGTH=2
# DRY_PENALTY_LAST_N=256

# 可选：空闲释放显存（秒，默认 900；设很大值=常驻）
# IDLE_UNLOAD_SECONDS=900
```

### 3.5 安装 systemd 服务

```bash
cd /data1/www/Unlimited-OCR
sudo cp unlimited-ocr-gguf.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now unlimited-ocr-gguf
sudo systemctl status unlimited-ocr-gguf
```

unit 关键点：

- `EnvironmentFile=-.env` 放在 `Environment=` 之后 → `.env` 覆盖默认值
- `ExecStart=... uvicorn server_gguf:app --host ${HOST} --port ${PORT}`
- `ExecStop` 发送 SIGTERM → FastAPI shutdown → 停止 llama-server 子进程

---

## 4. 验证

```bash
# 1) 健康检查（免认证；引擎未启动时 status=stopped）
curl http://localhost:18001/health

# 2) 认证与模型列表
curl -H "Authorization: Bearer <key>" http://localhost:18001/v1/models

# 3) 单图 OCR（本地路径）
curl -s http://localhost:18001/v1/chat/completions \
  -H "Authorization: Bearer <key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"Unlimited-OCR","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"/data/docs/page.png"}}]}],
      "max_tokens":4096}'

# 4) PDF 前 5 页（服务端自动转图）
curl -s http://localhost:18001/v1/chat/completions \
  -H "Authorization: Bearer <key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"Unlimited-OCR","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"/data/docs/report.pdf"}}]}],
      "max_pages":5,"page_mode":"batch"}'

# 5) 手动释放显存（下次请求自动拉起）
curl -X POST http://localhost:18001/admin/unload -H "Authorization: Bearer <key>"
```

预期：

- `status: stopped` → 首个请求后 `ok`（冷启动约 5s，模型页缓存命中）
- 输出为 Markdown（标题/正文/图片链接），图片保存在 `images/{req_id}/`
- `/images/{req_id}/xxx.jpg` 可直接访问

---

## 5. 性能实测（RTX 4090，Q8_0，ctx 32768）

| 场景 | prompt tokens | 生成 tokens | 耗时 |
|---|---|---|---|
| 单页 | 279 | 1500 | 5.5s |
| 双页 batch | 553 | 2500 | 9.3s |
| 三页 batch | 827 | 4600 | 18.8s |
| 5 页 PDF（3+2 分批） | — | — | 22.6s |

- 生成速度：约 250–285 t/s
- prompt 处理：约 900 t/s（首次）/ 3200 t/s（缓存命中）
- 显存：4k ctx 6.0GB / 8k 6.4GB / 32k 8.4GB

---

## 6. 与 vLLM FP8 方案的差异

| 项 | vLLM FP8（旧） | llama.cpp GGUF（当前） |
|---|---|---|
| 引擎 | docker 容器 `vllm-ocr` | 子进程 llama-server |
| 冷启动 | 30–40s | 约 5s |
| 并发 | 多请求 | **单请求串行**（`-np 1`） |
| 输出标签 | `<|det|>type [box]<|/det|>` | `<BLOCK>type[<LEVELn>][box]<FORK>文本`，`<MERGE>` 跨页合并 |
| 防重复 | vLLM NGram 处理器 | DRY 采样（`dry_*`） |
| SR-SWA 加速 | 完整支持 | 无（全 KV mask，结果等价） |
| 多页 batch | 支持 | 支持（受 `LLAMA_MAX_IMAGES_PER_REQ` 限制） |

对外 API、认证、PDF 转图、Markdown 输出规则、图片裁剪、`max_pages` /
`page_mode` / `stream` 行为均与旧版一致。

---

## 7. 常见问题

### 7.1 启动报 `unknown model architecture: 'unlimited-ocr2'`

用了主线 llama.cpp。必须使用 b11913 + 补丁编译的版本。

### 7.2 多页只输出第一页

请求里同时发了文本和图片。GGUF 内嵌 chat template 会自动生成
`<PAGE><image>...Multi page parsing.`；额外文本会被渲染到图片之前，导致页位置错乱。
适配层已处理（默认指令不发文本，自定义指令追加到图片后），
直接用 `server_gguf.py` 即可避免。

### 7.3 输出出现 `I'll. I'll. I'll. …` 循环

DRY 未开启。确认 `.env` 中 `DRY_MULTIPLIER` 未被设为 0（默认 0.8）。

### 7.4 启动日志有 `flash attention not supported by CUDA0`

补丁版在 CUDA 上未启用 FA，attention 走 mask 实现。只影响显存/速度，
不影响结果；上下文越大 mask 缓冲越高，建议按需设置 `LLAMA_CTX`。

### 7.5 图片没有出现在 Markdown 里

查看日志中的"图片裁剪跳过/全部失败"：坐标越界、全角逗号、原图损坏都会跳过该图，
不会生成死链。全角逗号已在解析层兼容。

### 7.6 空闲后首次请求很慢

属正常冷启动（加载 GGUF + mmproj），约 5s；模型文件在页缓存中时更快。
如需常驻，把 `IDLE_UNLOAD_SECONDS` 设大（如 `86400`）。

---

## 8. 运维命令

```bash
# 状态 / 日志
sudo systemctl status unlimited-ocr-gguf
sudo journalctl -u unlimited-ocr-gguf -f
tail -f /data1/www/Unlimited-OCR/log/llama-server.log

# 重启（改 .env 后）
sudo systemctl restart unlimited-ocr-gguf

# 健康 / 卸载
curl http://localhost:18001/health
curl -X POST http://localhost:18001/admin/unload -H "Authorization: Bearer <key>"

# 回退 vLLM FP8（需先停 GGUF 服务）
sudo systemctl disable --now unlimited-ocr-gguf
sudo systemctl enable --now unlimited-ocr-vllm
```
