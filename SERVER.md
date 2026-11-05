# Unlimited-OCR 推理服务文档

OpenAI 兼容的 OCR 推理 API 服务。当前生产运行方式：**llama.cpp GGUF 后端**
（补丁版 llama-server 子进程 + Unlimited-OCR-2 Q8_0）+ `server_gguf.py` 适配层（FastAPI），
监听 **0.0.0.0:18001**（端口由 `.env` 的 `PORT` 控制，systemd unit 通过 `${PORT}` 展开）。
API 层面与 vLLM FP8 / Transformers 版完全一致（见文末差异说明）。

## 目录

- [快速开始](#快速开始)
- [API 接口](#api-接口)
- [输入方式](#输入方式)
- [多图/PDF 输出格式](#多图pdf输出格式)
- [Markdown 格式](#markdown-格式)
- [Prompt 参考](#prompt-参考)
- [认证](#认证)
- [生命周期](#生命周期)
- [配置参考](#配置参考)
- [与 DeepSeek-OCR-2 对比](#与-deepseek-ocr-2-对比)

---

## 快速开始

> 当前生产 = llama.cpp GGUF（Unlimited-OCR-2 Q8_0）。vLLM FP8 / Transformers 版已停用，可回退（见文末）。

```bash
cd /data1/www/Unlimited-OCR

# systemd（推荐，开机自启已 enable）
sudo systemctl start unlimited-ocr-gguf
sudo systemctl status unlimited-ocr-gguf
sudo journalctl -u unlimited-ocr-gguf -f

# 手动启动（调试用）
.venv/bin/uvicorn server_gguf:app --host 0.0.0.0 --port 18001
```

首次请求会自动拉起下游 llama-server 子进程（冷启动约 5s，模型页缓存命中）；
空闲 900s 后自动停止子进程释放显存，下次请求再次自动拉起。

> 引擎要求：打了 `unlimited-ocr2` 补丁的 llama.cpp（主线不支持该架构），
> 详见 [DEPLOY_GGUF.md](DEPLOY_GGUF.md)。

---

## API 接口

| 方法 | 路径 | 认证 | 说明 |
|------|------|:---:|------|
| GET | `/health` | 免 | 健康检查 |
| GET | `/v1/models` | ✅ | 模型列表 |
| POST | `/v1/chat/completions` | ✅ | OCR 推理 |
| POST | `/admin/unload` | ✅ | 手动卸载模型 |
| GET | `/images/{req_id}/{file}` | 免 | 静态图片服务 |

**基础 URL**: `http://localhost:18001`

### Health 响应

```json
{
    "status": "ok",
    "model_loaded": true,
    "idle_seconds": 12.5,
    "idle_unload_limit": 900,
    "total_requests": 5
}
```

---

## 输入方式

支持六种来源，自动检测单图/多图/PDF并选择最优推理路径。

### 1. base64 图片

```python
import base64
with open("document.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
```

### 2. 本地绝对路径（图片）

```python
{"type": "image_url", "image_url": {"url": "/home/liu/document.jpg"}}
```

### 3. 本地绝对路径（PDF）

```python
{"type": "image_url", "image_url": {"url": "/data/www/docs/report.pdf"}}
```

服务端用 pymupdf 自动转换为 300 DPI 图片后推理。

### 4. HTTP/HTTPS URL（图片）

```python
{"type": "image_url", "image_url": {"url": "https://cdn.example.com/page.jpg"}}
```

### 5. HTTP/HTTPS URL（PDF）

```python
{"type": "image_url", "image_url": {"url": "https://cdn.example.com/report.pdf"}}
```

服务端下载后根据扩展名判断类型。

### 6. 多图数组（base64 + URL + 路径混合）

发送 ≥2 个 `image_url` 时自动切换到多图模式，一次推理处理所有页：

```python
resp = requests.post(
    "http://localhost:18001/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer your-key",
    },
    json={
        "model": "Unlimited-OCR",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "<image>\nMulti page parsing."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "image_url", "image_url": {"url": "https://cdn.com/page3.jpg"}},
            {"type": "image_url", "image_url": {"url": "/home/liu/page4.jpg"}},
        ]}],
        "max_tokens": 16384,
    },
    timeout=600,
)
```

### 输入汇总表

| 来源 | 单图 | PDF | 多图 |
|------|:---:|:---:|:---:|
| `data:image/...;base64,...` | ✅ | — | ✅ |
| `/absolute/path/file.jpg` | ✅ | — | ✅ |
| `/absolute/path/file.pdf` | — | ✅ | — |
| `http(s)://host/file.jpg` | ✅ | — | ✅ |
| `http(s)://host/file.pdf` | — | ✅ | — |

> 注意：不支持相对路径（如 `./document.jpg`），请使用绝对路径。

---

## 完整请求示例

### cURL（单图 base64）

```bash
curl -s http://localhost:18001/v1/chat/completions \
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
    }],
    "max_tokens": 4096
  }' | python3 -m json.tool
```

### Python（OpenAI SDK，单图）

```python
import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:18001/v1",
    api_key="your-key",
)

with open("document.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Unlimited-OCR",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "<image>document parsing."},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }},
        ],
    }],
    max_tokens=4096,
)
print(response.choices[0].message.content)

# 流式输出
response = client.chat.completions.create(
    model="Unlimited-OCR",
    messages=[...],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Python（requests，单图）

```python
import base64, json, requests

with open("document.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://localhost:18001/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer your-key",
    },
    json={
        "model": "Unlimited-OCR",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "<image>document parsing."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }},
            ],
        }],
        "max_tokens": 4096,
    },
    timeout=300,
)
resp.raise_for_status()
print(resp.json()["choices"][0]["message"]["content"])
```

### Python（本地绝对路径，PDF）

```python
import json, requests

resp = requests.post(
    "http://localhost:18001/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer your-key",
    },
    json={
        "model": "Unlimited-OCR",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "<image>\nMulti page parsing."},
                {"type": "image_url", "image_url": {
                    "url": "/data/www/docs/report.pdf"
                }},
            ],
        }],
        "max_tokens": 32768,
    },
    timeout=1200,
)
resp.raise_for_status()
print(resp.json()["choices"][0]["message"]["content"])
```

### Python（HTTP URL，远程图片）

```python
import json, requests

resp = requests.post(
    "http://localhost:18001/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer your-key",
    },
    json={
        "model": "Unlimited-OCR",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "<image>document parsing."},
                {"type": "image_url", "image_url": {
                    "url": "https://cdn.example.com/page.jpg"
                }},
            ],
        }],
        "max_tokens": 4096,
    },
    timeout=300,
)
```

---

## 多图/PDF 输出格式

多图或 PDF 推理时，输出页间用 `---` 分隔，每页独立提取图片：

```markdown
# 第1页标题
第1页内容...
![第1页标题](images/xxx/page_0/0.jpg)

---

# 第2页标题
第2页内容...
![第2页标题](images/xxx/page_1/0.jpg)
```

每页图片保存在 `images/{req_id}/page_{页码}/`。

### 多图模式选项

服务支持两种多图推理模式，通过 `page_mode` 参数控制：

| 选项 | 方法 | image_size | 速度 | 质量 | 跨页上下文 | 适用场景 |
|------|------|:---:|:---:|:---:|:---:|------|
| `batch`（默认） | infer_multi | 1024 | 基准 | 2513字/2页 | ✅ | 跨页表格、连续文章 |
| `single` | infer (gundam) | 640 | 快13% | +44%字数 | ❌ | 独立页面、追求质量 |

#### `max_pages` — 限制处理页数

```python
# 只处理前5页（适合预览或限制 token 消耗）
{"model": "Unlimited-OCR", "max_pages": 5, "messages": [...]}

# 不限制（默认，处理全部页面）
{"model": "Unlimited-OCR", "messages": [...]}
```

#### `page_mode` — 选择推理策略

```python
# 批量模式（默认）— 跨页上下文连贯
{"model": "Unlimited-OCR", "page_mode": "batch", "messages": [...]}

# 逐张模式 — 每页独立推理，gundam 640 高质量
{"model": "Unlimited-OCR", "page_mode": "single", "messages": [...]}
```

#### 完整示例：100页PDF，只处理前10页，逐张高质量

```python
resp = requests.post(
    "http://localhost:18001/v1/chat/completions",
    headers={"Content-Type": "application/json", "Authorization": "Bearer key"},
    json={
        "model": "Unlimited-OCR",
        "max_pages": 10,
        "page_mode": "single",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "<image>document parsing."},
            {"type": "image_url", "image_url": {"url": "/data/large-report.pdf"}},
        ]}],
        "max_tokens": 32768,
    },
    timeout=600,
)
```

### 推理参数

| 参数 | 单图 | 多图 batch | 多图 single |
|------|------|------------|-------------|
| 引擎调用 | `llama_chat()` 单图 | `llama_chat()` 多图 | `llama_chat()` 逐页循环 |
| 图片尺寸 | 原始 | 模型预处理器（最多 16 tiles） | 同左 |
| 采样 | `temperature` 请求值 | 同左 | 同左 |
| 防重复 | DRY（`dry_multiplier=0.8` 等） | 同左 | 同左 |
| 单请求图片数 | 1 | ≤ `LLAMA_MAX_IMAGES_PER_REQ`（3） | 1/页 |
| PDF DPI | — | 300 | 300 |
| 超时建议 | 300s | 600-1200s | 300s/页 |

---

## Markdown 格式

### 标题映射

模型输出的 `<BLOCK>` 标签自动映射为标准 Markdown（v2 输出格式，适配层已兼容）：

| BLOCK 类型 | Markdown | 示例 |
|----------|----------|------|
| `header` | `#` | `# CITYMAGAZINE / JUL 2009` |
| `title` | `##` | `## [Fiskars] 芬蘭釵剪` |
| `title<LEVEL2>` | `###` | `### 子标题` |
| `subtitle` | `###` | `### 产品详情` |
| `text` | 纯文本 | 段落自动合并，列表项自动识别 |
| `image_caption` | 纯文本 | 图注按正文输出 |
| `image` | `![]()` | 自动裁剪 + alt 文本（取最近标题） |
| `footer` / `page_number` | 跳过 | — |

跨页续写由模型输出 `<MERGE>` 标记，适配层据此合并为同一段落（不插空行）。
若图片坐标无效导致裁剪失败，该图片不输出链接（避免 404 死链）。

### 段落规则

| 上一行类型 | 当前行类型 | 行为 |
|-----------|-----------|------|
| text | text | **加空行**（v2 每个 `<BLOCK>text` 是独立段落） |
| text | text（`<MERGE>`） | 不加空行（跨页续写，同一段落） |
| text | title/header | 加空行 |
| text | image | 加空行 |
| image | image | 合并 |
| image | text | 加空行 |
| 列表项 | 列表项 | 不加空行（List 紧凑） |
| orphan | 任何 | 前后加空行 |

> 标准 Markdown 中单个换行只是软换行，渲染时会被并成同一段；
> 所以适配层用空行分隔段落，保证渲染后每个文本块独立成段。

### 重复图片去重

v2 模型在双栏 + 大图版面偶发把同一张图输出两次（坐标几乎相同，中间还可能夹着正文），
直接裁剪会得到两张一模一样的图片。适配层对本页已输出图片做 IoU 去重：

- 与已输出图片 IoU ≥ `IMAGE_DEDUP_IOU`（默认 `0.85`）→ 跳过不输出
- 不同位置的图片不受影响（IoU 很低）
- 设 `IMAGE_DEDUP_IOU=0` 可关闭去重

### 列表识别

以数字序号（`01`, `02`）、`-`、`*`、`•` 开头的文本自动转为 Markdown 列表项：

```markdown
- 01 典型巴洛克時期風格設計的產地燈。
- 02 連角的羊頭骨畫飾。
- 03 典型的歐洲農夫家庭木餐椅。
```

### 示例输出

```markdown
# CITYMAGAZINE / JUL 2009 / SPY

## CHECKLIST
Text by Jenny & Jo Jo Photo by Leo Chan & Daniel Ho
![CHECKLIST](images/abc123/0.jpg)

## [Fiskars] 芬蘭釵剪
Jenny: 我最鍾意北歐設計刀具，Fiskars 這個牌子足有350年歷史...
![[Fiskars] 芬蘭釵剪](images/abc123/1.jpg)
```

---

## Prompt 参考

Unlimited-OCR 使用 `<image>` 标记指定图片位置，支持多种 prompt 适配不同场景。

### 内置 Prompt 预设

> **v2 (GGUF) 后端说明**：当前生产后端为 Unlimited-OCR-2，适配层会把官方默认指令
> `document parsing.` / `Multi page parsing.` 统一**过滤掉**，交给模型内嵌模板补
> `Multi page parsing.`。实测显式发 `document parsing.` 会让 v2 模型进入错误解析路径、
> 输岀幻觉循环（同一张问卷图：模板默认 40 行正确 vs `document parsing.` 43 行全幻觉）。
> 因此调用时**不传文本**或传上面两个默认指令均可；只有自定义指令（如 `请提取表格`）
> 才会真正发送。

| Prompt | 用途 | 输出特点 |
|--------|------|----------|
| `<image>document parsing.` | **通用文档解析（默认）** | 自动识别标题、正文、图片，输出结构化 Markdown |
| `<image>\n<\|grounding\|>Convert the document to markdown.` | 结构化 Markdown | 同 DS-OCR-2 的 grounding 模式 |
| `<image>\ndocument parsing.` | 文档解析（换行变体） | 效果与无换行版等同 |
| `<image>\nMulti page parsing.` | 多页文档/PDF | 多图时自动使用，页间 `<PAGE>` 分隔 |
| `<image>\nParse the figure.` | 图表解析 | 专注图表结构提取 |
| `<image>\nExtract the text in the image.` | 纯文本提取 | 跳过布局分析，仅提取文字 |

### 使用示例

```python
# 默认通用模式（推荐）
{"type": "text", "text": "<image>document parsing."}

# 指定图片位置，自由组合提示词
{"type": "text", "text": "<image>\n识别图片中的表格内容。"}
{"type": "text", "text": "<image>\n提取所有文字，保留原有格式。"}

# 多图模式（≥2 张图片时自动切换，无需手动指定 prompt）
```

> `<image>` 是视觉编码器的占位标记，prompt 中必须至少包含一个。 |

---

## 认证

`.env` 中 `API_KEY` 非空时启用认证：

```bash
# .env
API_KEY=your-secret-key
```

- `/health` 和 `/images/` 为白名单，无需认证
- 其他接口需携带 `Authorization: Bearer <key>`
- 认证失败返回 401 + OpenAI 兼容错误格式

```python
# 请求示例
headers = {"Authorization": "Bearer your-key"}
```

---

## 生命周期

llama-server 以子进程运行，适配层负责生命周期（懒加载 + 空闲自动卸载）：

```
COLD (引擎未启动, 显存 0) ──首次请求──► 拉起 llama-server ──► LOADING (~5s) ──► HOT (Q8_0, 6-8.4GB)
                                                                              │
                                            空闲 > IDLE_UNLOAD_SECONDS (900s)
                                                                              │
                                                                              ▼
                                                          terminate 子进程（显存释放回 COLD）
```

模型文件（3.13GB + 0.83GB）通过 mmap 保留在系统页缓存，重新启动时无需磁盘读取，
所以冷启动很快；内存充裕的机器不必担心页缓存占用。

### 管理命令

```bash
# 查看状态（status: ok=引擎就绪 / stopped=已空闲卸载 / degraded=引擎异常）
curl http://localhost:18001/health | python3 -m json.tool

# 手动卸载（停止 llama-server，释放显存）
curl -X POST http://localhost:18001/admin/unload \
  -H "Authorization: Bearer your-key"

# 查看引擎进程与日志
pgrep -af llama-server
tail -f /data1/www/Unlimited-OCR/log/llama-server.log
```

---

## 配置参考

适配层通过环境变量 / service 文件配置。systemd unit 内 `Environment=` 提供默认值，
`EnvironmentFile=.env` 放在其后，同名变量以 `.env` 为准。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | `0.0.0.0` | 监听地址（unit 通过 `${HOST}` 展开） |
| `PORT` | `18001` | 监听端口（unit 通过 `${PORT}` 展开） |
| `API_KEY` | 空 | API 认证密钥（空则不校验） |
| `LLAMA_URL` | `http://127.0.0.1:9706/v1` | 下游 llama-server OpenAI 端点 |
| `LLAMA_SERVER_BIN` | `/data1/www/llama-ocr2/build/bin/llama-server` | 补丁版引擎可执行文件 |
| `LLAMA_MODEL` | `.../Unlimited-OCR-2-Q8_0.gguf` | 文本模型 GGUF |
| `LLAMA_MMPROJ` | `.../mmproj-Unlimited-OCR-2-F16.gguf` | 视觉编码器 GGUF |
| `LLAMA_GPU` | `0` | 引擎使用的 GPU 编号 |
| `LLAMA_CTX` | `32768` | 引擎上下文长度（4k≈6.0GB / 32k≈8.4GB 显存） |
| `LLAMA_MAX_TOKENS` | `8192` | 单次生成 token 上限 |
| `LLAMA_MAX_IMAGES_PER_REQ` | `3` | batch 模式每批最多图片数 |
| `LLAMA_START_TIMEOUT` | `240` | 等待引擎就绪超时(秒) |
| `DRY_MULTIPLIER` | `0.8` | DRY 采样强度（0=关闭，会循环） |
| `DRY_BASE` | `1.75` | DRY 基础惩罚 |
| `DRY_ALLOWED_LENGTH` | `2` | DRY 允许重复长度 |
| `DRY_PENALTY_LAST_N` | `256` | DRY 回看窗口 |
| `IMAGE_DEDUP_IOU` | `0.85` | 重复图片去重阈值（IoU，0=关闭） |
| `ENGINE_MODEL_NAME` | `Unlimited-OCR` | 引擎 alias（`--alias`） |
| `SERVED_MODEL_NAME` | `Unlimited-OCR` | API 响应的模型名 |
| `IDLE_UNLOAD_SECONDS` | `900` | 空闲后停止引擎(秒) |
| `WATCHDOG_POLL_SECONDS` | `10` | 看门狗检查间隔 |
| `MAX_PAGES_PER_BATCH` | `20` | PDF 每批最多页数 |

---

## 与 DeepSeek-OCR-2 对比

| 项目 | DS-OCR-2 | Unlimited-OCR |
|------|:---:|:---:|
| API 格式 | OpenAI 兼容 | **完全相同** |
| 端点 | 5 个 | **完全相同** |
| 请求/响应格式 | — | **完全相同** |
| 认证方式 | Bearer Token | **完全相同** |
| 懒加载/空闲卸载 | ✅ | ✅ |
| 单图 OCR | ✅ | ✅ |
| 多图/PDF | ❌ | ✅ |
| Markdown 标题 | ❌ 纯文本 | ✅ `#`/`##`/`###` |
| 列表识别 | ❌ | ✅ |
| 段落合并 | ❌ | ✅ |
| 图片 alt | ✅ | ✅ |

### 迁移

```diff
- model = "DeepSeek-OCR-2"
+ model = "Unlimited-OCR"
```

请求体和响应体格式完全一致，只需改模型名。

---

## 后端差异：llama.cpp GGUF vs vLLM FP8（旧）

当前服务（`server_gguf.py` + llama-server 子进程）与上一版（`server_vllm.py` + docker）
在 **API 层面完全一致**（认证/端点/输入方式/Markdown 输出/图片提取均相同），
内部差异：

| 维度 | vLLM FP8（旧） | llama.cpp GGUF（当前） |
|------|:---:|:---:|
| 单页耗时 | ~3.3s | ~5s |
| 显存（活跃） | ~8GB | 6.0GB（4k）/ 8.4GB（32k） |
| 显存（空闲） | 容器停止，全释放 | 子进程停止，全释放 |
| 冷启动 | ~30-40s | ~5s |
| 并发 | 多请求 | 单请求串行（`-np 1`） |
| 长文档 SR-SWA | 完整 | 无加速（全 KV mask，结果等价） |
| 防重复 | vLLM NGram 处理器 | DRY 采样 |
| 输出标签 | `<\|det\|>type [box]<\|/det\|>` | `<BLOCK>type[<LEVELn>][box]<FORK>文本`，`<MERGE>` 跨页合并 |
| 质量 | 基准 | Q8_0 与 bf16 基本一致 |

### 回退到 vLLM FP8 版

```bash
# 1. 停 GGUF 适配层（并停止 llama-server 子进程）
sudo systemctl disable --now unlimited-ocr-gguf

# 2. 启用 vLLM 版（需 vllm-ocr 容器与 v1 权重就位）
sudo systemctl enable --now unlimited-ocr-vllm
```

