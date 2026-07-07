# Unlimited-OCR 推理服务文档

OpenAI 兼容的 OCR 推理 API 服务，基于 HuggingFace Transformers。

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

```bash
cd /data/www/wwwroot/Unlimited-OCR
source .venv/bin/activate

# 手动启动
uvicorn server:app --host 0.0.0.0 --port 9705

# 或 systemd
sudo systemctl start unlimited-ocr
```

首次请求自动加载模型（~5-10s），空闲 15 分钟后自动卸载释放显存。

---

## API 接口

| 方法 | 路径 | 认证 | 说明 |
|------|------|:---:|------|
| GET | `/health` | 免 | 健康检查 |
| GET | `/v1/models` | ✅ | 模型列表 |
| POST | `/v1/chat/completions` | ✅ | OCR 推理 |
| POST | `/admin/unload` | ✅ | 手动卸载模型 |
| GET | `/images/{req_id}/{file}` | 免 | 静态图片服务 |

**基础 URL**: `http://localhost:9705`

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
    "http://localhost:9705/v1/chat/completions",
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
curl -s http://localhost:9705/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{
    "model": "Unlimited-OCR",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "<image>\nFree OCR."},
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
    base_url="http://localhost:9705/v1",
    api_key="your-key",
)

with open("document.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Unlimited-OCR",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "<image>\nFree OCR."},
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
    "http://localhost:9705/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer your-key",
    },
    json={
        "model": "Unlimited-OCR",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "<image>\nFree OCR."},
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
    "http://localhost:9705/v1/chat/completions",
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
    "http://localhost:9705/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer your-key",
    },
    json={
        "model": "Unlimited-OCR",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "<image>\nFree OCR."},
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

### 推理参数

| 参数 | 单图 | 多图/PDF |
|------|------|----------|
| 推理方法 | `model.infer()` | `model.infer_multi()` |
| image_size | 640 | 1024 |
| 动态分块 | ✅ (gundam) | ❌ (base) |
| ngram_window | 128 | 1024 |
| PDF DPI | — | 300 |
| 超时建议 | 300s | 600-1200s |

---

## Markdown 格式

### 标题映射

模型输出的 `<|det|>` 标签自动映射为标准 Markdown：

| det 类型 | Markdown | 示例 |
|----------|----------|------|
| `header` | `#` | `# CITYMAGAZINE / JUL 2009` |
| `title` | `##` | `## [Fiskars] 芬蘭釵剪` |
| `subtitle` | `###` | `### 产品详情` |
| `text` | 纯文本 | 段落自动合并，列表项自动识别 |
| `image` | `![]()` | 自动裁剪 + alt 文本 |
| `page_number` | 跳过 | — |

### 段落规则

| 上一行类型 | 当前行类型 | 行为 |
|-----------|-----------|------|
| text | text | 合并（不加空行） |
| text | title/header | 加空行 |
| text | image | 加空行 |
| image | image | 合并 |
| image | text | 加空行 |
| orphan | 任何 | 前后加空行 |

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

| Prompt | 用途 |
|--------|------|
| `<image>\nFree OCR.` | 通用文档解析（默认） |
| `<image>\ndocument parsing.` | 文档解析 |
| `<image>\nMulti page parsing.` | 多页/PDF（自动使用） |
| `<image>\nParse the figure.` | 图表解析 |
| `<image>\nExtract the text in the image.` | 纯文本提取 |

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

服务采用懒加载 + 空闲自动卸载：

```
COLD (显存 0) ──首次请求──► LOADING (~5-10s) ──就绪──► HOT (显存 ~6.4GB)
                                                          │
                                   空闲 > IDLE_UNLOAD_SECONDS
                                                          │
                                                          ▼
                                                     COLD (自动卸载)
```

### 管理命令

```bash
# 查看状态
curl http://localhost:9705/health | python3 -m json.tool

# 手动卸载
curl -X POST http://localhost:9705/admin/unload \
  -H "Authorization: Bearer your-key"
```

---

## 配置参考

```bash
# .env
MODEL_PATH=/data/www/models/Unlimited-OCR
SERVED_MODEL_NAME=Unlimited-OCR
HOST=0.0.0.0
PORT=9705
API_KEY=                    # 留空不校验
IDLE_UNLOAD_SECONDS=900     # 空闲超时(秒)
WATCHDOG_POLL_SECONDS=10    # 看门狗间隔(秒)
```

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
