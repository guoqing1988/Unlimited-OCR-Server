# Unlimited-OCR 推理服务文档

OpenAI 兼容的 OCR 推理 API 服务，基于 HuggingFace Transformers。

## 目录

- [快速开始](#快速开始)
- [API 接口](#api-接口)
- [单图推理](#单图推理)
- [多图/PDF 推理](#多图pdf推理)
- [Markdown 输出格式](#markdown-输出格式)
- [认证](#认证)
- [生命周期管理](#生命周期管理)
- [配置参考](#配置参考)
- [与 DeepSeek-OCR-2 API 对比](#与-deepseek-ocr-2-api-对比)

---

## 快速开始

```bash
cd /data/www/wwwroot/Unlimited-OCR
source .venv/bin/activate

# 启动服务
uvicorn server:app --host 0.0.0.0 --port 9705

# 或通过 systemd
sudo systemctl start unlimited-ocr
```

首次推理请求时自动加载模型（~5-10s），空闲 15 分钟后自动卸载以释放显存。

---

## API 接口

| 方法 | 路径 | 认证 | 说明 |
|------|------|:---:|------|
| GET | `/health` | 免 | 健康检查（模型状态、空闲时间） |
| GET | `/v1/models` | ✅ | 模型列表 |
| POST | `/v1/chat/completions` | ✅ | OCR 推理（单图/多图/PDF） |
| POST | `/admin/unload` | ✅ | 手动卸载模型 |
| GET | `/images/{req_id}/{file}` | 免 | 静态图片服务 |

**基础 URL**: `http://localhost:9705`（端口可在 .env 中配置）

### Health 响应示例

```json
{
    "status": "ok",
    "model_loaded": true,
    "idle_seconds": 12.5,
    "loaded_at": 1783408863.0,
    "last_used": 1783408900.0,
    "idle_unload_limit": 900,
    "total_requests": 5
}
```

---

## 单图推理

发送单张图片进行 OCR 识别，使用 gundam 模式（base_size=1024, crop_mode=True）。

### cURL

```bash
curl -s http://localhost:9705/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
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

### Python (OpenAI SDK)

```python
import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9705/v1",
    api_key="your-api-key",
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
```

### Python (requests)

```python
import base64, json, requests

with open("document.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://localhost:9705/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer your-api-key",
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

### Prompt 参考

| Prompt | 用途 |
|--------|------|
| `<image>\nFree OCR.` | 通用文档解析（默认） |
| `<image>\ndocument parsing.` | 文档解析 |
| `<image>\nParse the figure.` | 图表解析 |
| `<image>\nExtract the text in the image.` | 纯文本提取 |

---

## 多图/PDF 推理

### 多图

发送多张 `image_url`（≥2），服务自动切换到 `infer_multi` 模式，**一次推理**处理所有页。

```python
import base64, json, requests

# 读取多个页面
pages = ["page1.jpg", "page2.jpg", "page3.jpg"]
content = [{"type": "text", "text": "<image>\nMulti page parsing."}]

for path in pages:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
    })

resp = requests.post(
    "http://localhost:9705/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer your-api-key",
    },
    json={
        "model": "Unlimited-OCR",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 16384,
    },
    timeout=600,
)
```

输出中页间用 `---` 分隔，每页独立：

```markdown
# 第1页标题
第1页内容...
![第1页标题](images/xxx/page_0/0.jpg)

---

# 第2页标题
第2页内容...
![第2页标题](images/xxx/page_1/0.jpg)
```

每页的图片保存在 `images/{req_id}/page_{页码}/` 下。

### PDF

直接传入 `.pdf` 文件路径，服务自动检测并使用 `pymupdf` 转换为图片（300 DPI）：

```python
resp = requests.post(
    "http://localhost:9705/v1/chat/completions",
    headers={"Content-Type": "application/json", "Authorization": "Bearer key"},
    json={
        "model": "Unlimited-OCR",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "<image>\nMulti page parsing."},
                {"type": "image_url", "image_url": {
                    "url": "/path/to/document.pdf"  # 本地 PDF 路径
                }},
            ],
        }],
        "max_tokens": 32768,
    },
    timeout=1200,
)
```

### 多图/PDF 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 推理模式 | base | image_size=1024, 无动态分块 |
| ngram_window | 1024 | 比单图(128)更大，防止跨页重复 |
| 图片格式 | 300 DPI PNG | PDF 转换质量 |
| 最大页数 | 无硬限制 | 受 context_length=32768 约束，实测 ≥40 页 |

---

## Markdown 输出格式

模型输出标准 Markdown 结构：

### 标题层级

原始 `<|det|>` 标签按类型自动映射：

| det 标签类型 | Markdown | 示例 |
|-------------|----------|------|
| `header` | `#` 一级标题 | `# CITYMAGAZINE / JUL 2009` |
| `title` | `##` 二级标题 | `## [Fiskars] 芬蘭釵剪` |
| `subtitle` | `###` 三级标题 | `### 产品详情` |
| `text` | 纯文本段落 | `Jenny: 我最鍾意...` |
| `image` | `![alt](路径)` | `![Fiskars](images/.../0.jpg)` |
| `page_number` | 跳过 | — |

### 图片提取

- 嵌入图片自动从原图裁剪并保存
- 图片 URL 可通过 `/images/{req_id}/...` 直接访问
- Alt 文本自动取最近的标题

### 示例输出

```markdown
Text: "Free Speech & Data"

# CITYMAGAZINE / JUL 2009 / SPY

## CHECKLIST
Text by Jenny & Jo Jo Photo by Leo Chan & Daniel Ho
![CHECKLIST](images/abc123/0.jpg)

## [Fiskars] 芬蘭釵剪
Jenny: 我最鍾意北歐設計刀具，Fiskars 這個牌子足有350年歷史...
![[Fiskars] 芬蘭釵剪](images/abc123/1.jpg)
```

---

## 认证

`.env` 中 `API_KEY` 非空时启用认证：

```bash
# .env
API_KEY=your-secret-key
```

- `/health` 和 `/images/` 白名单，无需认证
- 其他接口需携带 `Authorization: Bearer <key>` 请求头
- 认证失败返回 401 + OpenAI 兼容错误格式

---

## 生命周期管理

服务采用懒加载 + 空闲自动卸载模式：

```
COLD (模型未加载, 显存 ~0GB)
  │  首次请求
  ▼
LOADING (~5-10s)
  │
  ▼
HOT (模型已加载, 显存 ~6.4GB)
  │  空闲超过 IDLE_UNLOAD_SECONDS
  ▼
COLD (模型自动卸载)
```

### 管理命令

```bash
# 查看状态
curl http://localhost:9705/health | python3 -m json.tool

# 手动卸载（立即释放显存）
curl -X POST http://localhost:9705/admin/unload \
  -H "Authorization: Bearer your-key"
```

---

## 配置参考

完整 `.env` 配置项：

```bash
# 模型
MODEL_PATH=/data/www/models/Unlimited-OCR

# 服务
SERVED_MODEL_NAME=Unlimited-OCR
HOST=0.0.0.0
PORT=9705

# 认证（留空不校验）
API_KEY=

# 生命周期
IDLE_UNLOAD_SECONDS=900   # 空闲超时（秒），默认15分钟
WATCHDOG_POLL_SECONDS=10  # 看门狗检查间隔（秒）
```

---

## 与 DeepSeek-OCR-2 API 对比

| 项目 | DeepSeek-OCR-2 | Unlimited-OCR |
|------|:---:|:---:|
| 端口 | 9705 | 9705（可配置） |
| 模型名称 | DeepSeek-OCR-2 | Unlimited-OCR |
| API 格式 | OpenAI 兼容 | **完全相同** |
| 端点 | 5 个 | **完全相同** |
| 请求/响应格式 | — | **完全相同** |
| 认证方式 | Bearer Token | **完全相同** |
| 图片服务 | /images/ | **完全相同** |
| 懒加载/空闲卸载 | ✅ | ✅ |
| 单页 OCR | ✅ | ✅ |
| 多页/PDF | ❌ | ✅ |
| Markdown 标题 | ❌ 纯文本 | ✅ #/##/### 标题 |
| 图片 alt | ✅ | ✅ |
| 推理速度 | ~5s | ~8-12s |

### 迁移步骤

从 DS-OCR-2 切换到 Unlimited-OCR 只需改两个地方：

```diff
- client = OpenAI(base_url="http://localhost:9705/v1", api_key="key")
+ client = OpenAI(base_url="http://localhost:9705/v1", api_key="key")  # 同 URL

- model = "DeepSeek-OCR-2"
+ model = "Unlimited-OCR"  # 只改模型名
```

请求体和响应体格式完全一致，无需修改客户端代码逻辑。
