# Unlimited-OCR 推理服务部署文档

OpenAI 兼容的 Unlimited-OCR 推理服务，基于 HuggingFace Transformers 后端。

## 环境要求

- Python 3.12+
- NVIDIA GPU，显存 >= 8 GB（模型 ~6.4 GB）
- NVIDIA Driver 570+ / CUDA 12.8
- uv（Python 包管理器）

## 快速开始

```bash
cd /data/www/wwwroot/Unlimited-OCR

# 1. 创建虚拟环境
uv venv .venv --python 3.12
source .venv/bin/activate

# 2. 安装 PyTorch（必须先从 PyTorch 官方 CUDA 12.8 索引安装）
uv pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128

# 3. 安装其余依赖
uv pip install -r requirements.txt
```

> 注意：`torch` 不能通过 `uv pip install -r requirements.txt` 自动安装，因为 PyPI 上的 torch 版本不含 CUDA 支持。必须先手动安装 CUDA 版本。

## 配置

通过 `.env` 文件设置（也支持环境变量覆盖）：

```bash
# .env
MODEL_PATH=/data/www/models/Unlimited-OCR
SERVED_MODEL_NAME=Unlimited-OCR
HOST=0.0.0.0
PORT=10000
API_KEY=
IDLE_UNLOAD_SECONDS=900
WATCHDOG_POLL_SECONDS=10
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `/data/www/models/Unlimited-OCR` | 模型本地路径 |
| `SERVED_MODEL_NAME` | `Unlimited-OCR` | API 返回的模型名称 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `10000` | 监听端口 |
| `API_KEY` | 空（不校验） | API 认证密钥，设置后需携带 `Authorization: Bearer <key>` |
| `IDLE_UNLOAD_SECONDS` | `900` | 空闲多少秒后自动卸载模型（15 分钟） |
| `WATCHDOG_POLL_SECONDS` | `10` | 看门狗检查空闲时间的间隔（秒） |

## 启动

### systemd（推荐）

```bash
# 安装服务
sudo ln -s /data/www/wwwroot/Unlimited-OCR/unlimited-ocr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now unlimited-ocr

# 管理
sudo systemctl status unlimited-ocr
sudo systemctl restart unlimited-ocr
sudo journalctl -u unlimited-ocr -f
```

### 手动

```bash
cd /data/www/wwwroot/Unlimited-OCR
source .venv/bin/activate
nohup .venv/bin/uvicorn server:app --host 0.0.0.0 --port 10000 > log/server.log 2>&1 &

# 停止
kill $(lsof -ti :10000)
```

## 懒加载与显存管理

服务启动时不加载模型。首次推理请求时按需加载，空闲超时后自动卸载以释放 GPU 显存。

- 模型加载耗时约 5-10 秒，之后的请求直接复用已加载的模型
- 卸载后 GPU 显存完全释放（约 6.4 GB），可用于其他任务
- 可通过 `POST /admin/unload` 立即手动卸载

```bash
# 查看模型加载状态和空闲时间
curl http://localhost:10000/health | python3 -m json.tool

# 手动卸载模型
curl -X POST http://localhost:10000/admin/unload
```

## API

与 DeepSeek-OCR-2 服务接口完全一致：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（含模型状态、空闲时间） |
| GET | `/v1/models` | 模型列表 |
| POST | `/v1/chat/completions` | 多模态 OCR 推理 |
| POST | `/admin/unload` | 手动卸载模型 |
| GET | `/images/{req_id}/{file}` | 静态图片服务 |

### Chat Completions

```bash
curl -s http://localhost:10000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Unlimited-OCR",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "<image>\nFree OCR."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }]
}' | python3 -m json.tool
```

### Python 调用

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:10000/v1", api_key="your-key")

with open("image.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Unlimited-OCR",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "<image>\nFree OCR."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]
    }],
    max_tokens=4096,
)
print(response.choices[0].message.content)
```

### Python 调用（流式）

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:10000/v1", api_key="your-key")
response = client.chat.completions.create(
    model="Unlimited-OCR",
    messages=[...],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Prompt 模式

| Prompt | 说明 |
|--------|------|
| `<image>\nFree OCR.` | 通用文档解析（推荐） |
| `<image>\ndocument parsing.` | 文档解析（与官方示例一致） |
| `<image>\nParse the figure.` | 图表解析 |
| `<image>\nExtract the text in the image.` | 纯文本提取 |

## 输出格式

OCR 结果以 Markdown 格式返回：

- 模型自动识别文档中的图片区域并裁剪提取
- 图片保存到 `images/{request_id}/` 目录，通过 `/images/` 路径提供服务
- Markdown 中的图片引用自动包含 alt 文本（取自最近的标题）
- 也生成 `result_with_boxes.jpg` 可视化框图

## 测试验证

```bash
cd /data/www/wwwroot/Unlimited-OCR
source .venv/bin/activate

# 发送测试图片
python -c "
import base64, json, requests

with open('test_image.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    'http://localhost:10000/v1/chat/completions',
    headers={'Content-Type': 'application/json'},
    json={
        'model': 'Unlimited-OCR',
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': '<image>\nFree OCR.'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}
            ]
        }],
        'max_tokens': 4096,
    },
    timeout=300,
)
resp.raise_for_status()
result = resp.json()
print(result['choices'][0]['message']['content'])
"
```

## 认证

`.env` 中设置 `API_KEY` 后，所有请求（除 `/health` 和 `/images/`）需携带 `Authorization` 头：

```bash
curl -H "Authorization: Bearer your-key" ...
```

```python
client = OpenAI(
    base_url="http://localhost:10000/v1",
    api_key="your-key",
)
```

## 与 DeepSeek-OCR-2 的差异

| 项目 | DeepSeek-OCR-2 | Unlimited-OCR |
|------|:---:|:---:|
| 端口 | 9705 | 10000 |
| 模型名称 | DeepSeek-OCR-2 | Unlimited-OCR |
| 推理速度 | ~5s | ~8-10s |
| 显存占用 | ~6.4 GB | ~6.4 GB |
| API 兼容 | - | 完全一致 |
| 输出格式 | Markdown | Markdown |
| 图片提取 | ✅ | ✅ |
| Alt 文本 | ✅ | ✅ |
| 懒加载/空闲卸载 | ✅ | ✅ |

## 项目文件

```
/data/www/wwwroot/Unlimited-OCR/
├── .venv/                  # Python 3.12 虚拟环境
├── server.py               # FastAPI 推理服务（核心文件）
├── .env                    # 环境配置
├── pyproject.toml          # 项目元数据
├── requirements.txt        # Python 依赖
├── unlimited-ocr.service   # systemd service 文件
├── images/                 # 运行时：提取的图片（按 req_id 子目录）
├── log/                    # 运行时：服务日志
├── infer.py                # SGLang 批量推理脚本（保留）
├── wheel/                  # SGLang wheel 包（保留）
└── docs/                   # 设计文档
```
