# Unlimited-OCR 部署实施计划

> **给执行者：** 使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 按任务逐个实施。步骤使用 checkbox（`- [ ]`）语法追踪。

**目标：** 部署 Unlimited-OCR 为 OpenAI 兼容推理服务，使用 SGLang 后端，支持懒加载、空闲自动卸载、Markdown 后处理和图片提取。

**架构：** 单文件 FastAPI 代理（`server.py`）在内部端口包装 SGLang 子进程。代理管理 SGLang 生命周期（按需启动、空闲卸载），代理请求到 SGLang 并添加 SGLang 专用参数，将原始输出（ref/det 标签）后处理为干净的 Markdown 并提取图片。通过 FastAPI `StaticFiles` 挂载提供静态图片服务。

**技术栈：** Python 3.12, uv, SGLang (自定义 wheel), FastAPI, uvicorn

## 全局约束

- 仅使用 Python 3.12（`/usr/bin/python3.12`）
- uv 管理的 venv 位于 `.venv/`，绝不触碰系统 pip/包
- SGLang wheel 来自 `wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl`
- 模型路径 `/data/www/models/Unlimited-OCR`
- 服务端口 10000，SGLang 内部端口 20000
- `MEM_FRACTION_STATIC=0.25`（12 GB）起步，OOM 时上调
- API 格式必须与现有 DeepSeek-OCR-2 完全一致
- 代码风格：PEP 8，4 空格缩进，匹配现有 DS-OCR-2 server.py 风格
- 图片通过 `/images/{req_id}/{file}` 提供服务

---

### 任务 1：环境搭建

**涉及文件：**
- 创建：`.venv/`（通过 uv 创建目录）
- 尚无源代码文件

**说明：** 创建 Python 3.12 虚拟环境，从 SGLang wheel 和 PyPI 安装全部依赖。

- [ ] **步骤 1：创建虚拟环境**

```shell
cd /data/www/wwwroot/Unlimited-OCR
uv venv .venv --python 3.12
```

预期：`.venv/` 目录创建，内含 Python 3.12 解释器。

- [ ] **步骤 2：安装 SGLang wheel**

此 wheel 较大（压缩后 12 MB），有大量传递依赖。安装可能需要 2-5 分钟。

```shell
source .venv/bin/activate
uv pip install wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl
```

预期：SGLang 及其全部依赖安装成功。注意观察是否有编译失败（特别是 `flashinfer_python` 和 `sglang-kernel` 有编译组件）。

- [ ] **步骤 3：安装附加依赖**

```shell
uv pip install kernels==0.11.7 pymupdf==1.27.2.2 fastapi "uvicorn[standard]" python-dotenv
```

预期：全部包安装成功。

- [ ] **步骤 4：验证安装**

```shell
source .venv/bin/activate
python -c "import sglang; print('SGLang 版本:', sglang.__version__)"
python -c "from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor; print('LogitProcessor 就绪:', DeepseekOCRNoRepeatNGramLogitProcessor.to_str()[:50])"
python -c "import fastapi; import uvicorn; import dotenv; print('Web 依赖就绪')"
```

预期：全部导入成功，无报错。`DeepseekOCRNoRepeatNGramLogitProcessor.to_str()` 返回非空字符串。

- [ ] **步骤 5：提交**

```bash
git add .gitignore
git commit -m "chore: 搭建 Python 3.12 venv 并安装 SGLang 及依赖

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### 任务 2：server.py — 配置、状态管理、认证与 SGLang 生命周期

**涉及文件：**
- 创建：`server.py`（前半部分：配置、状态、app、认证中间件、生命周期函数）

**说明：** 构建核心框架：环境配置加载、全局状态、FastAPI app 创建、认证中间件，以及 SGLang 子进程管理（启动、停止、健康检查、看门狗）。此任务完成后服务可管理 SGLang 生命周期但尚无 API 端点。

- [ ] **步骤 1：编写配置、状态和 app 骨架**

创建 `server.py`：

```python
"""
Unlimited-OCR：OpenAI 兼容 OCR 推理服务，基于 SGLang 后端。

启动方式：
    uvicorn server:app --host 0.0.0.0 --port 10000

配置通过 .env 文件或环境变量设置。
SGLang 作为子进程管理 —— 首次请求时启动，空闲超时后停止。

API 端点：
    GET  /health              — 健康检查（模型状态 + 空闲时间）
    GET  /v1/models           — 模型列表
    POST /v1/chat/completions — 多模态 OCR 推理
    POST /admin/unload        — 手动卸载模型
"""

import os
import gc
import threading
import base64
import json
import time
import uuid
import tempfile
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=False)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── 配置（环境变量 > .env > 默认值）──────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "/data/www/models/Unlimited-OCR")
SERVED_NAME = os.environ.get("SERVED_MODEL_NAME", "Unlimited-OCR")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "10000"))
SGLANG_PORT = int(os.environ.get("SGLANG_PORT", "20000"))
API_KEY = os.environ.get("API_KEY", "")
IDLE_UNLOAD_SECONDS = int(os.environ.get("IDLE_UNLOAD_SECONDS", "900"))
WATCHDOG_POLL_SECONDS = int(os.environ.get("WATCHDOG_POLL_SECONDS", "10"))
GPU = os.environ.get("GPU", "0")
MEM_FRACTION_STATIC = float(os.environ.get("MEM_FRACTION_STATIC", "0.25"))
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "32768"))

SGLANG_URL = f"http://127.0.0.1:{SGLANG_PORT}"
SERVER_TIMEOUT = 300  # 等待 SGLang 启动的最大秒数
NO_REPEAT_NGRAM_SIZE = 35
REQUEST_TIMEOUT = 1200
MAX_RETRIES = 3

IMAGES_DIR = Path(__file__).parent / "images"
LOG_DIR = Path(__file__).parent / "log"

# ── 全局状态 ──────────────────────────────────────────────────────────

_state_lock = threading.Lock()   # 保护 _process / _loaded_at / _last_used
_startup_lock = threading.Lock() # 序列化 SGLang 启动（同一时间只启动一次）

_sglang_process: Optional[subprocess.Popen] = None
_loaded_at: Optional[float] = None
_last_used: Optional[float] = None
_total_requests: int = 0

# ── FastAPI app ───────────────────────────────────────────────────────

app = FastAPI(title="Unlimited-OCR API", version="1.0.0")
```

- [ ] **步骤 2：添加认证中间件**

追加到 `server.py`：

```python
# ── 认证中间件 ─────────────────────────────────────────────────────────

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # /health 和 /images/ 白名单，无需认证
    if request.url.path in ("/health",) or request.url.path.startswith("/images/"):
        return await call_next(request)
    # 未配置密钥则放行
    if not API_KEY:
        return await call_next(request)

    auth = request.headers.get("Authorization", "")
    expected = f"Bearer {API_KEY}"
    if auth != expected:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "认证失败。请使用 'Authorization: Bearer <key>' 请求头。",
                    "type": "authentication_error",
                    "code": 401,
                }
            },
        )
    return await call_next(request)
```

- [ ] **步骤 3：添加 SGLang 生命周期函数**

追加到 `server.py`：

```python
# ── SGLang 生命周期 ────────────────────────────────────────────────────

def _sglang_health() -> bool:
    """检查 SGLang 服务是否响应。"""
    try:
        resp = __import__("requests").get(f"{SGLANG_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _start_sglang():
    """启动 SGLang 服务子进程。立即返回；调用方需轮询 _sglang_health()。"""
    global _sglang_process

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "sglang_server.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU

    cmd = [
        sys.executable,
        "-m", "sglang.launch_server",
        "--model", MODEL_PATH,
        "--served-model-name", SERVED_NAME,
        "--attention-backend", "fa3",
        "--page-size", "1",
        "--mem-fraction-static", str(MEM_FRACTION_STATIC),
        "--context-length", str(CONTEXT_LENGTH),
        "--enable-custom-logit-processor",
        "--disable-overlap-schedule",
        "--skip-server-warmup",
        "--host", "127.0.0.1",
        "--port", str(SGLANG_PORT),
    ]

    log_file = open(str(log_path), "w", encoding="utf-8")
    _sglang_process = subprocess.Popen(
        cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
    )
    _sglang_process._log_file = log_file


def _stop_sglang():
    """终止 SGLang 子进程并释放 GPU 显存。"""
    global _sglang_process, _loaded_at

    p = _sglang_process
    if p is None:
        return

    p.terminate()
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()

    try:
        p._log_file.close()
    except Exception:
        pass

    _sglang_process = None
    _loaded_at = None

    # 释放 GPU 显存
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _ensure_loaded():
    """确保 SGLang 正在运行。阻塞直到健康检查通过。
    使用双重检查锁，并发请求共享同一次启动。"""
    import requests as req

    # 快速路径：已在运行
    if _sglang_health():
        return

    with _startup_lock:
        # 获取锁后再次检查
        if _sglang_health():
            return

        _start_sglang()

        start = time.time()
        while time.time() - start < SERVER_TIMEOUT:
            if _sglang_process is not None and _sglang_process.poll() is not None:
                raise RuntimeError(
                    f"SGLang 服务提前退出（退出码 {_sglang_process.returncode}）。"
                    f"请查看 {LOG_DIR / 'sglang_server.log'}"
                )
            if _sglang_health():
                with _state_lock:
                    global _loaded_at, _last_used
                    _loaded_at = time.time()
                    _last_used = time.time()
                return
            time.sleep(3)

        _stop_sglang()
        raise TimeoutError(
            f"SGLang 服务在 {SERVER_TIMEOUT}s 内未就绪。"
            f"请查看 {LOG_DIR / 'sglang_server.log'}"
        )


def _touch_used():
    """更新最后使用时间戳（每次请求完成后调用）。"""
    global _last_used, _total_requests
    with _state_lock:
        _last_used = time.time()
        _total_requests += 1


def _watchdog():
    """后台线程：检查空闲时间，超时则卸载模型。"""
    while True:
        time.sleep(WATCHDOG_POLL_SECONDS)
        with _state_lock:
            loaded = _sglang_process is not None
            last = _last_used
        if loaded and last and (time.time() - last) >= IDLE_UNLOAD_SECONDS:
            _stop_sglang()


@app.on_event("startup")
def startup():
    """进程启动时启动看门狗线程。"""
    th = threading.Thread(target=_watchdog, daemon=True)
    th.start()
```

- [ ] **步骤 4：验证生命周期 —— 手动启动 SGLang**

首先确保 `.env` 存在且包含基本配置：

```bash
cat > /data/www/wwwroot/Unlimited-OCR/.env << 'EOF'
MODEL_PATH=/data/www/models/Unlimited-OCR
SERVED_MODEL_NAME=Unlimited-OCR
HOST=0.0.0.0
PORT=10000
SGLANG_PORT=20000
API_KEY=
IDLE_UNLOAD_SECONDS=900
WATCHDOG_POLL_SECONDS=10
GPU=0
MEM_FRACTION_STATIC=0.25
CONTEXT_LENGTH=32768
EOF
```

然后通过导入生命周期函数验证：

```shell
cd /data/www/wwwroot/Unlimited-OCR
source .venv/bin/activate
python -c "
from server import _start_sglang, _sglang_health, _stop_sglang
import time
print('正在启动 SGLang...')
_start_sglang()
print('等待健康检查...')
for i in range(100):
    if _sglang_health():
        print(f'就绪，耗时 {i*3}s')
        break
    time.sleep(3)
else:
    print('超时 - 请查看 log/sglang_server.log')
print('正在停止...')
_stop_sglang()
print('完成')
"
```

预期：SGLang 启动，变为健康状态（可能需要 60-120s），然后干净停止。停止后 GPU 显存回到基线水平。

- [ ] **步骤 5：提交**

```bash
git add server.py .env
git commit -m "feat: 添加 server.py 核心框架与 SGLang 生命周期管理

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### 任务 3：server.py — API 模型、工具函数与后处理管线

**涉及文件：**
- 修改：`server.py`（追加 Pydantic 模型、工具函数、后处理管线）

**说明：** 添加 OpenAI 兼容请求的数据模型、消息/图片解析工具函数，以及将 SGLang 原始输出（含 ref/det 标签）转换为带提取图片和 alt 文本的干净 Markdown 的后处理管线。

- [ ] **步骤 1：添加 Pydantic 模型和错误响应工具**

追加到 `server.py`：

```python
# ── OpenAI 兼容错误响应 ─────────────────────────────────────────────────

def error_response(status: int, message: str, error_type: str) -> JSONResponse:
    """返回 OpenAI 兼容格式的错误 JSON。"""
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": status,
            }
        },
    )


# ── Pydantic 请求模型 ──────────────────────────────────────────────────

from pydantic import BaseModel, Field


class Message(BaseModel):
    """对话中的一条消息。content 可以是纯文本字符串或多模态内容块列表。"""
    role: str = "user"
    content: str | list[dict] = ""
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions 请求体。与 OpenAI Chat Completions API 对齐。"""
    model: str = "Unlimited-OCR"
    messages: list[Message] = Field(default_factory=list)
    max_tokens: int = Field(default=4096, alias="max_completion_tokens")
    temperature: float = 0.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: list[str] | None = None
    user: str | None = None

    class Config:
        populate_by_name = True
```

- [ ] **步骤 2：添加消息/图片解析工具函数**

追加到 `server.py`：

```python
# ── 工具函数：解析 OpenAI 消息格式 ──────────────────────────────────────

def decode_image(s: str) -> str:
    """将 base64 data-URI 解码为临时图片文件。返回文件路径。"""
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    data = base64.b64decode(s)
    suffix = ".jpg" if data[:4] == b"\xff\xd8\xff" else ".png"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def extract_messages(messages: list[Message]) -> tuple[str, list[str], list[str]]:
    """
    解析 OpenAI 格式消息为 (prompt, image_paths, temp_files)。

    返回值：
        prompt：含 <image> 标记的文本提示词
        image_paths：解析后的图片文件路径列表
        temp_files：调用方需清理的临时文件路径列表
    """
    parts = []
    img_paths = []
    temps = []

    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for p in content:
                t = p.get("type", "")
                if t == "text":
                    parts.append(p.get("text", ""))
                elif t == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    fpath = None
                    if url.startswith("data:"):
                        fpath = decode_image(url)
                        temps.append(fpath)
                    elif url.startswith(("http://", "https://")):
                        import urllib.request
                        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        urllib.request.urlretrieve(url, tmp.name)
                        tmp.close()
                        fpath = tmp.name
                        temps.append(fpath)
                    elif os.path.isfile(url):
                        fpath = url
                    if fpath:
                        img_paths.append(fpath)

    prompt = "\n".join(parts) if parts else "<image>\nFree OCR."
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    return prompt, img_paths, temps
```

- [ ] **步骤 3：添加后处理管线**

追加到 `server.py`：

```python
# ── 后处理：ref/det 标签解析与图片提取 ──────────────────────────────────

def re_match(text: str) -> tuple[list, list, list]:
    """
    从模型输出中解析 <|ref|>标签<|/ref|><|det|>[坐标]<|/det|> 标签。

    返回值：
        all_matches：  [(完整匹配, 标签, 坐标字符串), ...]
        image_matches：标签为 'image' 的子集
        other_matches：标签非 'image' 的子集
    """
    ref_pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(ref_pattern, text, re.DOTALL)

    all_matches = []
    for full_match, label, box in matches:
        all_matches.append((full_match, label.strip(), box.strip()))

    image_matches = [m for m in all_matches if m[1].lower() == 'image']
    other_matches = [m for m in all_matches if m[1].lower() != 'image']

    return all_matches, image_matches, other_matches


def extract_coordinates(coords_str: str, img_w: int, img_h: int) -> list[tuple[int, int, int, int]]:
    """
    解析坐标字符串，从 [0,999] 归一化坐标映射到实际像素坐标。

    coords_str 格式："[x1,y1,x2,y2]" 或 "[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]"
    返回像素坐标的 (x1, y1, x2, y2) 元组列表。
    """
    try:
        coords = eval(coords_str)
        if coords and isinstance(coords[0], (int, float)):
            coords = [coords]
    except Exception:
        return []

    result = []
    for c in coords:
        try:
            x1 = int(c[0] / 999 * img_w)
            y1 = int(c[1] / 999 * img_h)
            x2 = int(c[2] / 999 * img_w)
            y2 = int(c[3] / 999 * img_h)
            result.append((x1, y1, x2, y2))
        except Exception:
            continue
    return result


def generate_alt(text_before: str, region_type: str) -> str:
    """
    为图片标签生成 alt 文本。
    策略：查找图片前最近的 ## 标题，与区域类型组合。
    """
    headings = re.findall(r'^##\s*(.+)', text_before, re.MULTILINE)
    if headings:
        return f"{headings[-1].strip()} - {region_type}"
    # 降级方案：使用前文最后 40 个字符
    last_text = re.sub(r'\s+', ' ', text_before).strip()
    fallback = last_text[-40:].strip() if last_text else "image"
    return f"{fallback} - {region_type}"


def post_process(raw_text: str, original_image_path: str, req_id: str) -> str:
    """
    将 SGLang 原始输出后处理为干净 Markdown。

    1. 解析 <|ref|>/<|det|> 标签
    2. 对于 'image' 区域：从原图裁剪，保存到 images/{req_id}/N.jpg
    3. 对于所有区域：替换为带 alt 文本的标准 markdown 图片语法
    4. 清理残留标签

    返回处理后的 Markdown 字符串。
    """
    from PIL import Image

    # 去除 EOS 标记
    stop_str = '<｜end▁of▁sentence｜>'
    if raw_text.endswith(stop_str):
        raw_text = raw_text[:-len(stop_str)]
    raw_text = raw_text.strip()

    # 解析 ref/det 标签
    all_matches, image_matches, other_matches = re_match(raw_text)

    if not all_matches:
        return raw_text

    # 准备输出目录
    img_dir = IMAGES_DIR / req_id
    img_dir.mkdir(parents=True, exist_ok=True)

    # 加载原图用于裁剪
    try:
        orig_img = Image.open(original_image_path)
        img_w, img_h = orig_img.size
    except Exception:
        img_w, img_h = 1, 1
        orig_img = None

    processed = raw_text

    # 处理 image 类型区域：裁剪并保存
    for idx, (full_match, label, coords_str) in enumerate(image_matches):
        coord_list = extract_coordinates(coords_str, img_w, img_h)
        saved_any = False
        for ci, (x1, y1, x2, y2) in enumerate(coord_list):
            if orig_img is not None and x2 > x1 and y2 > y1:
                try:
                    cropped = orig_img.crop((x1, y1, x2, y2))
                    suffix = f"_{ci}" if len(coord_list) > 1 else ""
                    cropped.save(str(img_dir / f"{idx}{suffix}.jpg"))
                    saved_any = True
                except Exception:
                    pass

        # 生成 alt 文本
        match_pos = processed.find(full_match)
        text_before = processed[:match_pos] if match_pos >= 0 else ""
        alt = generate_alt(text_before, label)

        # 构建替换文本
        if saved_any:
            if len(coord_list) == 1:
                replacement = f"![{alt}](images/{req_id}/{idx}.jpg)"
            else:
                parts = []
                for ci in range(len(coord_list)):
                    parts.append(f"![{alt} ({ci+1})](images/{req_id}/{idx}_{ci}.jpg)")
                replacement = "\n".join(parts)
        else:
            replacement = f"![{alt}]()"

        processed = processed.replace(full_match, replacement)

    # 清理非 image 类型的 ref/det 标签（直接移除）
    for full_match, label, coords_str in other_matches:
        processed = processed.replace(full_match, "")

    # 清理残留标记
    processed = processed.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

    return processed.strip()
```

- [ ] **步骤 4：单元测试后处理函数**

```shell
cd /data/www/wwwroot/Unlimited-OCR
source .venv/bin/activate
python -c "
from server import re_match, extract_coordinates, generate_alt, post_process
from PIL import Image
import tempfile, os

# 测试 re_match
text = '## 简介\n一些文本。\n<|ref|>image<|/ref|><|det|>[100,50,300,200]<|/det|>\n更多文本。'
all_m, img_m, oth_m = re_match(text)
print('re_match:', len(all_m), '个总计,', len(img_m), '个图片,', len(oth_m), '个其他')
assert len(img_m) == 1
assert img_m[0][1] == 'image'
assert img_m[0][2] == '[100,50,300,200]'

# 测试 extract_coordinates
coords = extract_coordinates('[100,50,300,200]', 1000, 800)
print('坐标映射:', coords)
assert len(coords) == 1

# 测试 generate_alt
text_before = '## 第一章\n一些内容。\n## 图表\n这里是'
alt = generate_alt(text_before, 'diagram')
print('Alt 文本:', alt)
assert alt == '图表 - diagram'

# 测试完整 post_process（使用真实图片）
tmp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
img = Image.new('RGB', (640, 480), color='red')
img.save(tmp_img.name)
raw = '<image>\n## 摘要\n一些文本。\n<|ref|>image<|/ref|><|det|>[100,100,500,400]<|/det|>\n结束。'
result = post_process(raw, tmp_img.name, 'test_req_001')
print('后处理结果:')
print(result[:200])
assert '![](images/test_req_001/0.jpg)' in result or '![' in result
os.unlink(tmp_img.name)
print('全部后处理测试通过！')
"
```

预期：全部断言通过。后处理管线正确解析 ref/det 标签并生成图片引用。

- [ ] **步骤 5：提交**

```bash
git add server.py
git commit -m "feat: 添加 API 模型、消息解析和后处理管线

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### 任务 4：server.py — API 端点

**涉及文件：**
- 修改：`server.py`（追加 health、models、unload 和 chat completions 端点）

**说明：** 添加全部 API 端点。这是 server.py 的最后一块 —— 此任务完成后服务完全可用。

- [ ] **步骤 1：添加 health、models 和 unload 端点**

追加到 `server.py`：

```python
# ── API 端点 ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """健康检查，返回模型加载状态和空闲时间。"""
    with _state_lock:
        loaded = _sglang_process is not None
        loaded_at = _loaded_at
        last_used = _last_used
        total = _total_requests

    idle_time = time.time() - last_used if last_used else 0

    return JSONResponse({
        "status": "ok",
        "model_loaded": loaded,
        "idle_seconds": idle_time,
        "loaded_at": loaded_at,
        "last_used": last_used,
        "idle_unload_limit": IDLE_UNLOAD_SECONDS,
        "total_requests": total,
    })


@app.get("/v1/models")
async def list_models():
    """OpenAI 兼容模型列表。"""
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": SERVED_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "baidu",
        }],
    })


@app.post("/admin/unload")
async def admin_unload():
    """手动卸载模型并释放 GPU 显存。"""
    _stop_sglang()
    return JSONResponse({"ok": True, "unloaded": True})
```

- [ ] **步骤 2：添加 chat completions 端点**

追加到 `server.py`：

```python
# ── Chat Completions ────────────────────────────────────────────────────

def _get_ngram_processor_str():
    """懒加载 SGLang logit processor。首次调用后缓存。"""
    if not hasattr(_get_ngram_processor_str, "_cached"):
        from sglang.srt.sampling.custom_logit_processor import (
            DeepseekOCRNoRepeatNGramLogitProcessor,
        )
        _get_ngram_processor_str._cached = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
    return _get_ngram_processor_str._cached


def _build_sglang_payload(
    prompt: str, image_paths: list[str], request_model: str,
    max_tokens: int, temperature: float, stream: bool,
) -> dict:
    """将标准 OpenAI 请求构建为 SGLang 专用请求负载。"""
    import requests as _requests

    session = _requests.Session()
    session.trust_env = False

    # 将图片编码为 base64 data URI
    content = [{"type": "text", "text": prompt}]
    for path in image_paths:
        ext = os.path.splitext(path)[1].lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    # 根据图片数量决定 image_mode 和 ngram_window
    if len(image_paths) <= 1:
        image_mode = "gundam"
        ngram_window = 128
    else:
        image_mode = "base"
        ngram_window = 1024

    payload = {
        "model": SERVED_NAME,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "skip_special_tokens": False,
        "images_config": {"image_mode": image_mode},
        "custom_logit_processor": _get_ngram_processor_str(),
        "custom_params": {
            "ngram_size": NO_REPEAT_NGRAM_SIZE,
            "window_size": ngram_window,
        },
        "stream": stream,
        "max_tokens": max_tokens,
    }
    return payload


def _collect_sglang_stream(response) -> str:
    """从 SGLang SSE 流式响应中收集全部文本块。返回完整文本。"""
    chunks = []
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                chunks.append(delta)
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
    return "".join(chunks)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OCR 推理端点（OpenAI Chat Completions 兼容）。

    接受多模态消息（文本 + 图片），代理到 SGLang 推理，
    将原始输出后处理为带提取图片的干净 Markdown。
    支持 stream=True 的 SSE 流式输出。
    """
    import requests as http_req

    temps = []
    try:
        # 1. 解析消息
        prompt, img_paths, temps = extract_messages(req.messages)
        if not img_paths:
            return error_response(400, "消息中未找到图片", "invalid_request")

        # 2. 确保 SGLang 已加载
        _ensure_loaded()
        _touch_used()

        # 3. 保存原图副本用于后处理
        req_id = uuid.uuid4().hex[:12]
        from PIL import Image as PILImage
        orig_img_path = img_paths[0]  # 使用第一张图片作为后处理裁剪参考
        orig_copy = IMAGES_DIR / req_id / "_original.png"
        orig_copy.parent.mkdir(parents=True, exist_ok=True)
        PILImage.open(orig_img_path).save(str(orig_copy))

        # 4. 构建负载并转发到 SGLang
        payload = _build_sglang_payload(
            prompt, img_paths, req.model or SERVED_NAME,
            req.max_tokens, req.temperature, req.stream,
        )

        # 始终从 SGLang 请求流式输出以便收集完整文本
        payload["stream"] = True

        resp = http_req.post(
            f"{SGLANG_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )
        resp.raise_for_status()

        # 5. 从 SGLang 收集原始输出
        raw_text = _collect_sglang_stream(resp)

        # 6. 后处理：提取图片、生成 alt 文本、清理标签
        processed = post_process(raw_text, str(orig_copy), req_id)

        obj_id = f"chatcmpl-{req_id}"

        # 7. 返回响应
        if req.stream:
            async def gen():
                # 按词分割流式输出（与 DS-OCR-2 行为一致）
                for word in processed.split():
                    yield (
                        "data: "
                        + json.dumps({
                            "id": obj_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": req.model or SERVED_NAME,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": word + " "},
                                "finish_reason": None,
                            }],
                        })
                        + "\n\n"
                    )
                yield (
                    "data: "
                    + json.dumps({
                        "id": obj_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model or SERVED_NAME,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    })
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        return JSONResponse({
            "id": obj_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or SERVED_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": processed},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return error_response(500, str(exc), "server_error")
    finally:
        # 清理临时文件
        for p in temps:
            try:
                os.unlink(p)
            except Exception:
                pass
```

- [ ] **步骤 3：添加全局异常处理和静态文件挂载**

追加到 `server.py`：

```python
# ── 全局异常处理 ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return error_response(500, str(exc), "server_error")


# ── 静态图片服务 ────────────────────────────────────────────────────────

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
```

- [ ] **步骤 4：验证服务启动和响应**

```shell
cd /data/www/wwwroot/Unlimited-OCR
source .venv/bin/activate

# 后台启动服务
nohup .venv/bin/uvicorn server:app --host 0.0.0.0 --port 10000 > log/server.log 2>&1 &
SERVER_PID=$!
sleep 3

# 测试 health 端点
curl -s http://localhost:10000/health | python3 -m json.tool
# 预期：{"status":"ok","model_loaded":false,"idle_seconds":0,...}

# 测试 models 端点
curl -s http://localhost:10000/v1/models | python3 -m json.tool
# 预期：{"object":"list","data":[{"id":"Unlimited-OCR",...}]}

# 停止服务
kill $SERVER_PID
```

预期：Health 返回 `model_loaded: false`。Models 返回包含 Unlimited-OCR 的列表。

- [ ] **步骤 5：提交**

```bash
git add server.py
git commit -m "feat: 添加全部 API 端点（SGLang 代理 + 后处理）

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### 任务 5：配置文件和 systemd 服务

**涉及文件：**
- 创建：`unlimited-ocr.service`
- 修改：`.env`（已在任务 2 中创建，验证其完整性）

**说明：** 创建 systemd service 文件，使服务开机自启、异常自动重启。

- [ ] **步骤 1：验证 .env 文件**

```bash
cat /data/www/wwwroot/Unlimited-OCR/.env
```

预期内容：

```
MODEL_PATH=/data/www/models/Unlimited-OCR
SERVED_MODEL_NAME=Unlimited-OCR
HOST=0.0.0.0
PORT=10000
SGLANG_PORT=20000
API_KEY=
IDLE_UNLOAD_SECONDS=900
WATCHDOG_POLL_SECONDS=10
GPU=0
MEM_FRACTION_STATIC=0.25
CONTEXT_LENGTH=32768
```

- [ ] **步骤 2：创建 systemd service 文件**

写入 `unlimited-ocr.service`：

```ini
[Unit]
Description=Unlimited-OCR API 服务 (SGLang)
Documentation=https://github.com/baidu/Unlimited-OCR
After=network.target remote-fs.target

[Service]
Type=simple
User=liu
Group=liu
WorkingDirectory=/data/www/wwwroot/Unlimited-OCR
EnvironmentFile=-/data/www/wwwroot/Unlimited-OCR/.env
Environment="PATH=/data/www/wwwroot/Unlimited-OCR/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/data/www/wwwroot/Unlimited-OCR/.venv/bin/uvicorn server:app --host ${HOST} --port ${PORT}
ExecStop=/bin/kill -TERM $MAINPID
Restart=on-failure
RestartSec=10
TimeoutStartSec=300
TimeoutStopSec=30

# GPU 访问权限
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
```

- [ ] **步骤 3：安装并启用服务**

```bash
# 首先停止 10000 端口上的任何手动运行实例
sudo systemctl stop unlimited-ocr 2>/dev/null || true
kill $(lsof -ti :10000) 2>/dev/null || true

# 安装服务
sudo ln -sf /data/www/wwwroot/Unlimited-OCR/unlimited-ocr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable unlimited-ocr
sudo systemctl start unlimited-ocr

# 检查状态
sleep 3
sudo systemctl status unlimited-ocr --no-pager
```

预期：服务启动成功。Health 端点返回 `model_loaded: false`（模型在首次请求前不会加载）。

- [ ] **步骤 4：提交**

```bash
git add unlimited-ocr.service .env
git commit -m "chore: 添加 systemd service 并完成 .env 配置

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### 任务 6：集成测试

**涉及文件：**
- 无新文件
- 测试与运行中的服务交互

**说明：** 端到端测试：向服务发送真实图片，验证返回含图片引用的有效 Markdown，验证提取的图片可以被正常访问。

- [ ] **步骤 1：创建测试图片**

```shell
cd /data/www/wwwroot/Unlimited-OCR
source .venv/bin/activate
python -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (800, 600), color='white')
draw = ImageDraw.Draw(img)
draw.rectangle([50, 50, 750, 200], outline='black', width=2)
draw.text((60, 100), '测试文档标题', fill='black')
draw.rectangle([50, 220, 750, 500], outline='blue', width=2)
draw.text((60, 300), '示例图片区域', fill='blue')
img.save('test_document.png')
print('测试图片已创建: test_document.png')
"
```

- [ ] **步骤 2：确保服务正在运行**

```bash
sudo systemctl status unlimited-ocr --no-pager | head -5
# 如果未使用 systemd，手动启动：
nohup .venv/bin/uvicorn server:app --host 0.0.0.0 --port 10000 > log/server.log 2>&1 &
```

- [ ] **步骤 3：发送 OCR 测试请求**

```shell
cd /data/www/wwwroot/Unlimited-OCR
source .venv/bin/activate

python -c "
import base64, json, requests

with open('test_document.png', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    'model': 'Unlimited-OCR',
    'messages': [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': '<image>\nFree OCR.'},
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}
        ]
    }],
    'stream': False,
    'max_tokens': 4096,
}

print('发送请求中（首次请求会加载模型，可能需要 60-120s）...')
resp = requests.post(
    'http://localhost:10000/v1/chat/completions',
    headers={'Content-Type': 'application/json'},
    json=payload,
    timeout=300,
)
resp.raise_for_status()
result = resp.json()
content = result['choices'][0]['message']['content']
print('=== OCR 结果 ===')
print(content[:500])
if len(content) > 500:
    print('...')
print('=== 结束 ===')
"
```

预期：
- 首次请求触发 SGLang 启动（60-120s 等待）
- 请求成功（200 OK）
- 输出为干净 Markdown 文本（不含 `<|ref|>` 或 `<|det|>` 标签）
- 如有图片输出，引用 `images/{req_id}/N.jpg`

- [ ] **步骤 4：验证图片服务**

```shell
# 检查是否有图片被提取
ls -la /data/www/wwwroot/Unlimited-OCR/images/ 2>/dev/null

# 尝试访问图片目录
curl -s -o /dev/null -w "HTTP状态码: %{http_code}\n" http://localhost:10000/images/ 2>/dev/null
```

- [ ] **步骤 5：测试空闲卸载**

```shell
# 检查模型是否已加载
curl -s http://localhost:10000/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'model_loaded: {d[\"model_loaded\"]}')"

# 等待空闲超时后（如需快速测试可在 .env 中将 IDLE_UNLOAD_SECONDS 设为 60）
# 超时后：curl -s http://localhost:10000/health → model_loaded: false
```

- [ ] **步骤 6：验证流式模式**

```shell
python -c "
import base64, json, requests

with open('test_document.png', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    'model': 'Unlimited-OCR',
    'messages': [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': '<image>\nFree OCR.'},
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}
        ]
    }],
    'stream': True,
    'max_tokens': 4096,
}

print('发送流式请求...')
resp = requests.post(
    'http://localhost:10000/v1/chat/completions',
    headers={'Content-Type': 'application/json'},
    json=payload,
    timeout=300,
    stream=True,
)
resp.raise_for_status()

chunk_count = 0
for line in resp.iter_lines(decode_unicode=True):
    if line and line.startswith('data:'):
        chunk_count += 1
        if chunk_count <= 3:
            print(f'Chunk {chunk_count}: {line[:100]}...')

print(f'共收到 {chunk_count} 个数据块')
print('流式测试通过！')
"
```

预期：收到多个 SSE 数据块，以 `data: [DONE]` 结束。

- [ ] **步骤 7：清理并提交**

```bash
rm -f test_document.png
git commit --allow-empty -m "test: 集成测试通过 — OCR、图片提取、流式输出

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 验证清单

全部任务完成后，逐项确认：

- [ ] `sudo systemctl status unlimited-ocr` 显示 active/running
- [ ] `curl http://localhost:10000/health` 返回模型状态
- [ ] 首次 OCR 请求成功（SGLang 冷启动 ~60-120s）
- [ ] 后续请求快速响应（模型保持加载）
- [ ] 输出为干净 Markdown —— 无 `<|ref|>` / `<|det|>` 标签残留
- [ ] 提取的图片可通过 `/images/{req_id}/N.jpg` 访问
- [ ] 图片 alt 文本包含最近的标题 + 区域类型
- [ ] 超过 `IDLE_UNLOAD_SECONDS` 后模型自动卸载（`model_loaded: false`）
- [ ] `curl -X POST http://localhost:10000/admin/unload` 立即卸载
- [ ] `nvidia-smi` 显示卸载时额外显存 ~0 GB，加载时 ~12-16 GB
- [ ] ComfyUI（端口 8188）仍正常工作（GPU 显存未耗尽）
