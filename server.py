"""
Unlimited-OCR：OpenAI 兼容 OCR 推理服务，基于 HuggingFace Transformers。

启动方式：
    uvicorn server:app --host 0.0.0.0 --port 10000

配置通过 .env 文件或环境变量设置，详见 README。
模型按需加载，空闲 IDLE_UNLOAD_SECONDS 秒后自动卸载释放显存。

接口:
    GET  /health              — 健康检查（含模型加载状态和空闲时间）
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
import traceback
import shutil
import re
from pathlib import Path
from typing import Optional

# 加载 .env 文件（优先级低于系统环境变量）
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=False)

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

# ═══════════════════════════════════════════════════════════════════════════
# 全局配置（环境变量 > .env > 默认值）
# ═══════════════════════════════════════════════════════════════════════════

# 模型本地路径
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/www/models/Unlimited-OCR")

# API 响应中返回的模型名称
SERVED_NAME = os.environ.get("SERVED_MODEL_NAME", "Unlimited-OCR")

# 服务监听地址和端口
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "10000"))

# API 认证密钥。为空则不校验；非空则请求必须携带匹配的 Authorization 头
API_KEY = os.environ.get("API_KEY", "")

# 空闲自动卸载间隔（秒），默认 900 秒（15 分钟）
IDLE_UNLOAD_SECONDS = int(os.environ.get("IDLE_UNLOAD_SECONDS", "900"))

# 看门狗检查间隔（秒）
WATCHDOG_POLL_SECONDS = int(os.environ.get("WATCHDOG_POLL_SECONDS", "10"))

# OCR 过程中提取的图片存放目录，按请求 ID 分子目录
IMAGES_DIR = Path(__file__).parent / "images"

# ── 懒加载全局状态 ─────────────────────────────────────────────────────────
_state_lock = threading.Lock()   # 保护 _model / _tokenizer / 状态字段
_infer_lock = threading.Lock()   # 推理互斥，保证单 GPU 串行

_model: Optional[object] = None
_tokenizer: Optional[object] = None
_loaded_at: Optional[float] = None
_last_used: Optional[float] = None
_total_requests: int = 0

app = FastAPI(title="Unlimited-OCR API", version="1.0.0")

# 将 OCR 提取的图片目录挂载为静态资源，可通过 /images/{req_id}/N.jpg 访问
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


# ═══════════════════════════════════════════════════════════════════════════
# API 认证中间件 (OpenAI 兼容: Authorization: Bearer <key>)
# ═══════════════════════════════════════════════════════════════════════════

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """
    认证中间件。
    - 未配置 API_KEY 时，放行所有请求
    - 配置后，只放行 /health、/images/ 和携带正确 Bearer token 的请求
    """
    # 白名单路径：无需认证
    if request.url.path in ("/health",) or request.url.path.startswith("/images/"):
        return await call_next(request)

    # 未配置密钥则放行
    if not API_KEY:
        return await call_next(request)

    # 校验 Authorization 头
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


# ═══════════════════════════════════════════════════════════════════════════
# 懒加载模型管理
# ═══════════════════════════════════════════════════════════════════════════

def _load_model():
    """
    从 MODEL_PATH 加载分词器和模型到 GPU。

    使用 bfloat16 精度 (~6.4 GB)，trust_remote_code=True 加载自定义模型代码。
    unlimited-ocr 架构会自动加载视觉编码器（SAM + CLIP）和投影器。
    """
    global _model, _tokenizer, _loaded_at, _last_used

    print("正在加载 Unlimited-OCR 模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()

    with _state_lock:
        _model = model
        _tokenizer = tokenizer
        _loaded_at = time.time()
        _last_used = time.time()
    print("模型加载完成。")


def _unload_model():
    """卸载模型并释放 GPU 显存。"""
    global _model, _tokenizer, _loaded_at

    m = None
    t = None
    with _state_lock:
        if _model is None:
            return
        print("空闲超时，正在卸载模型...")
        m = _model
        t = _tokenizer
        _model = None
        _tokenizer = None
        _loaded_at = None

    del m, t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    print("模型已卸载。")


def _ensure_loaded():
    """
    按需加载模型（双重检查锁，避免并发重复加载）。

    首次请求时自动加载模型到 GPU，后续请求复用已加载的实例。
    """
    with _state_lock:
        if _model is not None:
            return
    with _infer_lock:
        with _state_lock:
            if _model is not None:
                return
        _load_model()


def _touch_used():
    """更新最后使用时间戳。"""
    global _last_used, _total_requests
    with _state_lock:
        _last_used = time.time()
        _total_requests += 1


def _watchdog():
    """后台线程：定期检查空闲时间，超时则卸载模型。"""
    while True:
        time.sleep(WATCHDOG_POLL_SECONDS)
        with _state_lock:
            loaded = _model is not None
            last = _last_used
        if loaded and last and (time.time() - last) >= IDLE_UNLOAD_SECONDS:
            _unload_model()


# ═══════════════════════════════════════════════════════════════════════════
# 启动时启动看门狗线程，模型将在首次请求时按需加载
# ═══════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup():
    """
    FastAPI 启动事件。

    启动看门狗后台线程。模型不会在启动时加载 ——
    首次请求到达时由 _ensure_loaded() 按需加载。
    """
    th = threading.Thread(target=_watchdog, daemon=True)
    th.start()


# ═══════════════════════════════════════════════════════════════════════════
# OpenAI 兼容的错误响应格式
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic 请求模型 — 对齐 OpenAI API 规格
# ═══════════════════════════════════════════════════════════════════════════

class Message(BaseModel):
    """对话中的一条消息。content 可以是纯文本字符串或多模态内容块列表。"""
    role: str = "user"
    content: str | list[dict] = ""
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """
    POST /v1/chat/completions 请求体。

    与 OpenAI Chat Completions API 对齐。核心字段是 messages，
    需包含至少一个 image_url 类型的多模态内容块。
    """
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


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数：消息解析、图片处理、输出后处理
# ═══════════════════════════════════════════════════════════════════════════

# 模型提取的图片文件扩展名
IMAGE_EXT = {"jpg", "jpeg", "png", "webp", "bmp"}

# 模型输出的文本文件扩展名
TEXT_EXT = {".mmd", ".md", ".txt", ".json", ".xml", ".html"}


def decode_image(s: str) -> str:
    """
    将 base64 data-URI 解码为临时图片文件。

    参数:
        s: data:image/png;base64,... 格式的字符串

    返回:
        临时文件路径。调用方负责在使用后清理。
    """
    # 去掉 data URI 前缀
    if s.startswith("data:"):
        s = s.split(",", 1)[1]

    data = base64.b64decode(s)

    # 通过魔数判断 JPEG，其余默认 PNG
    suffix = ".jpg" if data[:4] == b"\xff\xd8\xff" else ".png"

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def extract_messages(messages: list[Message]) -> tuple[str, str | None, list[str]]:
    """
    解析 OpenAI 格式的消息，提取 prompt 文本和图片路径。

    支持的 content 格式:
        - 纯文本:     {"content": "some text"}
        - 多模态块:   {"content": [{"type":"text",...}, {"type":"image_url",...}]}

    支持的图片来源:
        - data:image/...;base64,...   base64 编码的 data URI
        - http://... / https://...   远程下载
        - /path/to/file.jpg          本地文件路径

    返回:
        (prompt文本, 第一张图片路径, 所有临时文件路径列表)
    """
    parts, img, temps = [], None, []

    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            # 纯文本消息
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
                    elif url.startswith(("http://", "https://")):
                        import urllib.request
                        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        urllib.request.urlretrieve(url, tmp.name)
                        tmp.close()
                        fpath = tmp.name
                    elif os.path.isfile(url):
                        fpath = url
                    if fpath:
                        temps.append(fpath)
                        # 使用第一张图片作为推理输入，其余在多图模式下使用
                        if img is None:
                            img = fpath

    # 组装 prompt，确保包含 <image> 标记供视觉编码器使用
    prompt = "\n".join(parts) if parts else "<image>\nFree OCR."
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    return prompt, img, temps


def save_output_and_images(out_dir: str, req_id: str) -> str:
    """
    处理模型输出:

    1. 将提取的图片从临时目录复制到 IMAGES_DIR/{req_id}/
    2. 读取文本输出文件（result.md 等）
    3. 将嵌入的图片路径从 images/N.jpg 改写为 images/{req_id}/N.jpg
    4. 为空 alt 的图片自动补充描述（用前面最近的 ## 标题）

    返回合并后的 Markdown 文本。
    """
    # 持久化提取的图片
    dest = IMAGES_DIR / req_id
    dest.mkdir(parents=True, exist_ok=True)

    for f in Path(out_dir).glob("images/*"):
        if f.is_file():
            shutil.copy2(f, dest / f.name)

    # 读取文本输出，替换 Markdown 中的图片路径
    results = []
    for f in sorted(Path(out_dir).glob("**/*")):
        if f.is_file() and f.suffix.lower() in TEXT_EXT:
            text = f.read_text()

            # 为空 alt 的图片补充描述：用前面最近的 ## 标题作为 alt
            def fill_alt(m):
                prefix = m.group(1)
                img_tag = m.group(2)
                # 找前面最近的 ## 标题
                headings = re.findall(r"^##\s*(.+)", prefix, re.MULTILINE)
                if headings:
                    alt = headings[-1].strip()
                else:
                    last_text = re.sub(r"\s+", " ", prefix).strip()
                    alt = last_text[-40:].strip() if last_text else "image"
                # 保留前缀内容，只替换图片标签
                return f"{prefix}![{alt}]({img_tag})"

            text = re.sub(
                r"(?s)(.*?)!\[\]\((images/\d+\.(?:jpg|png|jpeg|webp|bmp))\)",
                fill_alt,
                text,
            )

            # 将临时路径替换为持久化路径
            text = re.sub(
                r"images/(\d+\.(jpg|png|jpeg|webp|bmp))",
                f"images/{req_id}/\\1",
                text,
            )
            results.append(text)
    return "\n\n".join(results) if results else ""


def make_chunk(obj_id: str, model: str, delta: dict, finish: str | None = None) -> dict:
    """构造流式响应中的一个 SSE 数据块。"""
    return {
        "id": obj_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def make_response(obj_id: str, model: str, content: str) -> dict:
    """构造非流式的完整 Chat Completion 响应。"""
    return {
        "id": obj_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# API 端点
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """健康检查。返回模型加载状态和空闲时间。"""
    with _state_lock:
        loaded = _model is not None
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
    """
    模型列表（OpenAI 兼容）。

    返回单条记录，即当前服务的模型名称。
    """
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": SERVED_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "baidu",
            }
        ],
    })


@app.post("/admin/unload")
async def admin_unload():
    """手动卸载模型以释放 GPU 显存。"""
    _unload_model()
    return JSONResponse({"ok": True, "unloaded": True})


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OCR 推理主接口（OpenAI Chat Completions 兼容）。

    接受多模态 messages（文本 + 图片），调用 model.infer() 同步推理。
    支持 stream=True 以 SSE 方式流式返回结果。

    单 GPU 串行执行，同一时间只处理一个请求。
    Unlimited-OCR 使用 gundam 模式（base_size=1024, image_size=640, crop_mode=True）
    进行推理，内置 ref/det 标签解析和图片提取。
    """
    temps = []
    try:
        # 解析请求消息，提取 prompt 和图片
        prompt, img, temps = extract_messages(req.messages)
        if not img:
            return error_response(400, "消息中未找到图片", "invalid_request")

        # 按需加载模型并更新使用时间
        _ensure_loaded()
        _touch_used()

        req_id = uuid.uuid4().hex[:12]

        with _infer_lock:
            if _model is None:
                return error_response(503, "模型未加载", "server_error")

            # 在临时目录中执行推理
            # 使用 gundam 模式：base_size=1024（全局视图），image_size=640（切片分辨率）
            # crop_mode=True 启用动态分块，适合复杂排版的文档
            with tempfile.TemporaryDirectory() as out_dir:
                _model.infer(
                    _tokenizer,
                    prompt=prompt,
                    image_file=img,
                    output_path=out_dir,
                    base_size=1024,         # 全局视图分辨率
                    image_size=640,         # 切片裁剪分辨率（gundam 模式）
                    crop_mode=True,         # 动态分块，适应复杂排版
                    max_length=32768,       # 最大输出 token 数
                    no_repeat_ngram_size=35,# 防止 ngram 重复的窗口大小
                    ngram_window=128,       # 滑动窗口大小
                    save_results=True,      # 将文本和图片保存到 out_dir
                )
                text = save_output_and_images(out_dir, req_id) or "[empty]"

        obj_id = f"chatcmpl-{req_id}"

        # 流式输出：SSE (Server-Sent Events)
        if req.stream:
            async def gen():
                for chunk in text.split():
                    yield (
                        "data: "
                        + json.dumps(make_chunk(obj_id, req.model or SERVED_NAME, {"content": chunk + " "}))
                        + "\n\n"
                    )
                yield "data: " + json.dumps(make_chunk(obj_id, req.model or SERVED_NAME, {}, "stop")) + "\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        return JSONResponse(make_response(obj_id, req.model or SERVED_NAME, text))

    except Exception as exc:
        traceback.print_exc()
        return error_response(500, str(exc), "server_error")
    finally:
        # 清理临时图片文件
        for p in temps:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# 全局异常兜底
# ═══════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    """捕获所有未处理的异常，返回 OpenAI 兼容错误格式。"""
    traceback.print_exc()
    return error_response(500, str(exc), "server_error")
