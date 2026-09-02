"""
Unlimited-OCR：OpenAI 兼容 OCR 推理服务（vLLM 后端适配层）。

与 server.py 提供完全一致的 OpenAI 兼容 API，但本服务不在本进程加载模型——
所有推理转发给下游常驻的 vLLM 引擎（docker 容器，端口 9706）。

启动方式：
    uvicorn server_vllm:app --host 0.0.0.0 --port 9707

配置通过 .env 文件或环境变量设置。关键差异：
- VLLM_URL 指向下游 vLLM OpenAI 兼容端点（默认 http://127.0.0.1:9706/v1）
- 推理转发时显式传 skip_special_tokens=False，保留 <|det|> 标签供后处理
- /admin/unload 为兼容端点（vLLM 常驻，无实际卸载动作）

接口:（与 server.py 完全一致）
    GET  /health              — 健康检查（含下游引擎状态和空闲时间）
    GET  /v1/models           — 模型列表
    POST /v1/chat/completions — 多模态 OCR 推理
    POST /admin/unload        — 卸载模型（兼容端点）
"""

import os
import sys
import gc
import threading
import shlex
import subprocess
import urllib.request
import base64
import json
import time
import uuid
import tempfile
import traceback
import shutil
import re
import warnings
import logging
from pathlib import Path

# ── 日志 ─────────────────────────────────────────────────────────
# 统一使用 logging 输出到 stdout/stderr（systemd journald 可见）
logger = logging.getLogger("unlimited-ocr")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
from typing import Optional

# 屏蔽 Transformers generate() 内部关于 attention_mask/pad_token_id 的冗余警告
# 模型使用 images_seq_mask 管理视觉 token 的注意力，不需要标准 attention_mask
# 实测不影响输出质量，仅减少日志噪音
warnings.filterwarnings("ignore", message=".*attention_mask.*")
warnings.filterwarnings("ignore", message=".*pad_token_id.*")

# 加载 .env 文件（优先级低于系统环境变量）
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=False)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════
# 全局配置（环境变量 > .env > 默认值）
# ═══════════════════════════════════════════════════════════════════════════

# 后端推理引擎地址（本适配层不直接加载模型，所有推理转发给下游 vLLM 服务）
# vLLM 服务需以 skip_special_tokens=False 语义处理请求（由本层在每次请求中显式带上），
# 才能保留模型输出的 <|det|> 标签，供 process_raw_output() 解析。
# 兼容环境变量 VLLM_URL / ENGINE_URL
VLLM_URL = (
    os.environ.get("VLLM_URL")
    or os.environ.get("ENGINE_URL")
    or "http://127.0.0.1:9706/v1"
).rstrip("/")

# 下游 vLLM 引擎的 docker 容器名（空闲卸载 / 按需启动时管理该容器）
VLLM_CONTAINER = os.environ.get("VLLM_CONTAINER", "vllm-ocr")

# 容器不存在时用于重建的 docker run 参数（空格分隔；由 shlex.split 解析）。
# 注意：模型目录、端口映射、量化等参数必须与首次启动一致。
# 仅当 VLLM_START_ARGS 非空时，适配层才具备“容器不存在则重建”能力；
# 置空则只做 docker start（容器必须预先存在）。
VLLM_START_ARGS = os.environ.get(
    "VLLM_START_ARGS",
    "--gpus device=0 --ipc host -p 9706:8000 "
    "-v /data/www/models/Unlimited-OCR:/models/Unlimited-OCR:ro "
    "vllm/vllm-openai:unlimited-ocr "
    "/models/Unlimited-OCR --served-model-name Unlimited-OCR "
    "--trust-remote-code "
    "--logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor "
    "--no-enable-prefix-caching --mm-processor-cache-gb 0 "
    "--quantization fp8 --gpu-memory-utilization 0.2 "
    "--max-model-len 32768 --host 0.0.0.0 --port 8000",
)

# 启动容器后等待引擎就绪的超时时间（秒）。
# vLLM FP8 冷启动（权重加载 + CUDA graph 捕获）约需 60-120s。
VLLM_START_TIMEOUT = int(os.environ.get("VLLM_START_TIMEOUT", "240"))

# 下游引擎推理时使用的采样参数（对齐原 Transformers infer 的贪心+防重复配置）
VLLM_MAX_TOKENS = int(os.environ.get("VLLM_MAX_TOKENS", "8192"))

# ── NGram 防重复（关键！）────────────────────────────────────────────
# 下游 vLLM 引擎以 --logits_processors 注册了 NGramPerReqLogitsProcessor，
# 但该处理器只有在请求携带 extra_args(vllm_xargs) 时才会真正启用：
#   若 extra_args 缺 ngram_size，new_req_logits_processor() 返回 None，
#   防重复被静默禁用 → 模型自回归出现短句循环（如"圖謀刺殺圖謀刺殺…"）
#   时无任何防护，直接生成到 max_tokens 耗尽。
# vLLM unlimited_ocr 模块 docstring 推荐：ngram_size=35, window_size=128。
# 引擎采样窗口上限 32768，此值经官方模型验证，无需随 max_tokens 调整。
NGRAM_SIZE = int(os.environ.get("NGRAM_SIZE", "35"))
NGRAM_WINDOW = int(os.environ.get("NGRAM_WINDOW", "128"))

# 下游引擎名称（vLLM serve 时指定的 served-model-name）
ENGINE_MODEL_NAME = os.environ.get("ENGINE_MODEL_NAME", "Unlimited-OCR")

# 单次请求向下游引擎最多能接受的总图片数（超过则分多批）。
# vLLM 的 max_model_len=32768；单页 crop 最多 32 块 + 文本 token，
# 实测单页约 4-8k token，32768 上限下一次最多安全处理 4 页（保守取 3）
VLLM_MAX_IMAGES_PER_REQ = int(os.environ.get("VLLM_MAX_IMAGES_PER_REQ", "3"))

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

# PDF/多图分批处理：每批最多处理多少页
# 20页 × 257 token/页 ≈ 5140 image tokens，留足文本输出空间，避免 OOM
MAX_PAGES_PER_BATCH = int(os.environ.get("MAX_PAGES_PER_BATCH", "20"))

# OCR 过程中提取的图片存放目录，按请求 ID 分子目录
IMAGES_DIR = Path(__file__).parent / "images"

# ── 引擎健康状态 ───────────────────────────────────────────────────────────
_state_lock = threading.Lock()   # 保护状态字段
_infer_lock = threading.Lock()   # 推理互斥，保证单 GPU 串行

# 模型不再由本进程加载 —— 推理引擎是常驻的下游 vLLM 服务。
# 这里仅记录下游引擎的可达状态与请求计数，用于 /health 展示。
_engine_ok: bool = False          # 最近一次探测下游引擎是否可达
_engine_error: str = ""          # 探测失败原因
_loaded_at: Optional[float] = None   # 下游引擎首次探测成功时间
_last_used: Optional[float] = None   # 最后一次推理请求时间
_total_requests: int = 0

# 容器生命周期状态（空闲自动卸载 / 按需启动用）
_container_exists: bool | None = None  # None=未知；False=确认不存在（可重建）
_container_running: bool = False       # 适配层认为容器当前应处于运行中
_engine_stop_ts: float | None = None   # 最近一次主动停止容器的时间戳（防止竞态误启动）

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
# 下游 vLLM 引擎探测 / 容器生命周期管理
# ═══════════════════════════════════════════════════════════════════════════


def _docker(*args: str, timeout: int = 60) -> str:
    """
    执行 docker 子命令，返回 stdout（去除末尾换行）。

    参数:
        args: docker 子命令参数列表，如 ("ps", "-q")。
        timeout: 子进程超时秒数。

    异常:
        RuntimeError: docker 命令失败（非零退出码）。
    """
    cmd = ["docker"] + list(args)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"docker 命令超时: {' '.join(cmd)}") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("docker 命令不存在，无法管理下游引擎容器") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker {' '.join(args)} 失败({proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def _container_status() -> str:
    """
    查询下游引擎容器的当前状态（running / exited / missing）。

    返回:
        "running": 容器运行中；"exited": 容器存在但已停止；
        "missing": 容器不存在；"unknown": 查询失败。
    """
    global _container_exists, _container_running
    try:
        out = _docker("inspect", "-f", "{{.State.Status}}", VLLM_CONTAINER)
        st = out.strip().lower()
        with _state_lock:
            _container_exists = True
            _container_running = (st == "running")
        if st == "running":
            return "running"
        if st in ("exited", "created", "dead", "paused", "restarting"):
            return "exited"
        return "unknown"
    except RuntimeError:
        with _state_lock:
            _container_exists = False
            _container_running = False
        return "missing"


def _start_container() -> None:
    """
    启动下游引擎容器（不存在则按 VLLM_START_ARGS 重建）。

    仅执行 docker start / docker run，不等待引擎就绪
    （就绪由 _ensure_engine() 轮询探测）。
    """
    global _container_exists, _container_running, _engine_ok, _engine_error, _loaded_at
    status = _container_status()
    if status == "running":
        return
    if status == "missing" and VLLM_START_ARGS.strip():
        logger.info("容器 %s 不存在，按预设参数重建...", VLLM_CONTAINER)
        args = shlex.split(VLLM_START_ARGS)
        _docker("run", "-d", "--name", VLLM_CONTAINER, *args, timeout=120)
    elif status == "missing":
        raise RuntimeError(
            f"容器 {VLLM_CONTAINER} 不存在且未配置 VLLM_START_ARGS，无法自动启动"
        )
    else:
        logger.info("启动已停止的引擎容器 %s ...", VLLM_CONTAINER)
        _docker("start", VLLM_CONTAINER, timeout=120)
    with _state_lock:
        _container_exists = True
        _container_running = True
        _engine_ok = False        # 启动后引擎需重新探测确认
        _engine_error = "容器已启动，等待引擎就绪..."
        _loaded_at = None


def _stop_container() -> None:
    """
    停止下游引擎容器并释放 GPU 显存。

    等价于原 Transformers 版的 _unload_model()：空闲超时或手动触发后
    将常驻的 vLLM 引擎停止，下次请求时再由 _ensure_engine() 按需拉起。
    """
    global _container_exists, _container_running, _engine_ok, _engine_error
    global _engine_stop_ts
    with _state_lock:
        if not _container_running:
            return
    try:
        logger.info("空闲超时/手动触发，停止引擎容器 %s ...", VLLM_CONTAINER)
        _docker("stop", "-t", "30", VLLM_CONTAINER, timeout=60)
    except RuntimeError as exc:
        # 容器可能已被外部删除或停止，重新探测一次避免状态失真
        logger.warning("停止容器异常: %s", exc)
        _container_status()
        return
    with _state_lock:
        _container_exists = True
        _container_running = False
        _engine_ok = False
        _engine_error = "引擎已停止（空闲自动卸载），等待下次请求重新拉起"
        _engine_stop_ts = time.time()
    logger.info("引擎容器已停止，显存已释放。")


def _probe_engine() -> bool:
    """
    探测下游 vLLM 引擎是否可达（GET /v1/models）。

    返回:
        True 表示引擎可达且模型列表包含 ENGINE_MODEL_NAME；否则 False。
        探测失败时不修改容器状态（仅标记引擎不可达）。
    """
    global _engine_ok, _engine_error, _loaded_at
    try:
        req = urllib.request.Request(f"{VLLM_URL}/models", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        ids = [m.get("id", "") for m in data.get("data", [])]
        if ENGINE_MODEL_NAME not in ids:
            raise RuntimeError(f"引擎模型列表中没有 {ENGINE_MODEL_NAME}: {ids}")
        with _state_lock:
            _engine_ok = True
            _engine_error = ""
            if _loaded_at is None:
                _loaded_at = time.time()
        return True
    except Exception as exc:
        with _state_lock:
            _engine_ok = False
            _engine_error = f"{type(exc).__name__}: {exc}"
        return False


def _wait_engine_ready(timeout: float = VLLM_START_TIMEOUT) -> None:
    """
    轮询等待下游引擎就绪（模型加载完成、/v1/models 可达）。

    参数:
        timeout: 最大等待秒数。

    异常:
        RuntimeError: 超时仍未就绪。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _probe_engine():
            logger.info("下游引擎就绪。")
            return
        time.sleep(5)
    raise RuntimeError(f"等待下游引擎就绪超时({timeout}s): {_engine_error}")


def _ensure_engine_locked():
    """
    确保下游引擎可用（按需启动，对齐原版懒加载语义）。

    ⚠️ 调用方必须已持有 _infer_lock（见 chat_completions），
    以保证“启动/等待就绪/推理”与 watchdog 的停止动作互斥。

    - 引擎已就绪：直接返回（请求无需等待）；
    - 引擎未就绪：探测一次，若容器未运行则 docker start（或重建），
      然后轮询等待引擎就绪后再返回。
    """
    with _state_lock:
        if _engine_ok:
            return
    # 1) 容器可能运行中但引擎仍在加载 → 先探测一轮
    status = _container_status()
    if status == "running" and _probe_engine():
        return
    if status != "running":
        # 2) 容器未运行（或缺失）→ 启动容器
        _start_container()
    # 3) 等待引擎就绪（冷启动：权重加载 + CUDA graph 捕获）
    _wait_engine_ready()


def _ensure_engine():
    """
    确保下游引擎可用（线程安全入口，带 _infer_lock）。

    冷启动可能耗时较长（docker start + 模型加载），期间持有 _infer_lock，
    与看门狗的停止动作互斥，避免“停与用”竞态。
    """
    with _state_lock:
        ok = _engine_ok
    if ok:
        return
    with _infer_lock:
        _ensure_engine_locked()


def _touch_used():
    """更新最后使用时间戳与请求计数。"""
    global _last_used, _total_requests
    with _state_lock:
        _last_used = time.time()
        _total_requests += 1


def _watchdog():
    """
    后台看门狗线程：空闲超时自动停止引擎容器（对齐原版空闲卸载）。

    每 WATCHDOG_POLL_SECONDS 秒检查一次：
    - 引擎在运行（容器 running 且探测可达）；
    - 距最后一次推理请求超过 IDLE_UNLOAD_SECONDS；
    满足则停止容器释放显存。引擎不可达时不重复执行停止。
    停止动作在 _infer_lock 内执行：若此刻恰好有请求正在拉起/使用
    引擎（_ensure_engine 已持锁），则等待其完成，避免“停与用”竞态。
    """
    while True:
        time.sleep(WATCHDOG_POLL_SECONDS)
        try:
            with _state_lock:
                running = _container_running
                ok = _engine_ok
                last = _last_used
                stop_ts = _engine_stop_ts
            if not (running and ok):
                continue
            # 距离停止时间过近（<60s）不重复停止，避免误判竞态
            if stop_ts and (time.time() - stop_ts) < 60:
                continue
            idle = time.time() - last if last else float("inf")
            if idle > IDLE_UNLOAD_SECONDS:
                # 持推理锁后再确认一次空闲状态（期间可能有请求更新 _last_used）
                with _infer_lock:
                    with _state_lock:
                        last2 = _last_used
                    idle2 = time.time() - last2 if last2 else float("inf")
                    if idle2 > IDLE_UNLOAD_SECONDS:
                        _stop_container()
        except Exception:
            traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════
# 启动时启动看门狗线程；引擎按需由请求拉起（对齐原版懒加载）
# ═══════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup():
    """
    FastAPI 启动事件。

    启动看门狗线程。适配层启动时**不主动拉起引擎**（对齐原版：
    模型只在首次请求时加载）；仅记录容器当前状态，首次请求到达时
    由 _ensure_engine() 按需启动容器并等待就绪。
    """
    try:
        _container_status()
    except Exception:
        traceback.print_exc()
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

    Unlimited-OCR 扩展字段:
        max_pages:   PDF/多图时最多处理多少页。None=全部。例如 max_pages=10 只处理前10页。
        page_mode:   多图处理模式。
                     "batch" (默认) — infer_multi 一次推理，跨页上下文连贯
                     "single"        — 逐张 infer，640 gundam 高质量，速度快
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
    # ── Unlimited-OCR 扩展字段 ──
    # max_pages: PDF/多图时最多处理多少页。None=全部页。
    #   例如 max_pages=5 只取前5页，适合预览或限制 token 消耗
    max_pages: int | None = None

    # page_mode: 多图处理策略（单图时忽略，始终用 gundam 640）
    #   "batch"  — 默认，infer_multi 批量推理，跨页上下文连贯但速度略慢
    #   "single" — 逐张 infer gundam 640，每页独立推理，速度更快且质量更高
    page_mode: str = "batch"

    class Config:
        populate_by_name = True


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数：消息解析、图片处理、输出后处理
# ═══════════════════════════════════════════════════════════════════════════

# 模型输出的 det 标签类型到 Markdown 标题层级的映射
# 键: det 类型名, 值: (Markdown 前缀, 是否保留标签后的文本)
DET_TYPE_MAP = {
    "header":       ("# ", True),      # 一级标题
    "title":        ("## ", True),     # 二级标题
    "subtitle":     ("### ", True),    # 三级标题
    "text":         ("", True),        # 正文段落
    "image":        ("", False),       # 图片（只裁剪，不输出文本）
    "page_number":  ("", False),       # 页码（跳过）
    "footer":       ("", False),       # 页脚（跳过）
}

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


def extract_messages(messages: list[Message]) -> tuple[str, list[str], list[str]]:
    """
    解析 OpenAI 格式的消息，提取 prompt 文本和所有图片/PDF 路径。

    支持的图片来源:
        - data:image/...;base64,...   图片 base64
        - data:application/pdf;base64,...  PDF base64
        - http://host/doc.pdf         远程 PDF（自动下载）
        - http://host/img.jpg         远程图片（自动下载）
        - /absolute/path/file.jpg     本地图片
        - /absolute/path/file.pdf     本地 PDF（自动转图片）

    返回:
        (prompt文本, 所有文件路径列表, 临时文件路径列表)
    """
    parts, imgs, temps = [], [], []

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
                        # base64 编码: data:image/png;base64,... 或 data:application/pdf;base64,...
                        fpath = decode_image(url)
                    elif url.startswith(("http://", "https://")):
                        # 远程 URL: 根据扩展名决定后缀（PDF 会被后续检测到并转图片）
                        import urllib.request
                        url_lower = url.split("?")[0]  # 去掉查询参数
                        suffix = ".pdf" if url_lower.endswith(".pdf") else ".png"
                        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                        urllib.request.urlretrieve(url, tmp.name)
                        tmp.close()
                        fpath = tmp.name
                    elif os.path.isfile(url):
                        # 本地绝对路径：直接使用调用方的文件
                        # 不加入 temps —— 服务端只清理自己产生的临时文件（base64 解码、
                        # URL 下载），无权删除客户端/调用方的本地文件
                        imgs.append(url)
                        continue
                    if fpath:
                        temps.append(fpath)
                        imgs.append(fpath)

    # 组装 prompt
    # 多图时使用 "Multi page parsing." 提示词，单图时使用 "document parsing."
    # 注意：必须使用官方指令（README 示例），勿用 DeepSeek-OCR 的 "Free OCR." ——
    # 模型对 "Free OCR." 指令困惑，会在输出开头重复大量 "Free"（直至 no_repeat_ngram 打断），
    # 甚至偶发陷入生成循环（生成满 max_length 不停止），导致识别失败、耗时极长。
    if len(imgs) > 1:
        default_prompt = "<image>Multi page parsing."
    else:
        default_prompt = "<image>document parsing."

    prompt = "\n".join(parts) if parts else default_prompt
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    return prompt, imgs, temps


# ═══════════════════════════════════════════════════════════════════════════
# 下游 vLLM 引擎 HTTP 推理转发
# ═══════════════════════════════════════════════════════════════════════════


def image_to_data_uri(img_path: str) -> str:
    """
    将本地图片文件转为 data URI 字符串，供 OpenAI 兼容接口的 image_url 使用。

    参数:
        img_path: 本地图片文件绝对路径。

    返回:
        data:image/...;base64,... 格式字符串。
    """
    import mimetypes
    mime, _ = mimetypes.guess_type(img_path)
    if mime not in ("image/png", "image/jpeg", "image/webp", "image/bmp", "image/gif"):
        # 未知格式回退到 PNG（魔数判断，与 decode_image 逻辑一致）
        with open(img_path, "rb") as f:
            head = f.read(4)
        mime = "image/jpeg" if head[:2] == b"\xff\xd8" else "image/png"
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def vllm_chat(
    prompt: str,
    image_paths: list[str],
    max_tokens: int = VLLM_MAX_TOKENS,
    temperature: float = 0.0,
) -> str:
    """
    转发一次 OCR 推理请求到下游 vLLM 引擎，返回模型原始文本输出。

    与 Transformers 版的 _model.infer() / infer_multi() 等价。
    关键点：必须显式传 skip_special_tokens=False，否则 vLLM 会剥掉模型
    输出的 <|det|> / <|/det|> 标签，process_raw_output() 将无法解析。

    参数:
        prompt:       最终 prompt（含 <image> 占位符文本）。
        image_paths:  本地图片路径列表（单图传 [path]，多图传全部）。
        max_tokens:   生成 token 上限。
        temperature:  采样温度。

    返回:
        模型原始输出文本（含 <|det|> 标签与 <PAGE> 分页符，引擎常驻
        无 EOS 包装差异，与 Transformers eval_mode=True 输出一致）。

    异常:
        RuntimeError: 引擎不可达或返回非 200。
    """
    content: list[dict] = [{"type": "text", "text": prompt}]
    for p in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_uri(p)},
        })

    payload = {
        "model": ENGINE_MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # 关键：保留 <|det|> 标签供 process_raw_output 解析
        "skip_special_tokens": False,
        # 关键：启用引擎侧 NGram 防重复（无此字段处理器被静默禁用，
        # 模型短句循环时无防护 → 会一直重复生成到 max_tokens 耗尽）
        "vllm_xargs": {
            "ngram_size": NGRAM_SIZE,
            "window_size": NGRAM_WINDOW,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{VLLM_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # 读取错误响应体，便于排查
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        raise RuntimeError(f"vLLM 引擎返回 {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"vLLM 引擎不可达: {e.reason}") from e

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"vLLM 引擎响应缺少 choices: {json.dumps(data)[:300]}")
    text = choices[0]["message"].get("content", "")
    return text


# ═══════════════════════════════════════════════════════════════
# 幻觉护栏（后处理防循环/无中生有）
# ═══════════════════════════════════════════════════════════════
# 模型对读不出的区域（英文报纸密排小字、图形栏等）会陷入两类幻觉：
#   1. 循环重复：同一短句反复输出（"圖謀刺殺圖謀刺殺…"、"The first, The first…"），
#      或在同坐标带反复输出同一行，直到 max_tokens 耗尽；
#   2. 声明式幻觉：把"该栏无文字"误当正文输出（
#      "The image contains no text. The horizontal lines are…"），
#      或塞入训练语料高频垃圾句（"quick brown fox"、"2017年…交易金额…"）。
# 此类行对用户无意义且污染 Markdown，统一在 process_raw_output 前过滤。
# 注意：护栏只删"确定幻觉"的行，对正常文本零影响（NGram 防重复仍由引擎负责）。

# 已知幻觉/垃圾文本标记（子串匹配，出现即整行丢弃）。
# 覆盖：英文练习句、无文字区声明、财务模板幻句等训练语料高频噪声。
_HALLUC_MARKERS = (
    # 英文字母练习句（font specimen 常见，真实版面不会整句出现）
    'quick brown fox jumps over the lazy dog',
    # 无文字/图形区说明（模型把版面判断当正文输出）
    'The image contains no text', 'horizontal lines are', 'background elements',
    'must be ignored', 'must not be ignored',
    # 中文财务模板幻觉（完整句才触发，避免误伤正常含金融词的文档）
    '2017年1月1日', '公司与关联方', '4,000万元',
    # 英文版面碎句（读不出时逐行编造的残句）
    'Mrs. Egan said', 'Fairytale', 'Priscope', 'Antipodeus',
    'The world and a DC', 'They were a DC',
)

# 文本行 y 坐标越界阈值（归一化 [0,999]）：y1 贴底/超高说明模型在页面外继续虚构行
_Y1_OUT_OF_PAGE = 995
# 文本行 y 跨度阈值：正常文字行高度远小于整页，span>=500 说明是"整栏无文字说明"
_Y_SPAN_ABSURD = 500
# 重复文本行丢弃阈值：同一归一化文本出现 >=N 次视为循环（保留首次）
_REPEAT_DROP_N = 2


def _norm_text(t: str) -> str:
    """归一化文本用于重复检测：去行首序号/符号，小写并保留中英文/数字。"""
    s = re.sub(r'^[\s(（]*\d{1,3}[\s)）、.:，,]*', '', t)
    return re.sub(r'[^\w\u4e00-\u9fff]', '', s.lower())


def _cut_inline_loop(t: str) -> str:
    """截断行内循环：若文本含 >=4 字符单元连续重复 >=3 次，截到循环起点。

    覆盖单行内反复输出同一短句的幻觉（如 "The first, The first, The first…"、
    中文 "圖謀刺殺圖謀刺殺…" 4 字单元循环）。大小写不敏感匹配；未发现循环时
    原样返回。下限 4 字符：正常文本不会连续 3 次以上重复 4 字短语。
    """
    tl = t.lower()
    for u in range(4, min(61, len(tl) // 3 + 1)):
        s = 0
        while s + 3 * u <= len(tl):
            unit = tl[s:s + u]
            if tl[s + u:s + 2 * u] == unit and tl[s + 2 * u:s + 3 * u] == unit:
                return t[:s + u]
            s += 1
    return t


def strip_hallucinations(raw_text: str) -> str:
    """清理模型输出中的循环/声明式幻觉行（防幻觉护栏，纯后处理）。

    规则（按序，命中即丢弃该行）：
      1. det text 行 y1 >= 995：模型在页面底部外继续虚构行；
      2. det text 行 y 跨度 >= 500：模型把"整栏无文字"当正文输出；
      3. 文本含已知幻觉标记（_HALLUC_MARKERS）；
      4. 行内循环（_cut_inline_loop 截断）后仍与其它行重复 >= 2 次（保留首次）；
      5. 孤儿行若以 det 标签开头（解析失败碎片）整行丢弃。

    参数:
        raw_text: 模型原始输出（含 <|det|> 标签）。

    返回:
        清理后的文本。正常页面应逐字节不变（对合法文本零影响）。
    """
    det_re = re.compile(r'<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>\s*(.*)')
    parsed = []
    for ln in raw_text.split('\n'):
        m = det_re.match(ln)
        if m:
            # group(2) 形如 "[918, 28, 928, 38]"，去首尾方括号后切分
            nums = [int(x) for x in m.group(2)[1:-1].split(',')]
            y1, y2 = (nums[1], nums[3]) if len(nums) >= 4 else (-1, -1)
            parsed.append({
                'ln': ln, 'dt': m.group(1), 't': m.group(3),
                'y1': y1, 'y2': y2, 'orphan': False,
            })
        else:
            t = ln.strip()
            parsed.append({
                'ln': ln, 'dt': None, 't': t or None,
                'y1': None, 'y2': None, 'orphan': bool(t),
            })

    # 全局限重计数（det text 与孤儿行统一按归一化文本统计）
    cnt: dict[str, int] = {}
    for p in parsed:
        if p['t'] and len(_norm_text(p['t'])) >= 4:
            n = _norm_text(p['t'])
            cnt[n] = cnt.get(n, 0) + 1

    out: list[str] = []
    seen: set[str] = set()
    for p in parsed:
        t = p['t']
        if not t:
            out.append(p['ln'])
            continue

        # 规则5：孤儿行若是 det 解析失败碎片（以标签开头但格式残缺）→ 丢弃
        if p['orphan'] and t.startswith('<|det|>'):
            continue

        drop = False
        if not p['orphan'] and p['dt'] == 'text':
            # 规则1+2：越界/全页高行（模型在页面外或整栏虚构）
            if p['y1'] is not None and p['y1'] >= _Y1_OUT_OF_PAGE:
                drop = True
            elif p['y2'] is not None and p['y2'] - p['y1'] >= _Y_SPAN_ABSURD:
                drop = True

        # 规则3：已知幻觉/垃圾标记
        if not drop and any(mk in t for mk in _HALLUC_MARKERS):
            drop = True

        # 行内循环截断（det 行重建文本；孤儿行直接替换）
        t2 = t if drop else _cut_inline_loop(t)
        if not drop and t2 != t:
            if p['orphan']:
                t = t2
                p['ln'] = t2
            else:
                m2 = det_re.match(p['ln'])
                p['ln'] = p['ln'][:m2.start(3)] + t2  # 仅替换标签后的文本部分
            t = t2

        # 规则4：全局限重（>=N 次只保留首次）
        if not drop and t and len(_norm_text(t)) >= 4:
            n = _norm_text(t)
            if cnt.get(n, 0) >= _REPEAT_DROP_N:
                if n in seen:
                    drop = True
                else:
                    seen.add(n)

        out.append('' if drop else p['ln'])

    return '\n'.join(out)


def process_raw_output(raw_text: str, original_image_path: str, req_id: str) -> str:
    """
    将模型 eval_mode=True 返回的 raw 文本处理为完整 Markdown。

    Unlimited-OCR 模型在 eval_mode=True 时返回的原始文本格式如下：
        Text: "Free Speech & Data"
        <|det|>header [27, 26, 252, 37]<|/det|>CITYMAGAZINE / JUL 2009 / SPY
        <|det|>title [28, 381, 215, 399]<|/det|>[Fiskars] 芬蘭釵剪
        <|det|>image [27, 85, 387, 364]<|/det|>
        ...

    其中 <|det|> 标签的含义：
        - header:  页面主标题  → 映射为 # 一级标题
        - title:   区域标题    → 映射为 ## 二级标题
        - subtitle:区域副标题  → 映射为 ### 三级标题
        - text:    正文段落    → 保留纯文本
        - image:   嵌入图片    → 按坐标裁剪原图，替换为 ![](images/{req_id}/N.jpg)
        - page_number: 页码   → 跳过不输出

    坐标格式 [x1, y1, x2, y2] 归一化到 [0, 999]，需要按实际图片尺寸映射到像素坐标。

    alt 文本策略：
        记录最近遇到的 header/title/subtitle，后续 image 标签的 alt 自动使用该标题。
        若无标题，则降级为 "image"。

    Args:
        raw_text:            模型 eval_mode=True 返回的原始文本
        original_image_path: 原始输入图片的路径（用于裁剪提取子图）
        req_id:              本次请求的唯一标识，用于图片子目录命名

    Returns:
        格式化后的 Markdown 字符串（含标题层级、图片引用和 alt 文本）
    """
    # 延迟导入，避免循环依赖
    from PIL import Image as PILImage

    # ── 第1步：去除 EOS 终止标记 ──
    # 模型生成会在末尾附加 EOS token "<｜end▁of▁sentence｜>"
    stop_str = '<｜end▁of▁sentence｜>'
    if raw_text.endswith(stop_str):
        raw_text = raw_text[:-len(stop_str)]
    raw_text = raw_text.strip()

    # ── 第1.5步：幻觉护栏（后处理过滤循环/无中生有行） ──
    # 模型对读不出的区域（英文密排小字、图形栏）会循环重复同一短句或输出
    # "该栏无文字"等声明式幻觉，直至 max_tokens 耗尽。此步在解析前清洗，
    # 正常页面零影响（详见 strip_hallucinations docstring）。
    raw_text = strip_hallucinations(raw_text)

    # ── 第2步：加载原始图片（用于后续按坐标裁剪） ──
    # 图片宽高用于将 [0,999] 归一化坐标映射到实际像素坐标
    # 例如：坐标 [100, 50, 300, 200] 在 800×600 图片上对应像素区域 (80, 30, 240, 120)
    try:
        orig_img = PILImage.open(original_image_path)
        img_w, img_h = orig_img.size
    except Exception as e:
        # 如果原图加载失败，后面裁剪会跳过，但不阻塞 Markdown 文本生成
        logger.warning(
            "裁剪源图片打开失败: %s 异常=%s: %s",
            original_image_path, type(e).__name__, e,
        )
        orig_img = None
        img_w, img_h = 1, 1

    # ── 第3步：准备图片输出目录 ──
    # 提取的子图保存到 images/{req_id}/，通过 FastAPI StaticFiles 对外服务
    # URL 访问路径：http://host:port/images/{req_id}/0.jpg
    img_dir = IMAGES_DIR / req_id
    img_dir.mkdir(parents=True, exist_ok=True)

    # 检测列表项/独立条目: 以数字序号、破折号、圆点等开头
    # "01 xxx", "02 xxx", "- xxx", "1. xxx", "• xxx"
    _listish_pattern = re.compile(r'^(\d{1,2}\s|[•\-\*\–\—]\s|\d+\.\s)')
    # 匹配格式：<|det|>类型名 [坐标列表]<|/det|>
    # 类型名：字母开头，后续可含字母数字下划线连字符（如 header, title, image）
    # 坐标：方括号内的任意非方括号字符（如 [27,26,252,37]）
    det_pattern = re.compile(
        r'<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>'
    )

    # ── 第5步：逐行解析并构建 Markdown ──
    # 空行插入策略（只在类型变化时加空行）：
    #   文本段 → 文本段：不加空行（合并为同一段落）
    #   文本段 → 标题/图片：加空行
    #   图片 → 图片：不加空行（连续图片在一起）
    #   图片 → 文本/标题：加空行
    #   标题 → 任何：标题自身是块元素，渲染自然分隔
    lines = raw_text.split('\n')
    result_lines = []         # 最终输出的 Markdown 行列表
    last_heading = ""         # 记录最近遇到的标题文本，作为后续图片的 alt
    image_idx = 0             # 图片序号，用于文件命名（N.jpg）
    prev_type = ""            # 上一行的 det 类型
    prev_is_listish_flag = False  # 上一行是否为列表项（用于连续列表项合并）

    for line in lines:
        m = det_pattern.search(line)

        if not m:
            # ── 无 det 标签的行（孤立行） ──
            line_clean = line.strip()
            if line_clean:
                # 检测列表项：以序号、破折号、圆点开头
                if _listish_pattern.match(line_clean):
                    # 连续列表项不加空行
                    if prev_type and not (prev_type == "text" and prev_is_listish_flag):
                        result_lines.append("")
                    result_lines.append("- " + line_clean)
                    prev_type = "text"
                    prev_is_listish_flag = True
                else:
                    # 孤立行前后加空行
                    if prev_type:
                        result_lines.append("")
                    result_lines.append(line_clean)
                    result_lines.append("")
                    prev_type = "orphan"
                    prev_is_listish_flag = False
            continue

        # ── 解析 det 标签 ──
        det_type = m.group(1).strip()
        coords_str = m.group(2)
        text_after = line[m.end():].strip()

        mapping = DET_TYPE_MAP.get(det_type)
        if mapping is None:
            # 未知类型 → 视为 text
            if text_after:
                if prev_type not in ("", "text"):
                    result_lines.append("")
                result_lines.append(text_after)
                prev_type = "text"
            continue

        prefix, keep_text = mapping

        if det_type == "image":
            # ── 图片处理 ──
            # 类型变化时加空行（image→image 不加）
            if prev_type not in ("", "image"):
                result_lines.append("")
            try:
                coords = eval(coords_str)
                # 如果是单个坐标组 [100, 50, 300, 200]，包装为 [[100, 50, 300, 200]]
                if coords and isinstance(coords[0], (int, float)):
                    coords = [coords]
            except Exception as e:
                # 坐标解析失败（格式异常），跳过此图片
                logger.warning(
                    "图片坐标解析失败: req_id=%s 坐标=%s 异常=%s: %s",
                    req_id, coords_str, type(e).__name__, e,
                )
                coords = []

            # 记录实际保存成功的子图文件名（裁剪失败的图片不会生成链接）
            saved_files: list[str] = []

            # 遍历可能的多组坐标（一个 image 区域可能包含多张子图）
            for ci, c in enumerate(coords):
                fname = f"{image_idx}{('_' + str(ci)) if len(coords) > 1 else ''}.jpg"
                try:
                    # 将 [0,999] 归一化坐标映射到实际像素坐标
                    # 公式：像素坐标 = 归一化坐标 / 999 × 实际尺寸
                    x1 = int(c[0] / 999 * img_w)
                    y1 = int(c[1] / 999 * img_h)
                    x2 = int(c[2] / 999 * img_w)
                    y2 = int(c[3] / 999 * img_h)

                    # 只有坐标有效且原图加载成功才执行裁剪
                    if orig_img is not None and x2 > x1 and y2 > y1:
                        cropped = orig_img.crop((x1, y1, x2, y2))
                        # 多图时加 _ci 后缀区分：0_0.jpg, 0_1.jpg
                        suffix = f"_{ci}" if len(coords) > 1 else ""
                        cropped.save(str(img_dir / fname))
                        saved_files.append(fname)
                    else:
                        # 坐标无效或原图缺失：记录原因，便于排查"无图/死链"问题
                        logger.warning(
                            "图片裁剪跳过: req_id=%s 文件=%s 坐标=%s 原图=%s 原因=%s",
                            req_id, fname, c,
                            "已加载" if orig_img is not None else "缺失",
                            "坐标无效(x2<=x1 或 y2<=y1)" if x2 <= x1 or y2 <= y1 else "未知",
                        )
                except Exception as e:
                    # 单张图片裁剪失败不影响其他图片，但必须记录，便于排查
                    logger.warning(
                        "图片裁剪异常: req_id=%s 文件=%s 坐标=%s 原图=%s 异常=%s: %s",
                        req_id, fname, c, original_image_path, type(e).__name__, e,
                    )
                    continue

            # 只有实际保存成功的图片才生成 Markdown 链接；
            # 若该 image 标签的所有子图均裁剪失败，则不输出任何图片（避免死链 404）
            if not saved_files:
                # 汇总日志：整条 image 标签无任何子图产出
                logger.warning(
                    "图片裁剪全部失败，跳过输出: req_id=%s 坐标=%s",
                    req_id, coords_str,
                )
                continue

            # ── 生成 alt 文本 ──
            # 策略：使用最近遇到的标题（由 header/title/subtitle 标签记录）
            # 如果前面没有标题，降级为 "image"
            alt = last_heading if last_heading else "image"

            # ── 构建 Markdown 图片引用 ──
            # 单图: ![标题文本](images/req_id/0.jpg)
            # 多图: ![标题文本 (1)](images/req_id/0_0.jpg)
            #       ![标题文本 (2)](images/req_id/0_1.jpg)
            for ci, fname in enumerate(saved_files):
                if len(saved_files) == 1:
                    result_lines.append(f"![{alt}](images/{req_id}/{fname})")
                else:
                    result_lines.append(f"![{alt} ({ci+1})](images/{req_id}/{fname})")

            image_idx += 1
            prev_type = "image"

        elif keep_text:
            # ── 文本/标题类型 ──
            md_line = f"{prefix}{text_after}" if text_after else ""
            if md_line:
                if det_type in ("header", "title", "subtitle"):
                    last_heading = text_after if text_after else last_heading
                    # 标题前加空行（与上文分隔，标题→标题除外）
                    if prev_type not in ("", "header", "title", "subtitle"):
                        result_lines.append("")
                    result_lines.append(md_line)
                    prev_type = det_type
                else:
                    # 正文段落
                    # 检测当前行是否为列表项（01, 02, - item, 1. item 等）
                    # 列表项不应与上文合并，需独立成行
                    cur_is_listish = bool(_listish_pattern.match(text_after))
                    # 上一行也是列表项 → 不加空行（连续列表项保持紧凑）
                    prev_was_listish = bool(prev_type == "text" and prev_is_listish_flag)
                    if cur_is_listish:
                        if not prev_was_listish:
                            result_lines.append("")  # 普通文本 → 列表，加空行
                        result_lines.append("- " + text_after)  # 统一用 - 前缀
                        prev_is_listish_flag = True
                    else:
                        if prev_type not in ("", "text") or prev_is_listish_flag:
                            result_lines.append("")
                        result_lines.append(md_line)
                        prev_is_listish_flag = False
                    prev_type = "text"

        # 注意：det_type 为非 image 且 keep_text=False 的情况（如 page_number）
        # 直接跳过，不输出任何内容

    # ── 第6步：清理并返回 ──
    # 替换 LaTeX 数学模式转义字符为标准符号
    result = '\n'.join(result_lines)
    result = result.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    # 压缩连续的多个空行为单个空行（Markdown 规范：段落间一个空行即可）
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


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
    """健康检查。返回下游引擎状态与空闲时间。"""
    with _state_lock:
        ok = _engine_ok
        err = _engine_error
        loaded_at = _loaded_at
        last_used = _last_used
        total = _total_requests
        container_running = _container_running

    idle_time = time.time() - last_used if last_used else 0

    return JSONResponse({
        "status": "ok" if ok else ("stopped" if not container_running else "degraded"),
        # 语义保持兼容：下游引擎可达即视为“模型已就绪”
        "model_loaded": ok,
        "engine_error": err or None,
        "container_running": container_running,
        "idle_seconds": idle_time,
        "loaded_at": loaded_at,
        "last_used": last_used,
        # 下游引擎空闲超时后容器被停止；该值保留字段兼容
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
    """
    手动卸载模型以释放 GPU 显存（对齐原版行为）。

    停止下游 vLLM 引擎容器并释放显存；下次请求到达时自动重新拉起。
    持 _infer_lock 执行：若此刻有推理在跑则等待其完成后再停止，
    避免打断进行中的请求。
    """
    with _infer_lock:
        _stop_container()
    return JSONResponse({"ok": True, "unloaded": True})


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OCR 推理主接口（OpenAI Chat Completions 兼容）。

    支持单图和多图两种模式:
    - 单图 (1 张 image_url): 使用 gundam 模式（crop_mode=True, image_size=640）
    - 多图 (≥2 张 image_url): 使用 base 模式（image_size=1024），一次推理处理所有页
      输出页间用 <PAGE> 分隔，每页独立提取图片和 Markdown 结构

    也支持 PDF 文件: 传入 .pdf 文件路径，服务端自动转换为多图后推理。

    推理在 `_infer_lock` 内串行执行，同一时间只处理一个请求
    （对齐原 Transformers 版单 GPU 串行语义）。
    """
    temps = []
    try:
        # ── 第1步：解析消息，提取所有图片路径 ──
        prompt, imgs, temps = extract_messages(req.messages)
        if not imgs:
            return error_response(400, "消息中未找到图片", "invalid_request")

        # 自动检测 PDF 文件并转换为图片
        # 支持: 本地 .pdf 文件路径、data:application/pdf base64 编码
        is_pdf = imgs[0].lower().endswith('.pdf')
        if is_pdf:
            import fitz  # PyMuPDF
            pdf_path = imgs[0]
            pdf_dpi = 300
            pdf_tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
            mat = fitz.Matrix(pdf_dpi / 72, pdf_dpi / 72)
            pdf_imgs = []
            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc):
                out = os.path.join(pdf_tmp_dir, f"page_{i + 1:04d}.png")
                page.get_pixmap(matrix=mat).save(out)
                pdf_imgs.append(out)
            doc.close()
            # 替换: 用转换后的图片列表替代原 PDF 路径
            temps.append(pdf_tmp_dir)  # 清理时会删除
            imgs = pdf_imgs
            if "Multi page" not in prompt:
                prompt = "<image>Multi page parsing."

        # ── 应用 max_pages 限制（仅对多图/PDF 生效） ──
        if req.max_pages is not None and req.max_pages > 0 and len(imgs) > req.max_pages:
            imgs = imgs[:req.max_pages]

        # ── 第2步：确保下游引擎可达（懒加载）并标记使用时间 ──
        # 全部在 _infer_lock 内完成：冷启动 docker start + 模型加载可能耗时
        # 60-120s，期间持锁可与看门狗的“空闲停止”互斥，避免停/用竞态。
        req_id = uuid.uuid4().hex[:12]

        with _infer_lock:
            _ensure_engine_locked()
            _touch_used()

            # 决定使用哪种推理路径
            use_batch = (len(imgs) > 1 and req.page_mode == "batch")

            if len(imgs) == 1 or req.page_mode == "single":
                # ═══════════════════════════════════════════════════════
                # 逐张模式: 每张图单独发一次请求（等价原 model.infer()
                # gundam 640 crop_mode —— vLLM 对单图请求自动启用 crop）
                # 多图时逐张循环处理，结果用 --- 合并。
                # ═══════════════════════════════════════════════════════
                all_processed = []
                for page_idx, img_path in enumerate(imgs):
                    orig_copy = IMAGES_DIR / req_id / f"_page_{page_idx}.png"
                    orig_copy.parent.mkdir(parents=True, exist_ok=True)
                    from PIL import Image as PILImage
                    PILImage.open(img_path).save(str(orig_copy))
                    # 立即验证副本可完整读取：副本损坏会导致后续裁剪失败（静默丢图）
                    try:
                        _verify = PILImage.open(orig_copy)
                        _verify.load()
                    except Exception as _e:
                        logger.error(
                            "裁剪源副本损坏: req_id=%s page=%d 文件=%s 异常=%s: %s",
                            req_id, page_idx, orig_copy, type(_e).__name__, _e,
                        )

                    # 单图请求 → vLLM 内部自动 crop（等价原 infer 单图路径）
                    raw_text = vllm_chat(prompt, [img_path])
                    page_md = process_raw_output(raw_text, str(orig_copy), f"{req_id}/page_{page_idx}")
                    if page_md and page_md != "[empty]":
                        all_processed.append(page_md)

                text = "\n\n---\n\n".join(all_processed) if all_processed else "[empty]"

            elif use_batch:
                # ═══════════════════════════════════════════════════════
                # 批量模式: 多张图一次请求（等价原 infer_multi base 1024）
                # vLLM 多图请求自动回退非 crop 模式，输出以 <PAGE> 分隔。
                # 每批图片数受 VLLM_MAX_IMAGES_PER_REQ 限制（引擎 context 上限）。
                # ═══════════════════════════════════════════════════════
                total_pages = len(imgs)
                batch_size = min(MAX_PAGES_PER_BATCH, VLLM_MAX_IMAGES_PER_REQ)
                all_processed = []
                global_page_idx = 0

                for batch_start in range(0, total_pages, batch_size):
                    batch_end = min(batch_start + batch_size, total_pages)
                    batch_imgs = imgs[batch_start:batch_end]

                    raw_output = vllm_chat(prompt, batch_imgs)

                    pages = raw_output.split('<PAGE>')[1:]
                    for page_raw in pages:
                        page_raw = page_raw.strip()
                        if not page_raw:
                            continue
                        if global_page_idx < total_pages:
                            page_copy = IMAGES_DIR / req_id / f"_page_{global_page_idx}.png"
                            page_copy.parent.mkdir(parents=True, exist_ok=True)
                            from PIL import Image as PILImage
                            PILImage.open(imgs[global_page_idx]).save(str(page_copy))
                            # 立即验证副本可完整读取：副本损坏会导致后续裁剪失败（静默丢图）
                            try:
                                _verify = PILImage.open(page_copy)
                                _verify.load()
                            except Exception as _e:
                                logger.error(
                                    "裁剪源副本损坏: req_id=%s page=%d 文件=%s 异常=%s: %s",
                                    req_id, global_page_idx, page_copy,
                                    type(_e).__name__, _e,
                                )
                            page_md = process_raw_output(
                                page_raw, str(page_copy), f"{req_id}/page_{global_page_idx}"
                            )
                            if page_md and page_md != "[empty]":
                                all_processed.append(page_md)
                        global_page_idx += 1

                text = "\n\n---\n\n".join(all_processed) if all_processed else "[empty]"

        obj_id = f"chatcmpl-{req_id}"

        # ── 流式输出 ──
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
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
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
