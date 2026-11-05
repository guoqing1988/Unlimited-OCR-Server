"""
Unlimited-OCR：OpenAI 兼容 OCR 推理服务（llama.cpp GGUF 后端适配层）。

对外 API 与 server_vllm.py 完全一致，但下游引擎从 vLLM（docker）换成
打了 Unlimited-OCR-2 补丁的 llama-server（子进程，端口 9706）。

后端差异：
- 模型：Unlimited-OCR-2 Q8_0 GGUF + mmproj（视觉编码器）
- 输出格式：v2 使用 <BLOCK>type[box]<FORK>文本，跨页合并用 <MERGE>
  （v1 的 <|det|> 格式已不再出现）
- 多页请求：只把图片发给 llama-server，由 GGUF 内嵌 chat template
  自动生成 "<PAGE><image>...Multi page parsing."；若同时发文本，
  模板会把文本放错位置并导致只输出第一页（实测）。
- 防重复：v1 用 vLLM 的 NGramPerReqLogitsProcessor，v2 改用 DRY 采样
  （请求级 dry_multiplier / dry_base / dry_allowed_length / dry_penalty_last_n）
- 引擎生命周期：懒启动 + 空闲停止 llama-server 子进程（替代 docker stop）

启动方式：
    uvicorn server_gguf:app --host 0.0.0.0 --port 9705

接口:（与 server_vllm.py 完全一致）
    GET  /health              — 健康检查（含下游引擎状态和空闲时间）
    GET  /v1/models           — 模型列表
    POST /v1/chat/completions — 多模态 OCR 推理
    POST /admin/unload        — 卸载模型（停止 llama-server，释放显存）
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
from collections import deque
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

# 下游 llama-server 的 OpenAI 兼容端点（本适配层不直接加载模型）
# 请求需显式传 skip_special_tokens=False，才能保留 <BLOCK>/<FORK> 等
# 特殊 token 供 process_raw_output() 解析。
# 兼容环境变量 LLAMA_URL / ENGINE_URL
LLAMA_URL = (
    os.environ.get("LLAMA_URL")
    or os.environ.get("ENGINE_URL")
    or "http://127.0.0.1:9706/v1"
).rstrip("/")

# 补丁版 llama-server 可执行文件（必须包含 unlimited-ocr2 架构支持）
LLAMA_SERVER_BIN = os.environ.get(
    "LLAMA_SERVER_BIN",
    "/data1/www/llama-ocr2/build/bin/llama-server",
)

# 文本模型 / 视觉编码器 GGUF 路径
LLAMA_MODEL = os.environ.get(
    "LLAMA_MODEL",
    "/data1/models/Unlimited-OCR-2-GGUF/Unlimited-OCR-2-Q8_0.gguf",
)
LLAMA_MMPROJ = os.environ.get(
    "LLAMA_MMPROJ",
    "/data1/models/Unlimited-OCR-2-GGUF/mmproj-Unlimited-OCR-2-F16.gguf",
)

# llama-server 监听端口与上下文长度。
# ctx 决定显存占用（实测：4k≈6.0GB、8k≈6.4GB、32k≈8.4GB），
# 也决定一次 batch 能塞下多少页（base 模式约 273 image token/页）。
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "9706"))
LLAMA_CTX = int(os.environ.get("LLAMA_CTX", "32768"))

# 传给 llama-server 的 GPU（CUDA_VISIBLE_DEVICES 值；空=继承当前环境）
LLAMA_GPU = os.environ.get("LLAMA_GPU", "")

# 额外 llama-server 参数（空格分隔，由 shlex.split 解析）
# 默认 --jinja 启用 GGUF 内嵌 chat template（多页 <PAGE> 逻辑依赖它）
LLAMA_EXTRA_ARGS = os.environ.get("LLAMA_EXTRA_ARGS", "")

# llama-server 日志文件（stdout/stderr 重定向）
LLAMA_LOG = os.environ.get(
    "LLAMA_LOG",
    str(Path(__file__).parent / "log" / "llama-server.log"),
)

# 启动 llama-server 后等待引擎就绪的超时时间（秒）。
# Q8_0 冷启动（加载 GGUF + mmproj）约 20-40s。
LLAMA_START_TIMEOUT = int(os.environ.get("LLAMA_START_TIMEOUT", "240"))

# 下游引擎推理时使用的生成 token 上限
LLAMA_MAX_TOKENS = int(os.environ.get("LLAMA_MAX_TOKENS", "8192"))

# ── DRY 防重复（关键！）──────────────────────────────────────────────
# v2 模型对读不出的区域会陷入短句循环（如 "I'll. I'll. I'll. …"）。
# 原 vLLM 方案靠 NGramPerReqLogitsProcessor 抑制；llama.cpp 没有该处理器，
# 改用 DRY 采样：实测默认参数下会循环，开启下列参数后 3 页全量输出无重复。
DRY_MULTIPLIER = float(os.environ.get("DRY_MULTIPLIER", "0.8"))
DRY_BASE = float(os.environ.get("DRY_BASE", "1.75"))
DRY_ALLOWED_LENGTH = int(os.environ.get("DRY_ALLOWED_LENGTH", "2"))
DRY_PENALTY_LAST_N = int(os.environ.get("DRY_PENALTY_LAST_N", "256"))

# 下游引擎名称（llama-server 启动时 --alias 指定的名称）
ENGINE_MODEL_NAME = os.environ.get("ENGINE_MODEL_NAME", "Unlimited-OCR")

# 单次请求向下游引擎最多能接受的总图片数（超过则分多批）。
# v2 base 模式约 273 image token/页；LLAMA_CTX=32768 时 3 页很宽松，
# 若调小 LLAMA_CTX，应同步下调本值（batch 推理总长度 = prompt + 生成）。
LLAMA_MAX_IMAGES_PER_REQ = int(os.environ.get("LLAMA_MAX_IMAGES_PER_REQ", "3"))

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

# ── 重复图片去重 ──────────────────────────────────────────────────────
# v2 模型在双栏/大图版面偶发把同一张图输出两次（坐标几乎相同，前后可能夹着正文），
# 直接裁剪会得到两张一模一样的图片。与已输出图片的 IoU ≥ 本阈值时视为重复并跳过。
# 设为 0 可关闭去重。
IMAGE_DEDUP_IOU = float(os.environ.get("IMAGE_DEDUP_IOU", "0.85"))

# ── 引擎健康状态 ───────────────────────────────────────────────────────────
_state_lock = threading.Lock()   # 保护状态字段
_infer_lock = threading.Lock()   # 推理互斥，保证单 GPU 串行

# 模型不再由本进程加载 —— 推理引擎是下游 llama-server 子进程。
# 这里仅记录下游引擎的可达状态与请求计数，用于 /health 展示。
_engine_ok: bool = False          # 最近一次探测下游引擎是否可达
_engine_error: str = ""          # 探测失败原因
_loaded_at: Optional[float] = None   # 下游引擎首次探测成功时间
_last_used: Optional[float] = None   # 最后一次推理请求时间
_total_requests: int = 0

# 引擎进程生命周期状态（空闲自动卸载 / 按需启动用）
_engine_proc: subprocess.Popen | None = None  # llama-server 子进程句柄
_container_running: bool = False       # 引擎进程当前是否运行（字段名保持 /health 兼容）
_engine_stop_ts: float | None = None   # 最近一次主动停止引擎的时间戳（防止竞态误启动）

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
# 下游 llama-server 引擎探测 / 进程生命周期管理
# ═══════════════════════════════════════════════════════════════════════════


def _engine_status() -> str:
    """
    查询下游 llama-server 子进程状态。

    返回:
        "running": 进程存活；"exited": 进程已退出；
        "missing": 从未启动（或已回收句柄）。
    """
    global _container_running, _engine_proc
    with _state_lock:
        proc = _engine_proc
    if proc is None:
        return "missing"
    if proc.poll() is None:
        with _state_lock:
            _container_running = True
        return "running"
    with _state_lock:
        _container_running = False
    return "exited"


def _build_engine_cmd() -> list[str]:
    """
    构造 llama-server 启动命令。

    关键参数:
        --jinja           启用 GGUF 内嵌 chat template（多页 <PAGE> 逻辑依赖它）
        -np 1             单 slot：模型按 token 历史推导页位置，只支持单序列
        -c LLAMA_CTX      上下文长度，同时决定显存占用与单批可容纳页数
        --alias           让 /v1/models 返回统一模型名，便于 _probe_engine 校验

    返回:
        可直接交给 subprocess.Popen 的参数列表。
    """
    cmd = [
        LLAMA_SERVER_BIN,
        "-m", LLAMA_MODEL,
        "--mmproj", LLAMA_MMPROJ,
        "--jinja",
        "-np", "1",
        "-c", str(LLAMA_CTX),
        "--host", "127.0.0.1",
        "--port", str(LLAMA_PORT),
        "-ngl", "99",
        "--alias", ENGINE_MODEL_NAME,
    ]
    if LLAMA_EXTRA_ARGS.strip():
        cmd += shlex.split(LLAMA_EXTRA_ARGS)
    return cmd


def _start_engine() -> None:
    """
    启动下游 llama-server 子进程（不等待就绪）。

    就绪由 _wait_engine_ready() 轮询 /v1/models 判断。
    子进程 stdout/stderr 重定向到 LLAMA_LOG，便于排查加载失败。
    """
    global _engine_proc, _container_running, _engine_ok, _engine_error, _loaded_at
    if _engine_status() == "running":
        return

    Path(LLAMA_LOG).parent.mkdir(parents=True, exist_ok=True)
    log_f = open(LLAMA_LOG, "ab", buffering=0)
    env = os.environ.copy()
    if LLAMA_GPU:
        env["CUDA_VISIBLE_DEVICES"] = LLAMA_GPU
    cmd = _build_engine_cmd()
    logger.info("启动 llama-server: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=log_f, env=env,
            start_new_session=True,  # 独立进程组，避免适配层退出时被连带杀掉
        )
    finally:
        log_f.close()
    with _state_lock:
        _engine_proc = proc
        _container_running = True
        _engine_ok = False        # 启动后需重新探测确认
        _engine_error = "llama-server 已启动，等待引擎就绪..."
        _loaded_at = None


def _stop_engine() -> None:
    """
    停止下游 llama-server 并释放 GPU 显存。

    对应原 vLLM 版的 _stop_container()：空闲超时或 /admin/unload 触发后
    终止引擎进程，下次请求时再由 _ensure_engine() 按需拉起。
    """
    global _engine_proc, _container_running, _engine_ok, _engine_error
    global _engine_stop_ts
    with _state_lock:
        proc = _engine_proc
    if proc is None or proc.poll() is not None:
        with _state_lock:
            _engine_proc = None
            _container_running = False
            _engine_ok = False
            _engine_error = "引擎未运行（空闲自动卸载），等待下次请求重新拉起"
            _engine_stop_ts = time.time()
        return

    try:
        logger.info("空闲超时/手动触发，停止 llama-server (pid=%s) ...", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server 未在 30s 内退出，强制 kill")
            proc.kill()
            proc.wait(timeout=10)
    except Exception as exc:
        logger.warning("停止 llama-server 异常: %s", exc)

    with _state_lock:
        _engine_proc = None
        _container_running = False
        _engine_ok = False
        _engine_error = "引擎已停止（空闲自动卸载），等待下次请求重新拉起"
        _engine_stop_ts = time.time()
    logger.info("llama-server 已停止，显存已释放。")


def _probe_engine() -> bool:
    """
    探测下游 llama-server 是否可达（GET /v1/models）。

    返回:
        True 表示引擎可达且模型列表包含 ENGINE_MODEL_NAME；否则 False。
    """
    global _engine_ok, _engine_error, _loaded_at
    try:
        req = urllib.request.Request(f"{LLAMA_URL}/models", method="GET")
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


def _wait_engine_ready(timeout: float = LLAMA_START_TIMEOUT) -> None:
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

    ⚠️ 调用方必须已持有 _infer_lock，保证“启动/等待就绪/推理”
    与看门狗的停止动作互斥。

    - 引擎已就绪：直接返回；
    - 引擎进程存活但未就绪：等待加载完成；
    - 引擎进程不存在/已退出：启动后等待就绪。
    """
    with _state_lock:
        if _engine_ok:
            return
    status = _engine_status()
    # 先探测一次：进程句柄丢失但引擎实际可达（如适配层重启后残留的
    # llama-server）时直接复用，避免重复启动导致端口冲突
    if _probe_engine():
        with _state_lock:
            _container_running = True
        return
    if status != "running":
        _start_engine()
    _wait_engine_ready()


def _ensure_engine():
    """
    确保下游引擎可用（线程安全入口，带 _infer_lock）。

    冷启动可能耗时较长（加载 GGUF + mmproj），期间持有 _infer_lock，
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
    后台看门狗线程：空闲超时自动停止引擎（对齐原版空闲卸载）。

    每 WATCHDOG_POLL_SECONDS 秒检查一次：
    - 引擎在运行且探测可达；
    - 距最后一次推理请求超过 IDLE_UNLOAD_SECONDS；
    满足则停止 llama-server 释放显存。停止动作在 _infer_lock 内执行：
    若此刻恰好有请求正在拉起/使用引擎，则等待其完成，避免竞态。
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
                        _stop_engine()
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
    模型只在首次请求时加载）；仅记录当前引擎状态，首次请求到达时
    由 _ensure_engine() 按需启动 llama-server 并等待就绪。
    """
    try:
        _engine_status()
    except Exception:
        traceback.print_exc()
    th = threading.Thread(target=_watchdog, daemon=True)
    th.start()


@app.on_event("shutdown")
def shutdown():
    """
    FastAPI 关闭事件。

    停止 llama-server，避免适配层退出/重启后残留孤儿进程占用显存。
    """
    with _infer_lock:
        _stop_engine()


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
# 键: 类型名, 值: (Markdown 前缀, 是否保留标签后的文本)
# v2 使用 <BLOCK>type[box]<FORK>text 格式，类型集合与 v1 基本一致，
# 新增 image_caption（图片说明，按正文输出）、title 的 <LEVELn> 层级标记。
DET_TYPE_MAP = {
    "header":       ("# ", True),      # 一级标题
    "title":        ("## ", True),     # 二级标题（<LEVEL2> 时降为三级）
    "subtitle":     ("### ", True),    # 三级标题
    "text":         ("", True),        # 正文段落
    "image":        ("", False),       # 图片（只裁剪，不输出文本）
    "image_caption": ("", True),      # 图片说明（按正文段落输出）
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
# 下游 llama-server HTTP 推理转发
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


# ── 生成期循环熔断（流式早停）──────────────────────────────────
# 模型对读不出的页面会整行重复（同一行反复输出），跑满 max_tokens 才停
# （实测 8192 tokens ≈ 40s）。这里改用流式接收，一旦检测到整行循环就主动
# 断开连接：llama-server 会立即 cancel 任务释放算力，已生成的有效内容照常
# 返回，最坏耗时从 40s 降到几秒。后处理的 strip_hallucinations 仍作为第二道防线。
_STREAM_LOOP_MIN_LINES = 25     # 至少累积行数后才启用检测（避免短输出误判）
_STREAM_LOOP_REPEAT_N = 4       # 同一行（规范化后）出现 >=N 次判为循环
_STREAM_LOOP_WINDOW = 60        # 只看最近 N 行
_STREAM_LOOP_NGRAM_SIZE = 5     # 近似重复检测的字符 n-gram 长度
_STREAM_LOOP_NGRAM_MOD = 0.7    # 同一 n-gram 覆盖 >=70% 的行判为近似循环


def _stream_loop_detected(recent, since_last: int = 0) -> bool:
    """判断最近输出行是否已陷入循环（用于流式早停）。

    两条规则（命中任一即判循环）：
      1. 整行完全相同：同一行规范化后出现 >= _STREAM_LOOP_REPEAT_N 次；
      2. 近似重复：同一 5 字词组出现在窗口内 >= 70% 的行中
         （模型对读不出的区域会反复套同一句式，每行微调几个字，
         整行不相同但词组成分高度重复，如 “本調查委員會…本調查委員會…”）。

    要求总行数 >= _STREAM_LOOP_MIN_LINES，避免短文档的合法重复被误判。

    Args:
        recent: 最近若干行文本（deque），已去掉结构标签行与空行。
        since_last: 距上次检测新增的行数；不足 3 行时跳过开销较大的
                    n-gram 统计（每次 chunk 都算会拖慢流式接收）。

    Returns:
        True 表示检测到循环，调用方应立即断开流。
    """
    if len(recent) < _STREAM_LOOP_MIN_LINES:
        return False

    # ── 规则 1：整行完全相同 ──
    cnt: dict[str, int] = {}
    for ln in recent:
        n = _norm_text(ln)
        if len(n) < 4:
            continue
        cnt[n] = cnt.get(n, 0) + 1
        if cnt[n] >= _STREAM_LOOP_REPEAT_N:
            return True

    # ── 规则 2：近似重复（同一词组覆盖大多数行）──
    if since_last < 3:
        return False
    rows: list[str] = []
    for ln in recent:
        txt = re.sub(r'\s+', '', ln.split('<FORK>')[-1])
        if len(txt) >= 8:
            rows.append(txt)
    if len(rows) < 20:
        return False
    gram: dict[str, int] = {}
    for r in rows:
        for g in {r[i:i + _STREAM_LOOP_NGRAM_SIZE] for i in range(len(r) - _STREAM_LOOP_NGRAM_SIZE + 1)}:
            gram[g] = gram.get(g, 0) + 1
    if not gram:
        return False
    return max(gram.values()) >= _STREAM_LOOP_NGRAM_MOD * len(rows)


def llama_chat(
    prompt: str,
    image_paths: list[str],
    max_tokens: int = LLAMA_MAX_TOKENS,
    temperature: float = 0.0,
) -> str:
    """
    转发一次 OCR 推理请求到下游 llama-server，返回模型原始文本输出。

    与 v1 的 vllm_chat() 等价，但请求体构造不同：
    - 图片放在 content 数组中，**不附带默认指令文本**。GGUF 内嵌 chat
      template 会自动在每个图片前插入 <PAGE>，并在消息末尾补
      "Multi page parsing."；若同时发文本，模板会把文本渲染到图片之前，
      导致多页只输出第一页（实测）。
    - 只有调用方显式传入自定义指令（非默认 "document parsing." /
      "Multi page parsing."）时，才把指令放在图片之后发送。

    参数:
        prompt:       最终 prompt（含 <image> 占位符文本）。
        image_paths:  本地图片路径列表（单图传 [path]，多图传全部）。
        max_tokens:   生成 token 上限。
        temperature:  采样温度。

    返回:
        模型原始输出文本（含 <BLOCK>/<FORK>/<PAGE>/<MERGE> 标签）。

    异常:
        RuntimeError: 引擎不可达或返回非 200。
    """
    # ── 指令文本处理（实测结论，勿改）────────────────────────────
    # 模型（Unlimited-OCR-2）只认 "Multi page parsing."：
    #   - 不发文本 → 内嵌模板自动补 "Multi page parsing." → 正确
    #   - 显式发 "Multi page parsing."               → 正确
    #   - 发 "document parsing."                      → 幻觉循环（实测）
    # 所以官方两个默认指令都不显式发送，交给模板；只有真正的自定义指令
    # （如 "请提取表格"）才放在图片之后发送。
    # 注意：自定义文本必须放在图片之后，否则模板会把它渲染到第一页前面。
    custom = prompt.replace("<image>", "").strip()
    if custom in ("document parsing.", "Multi page parsing."):
        custom = ""

    content: list[dict] = []
    for p in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_uri(p)},
        })
    # ── 指令文本必须放在图片之后 ──
    # 模板会渲染为 "<PAGE><image>Multi page parsing."；若文本在图片之前会被
    # 放到第一页前面（多页时会错位）。
    if custom:
        content.append({"type": "text", "text": custom})

    # 流式接收的累积缓冲与循环检测窗口
    chunks: list[str] = []
    recent: deque[str] = deque(maxlen=_STREAM_LOOP_WINDOW)
    loop_broken = False
    line_buf = ""          # 未以换行结尾的残行（等后续 chunk 拼完再入窗口）
    since_last_check = 0   # 距上次循环检测新增的完整行数

    payload = {
        "model": ENGINE_MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # 流式返回：用于生成期循环检测（检测到整行循环立即断开，省下大量时间）
        "stream": True,
        # 关键：保留 <BLOCK>/<FORK>/<PAGE> 等特殊 token 供后处理解析
        "skip_special_tokens": False,
        # 关键：DRY 防重复（llama.cpp 无 vLLM 的 NGram 处理器，
        # 关闭 DRY 时模型短句循环无防护，会重复到 max_tokens 耗尽）
        "dry_multiplier": DRY_MULTIPLIER,
        "dry_base": DRY_BASE,
        "dry_allowed_length": DRY_ALLOWED_LENGTH,
        "dry_penalty_last_n": DRY_PENALTY_LAST_N,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{LLAMA_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            if hasattr(resp, "__iter__"):
                # ── 流式接收：边收边检测整行循环 ──
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    for ch in obj.get("choices") or []:
                        piece = (ch.get("delta") or {}).get("content")
                        if not piece:
                            continue
                        chunks.append(piece)
                        # 按换行切出完整行再入窗口（最后一段残行留给下个 chunk）
                        line_buf += piece
                        while "\n" in line_buf:
                            ln, line_buf = line_buf.split("\n", 1)
                            ln = ln.strip()
                            # 结构标签行（<PAGE>/<MERGE>）不参与循环统计
                            if ln and not ln.startswith("<"):
                                recent.append(ln)
                                since_last_check += 1
                    if since_last_check >= 3 and _stream_loop_detected(recent, since_last_check):
                        loop_broken = True
                        break
            else:
                # 非流式响应（测试替身 / 旧版引擎）：一次性解析
                data = json.loads(resp.read())
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(
                        f"llama-server 响应缺少 choices: {json.dumps(data)[:300]}"
                    )
                return choices[0]["message"].get("content", "")
    except urllib.error.HTTPError as e:
        # 读取错误响应体，便于排查
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        raise RuntimeError(f"llama-server 返回 {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"llama-server 不可达: {e.reason}") from e

    if loop_broken:
        logger.info(
            "生成期检测到整行循环，提前断开（已接收 %d 字符，引擎已停止生成）",
            sum(len(c) for c in chunks),
        )
    return "".join(chunks)


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

# 模型格式崩溃时产生的碎片行特征：
#   1) 缺 <BLOCK> 前缀的块标签，如 image[0, 0, 999, 999]、text[12, 34, 56, 78]
#   2) 残缺标签，如单独一行的 <BLOCK>/<FORK>/<PAGE>/<MERGE>/<LEVEL1>
_JUNK_LABEL_RE = re.compile(
    r'^\s*(?:'
    r'<(?:BLOCK|FORK|LEVEL\d+)>'  # 残缺标签（<PAGE>/<MERGE> 是结构标签，不在此列）
    r'|(?:image|text|title|subtitle|header|footer|page_number)\s*\['  # 缺 BLOCK 前缀的块标签
    r')'
)
# 纯坐标数组行（必须以 [ 开头或 ] 结尾，避免误伤 "1, 2, 3" 这类纯数字正文）
_JUNK_COORD_RE = re.compile(r'^\s*(?:\[[\s\[\]0-9,，.\-]*|[\s0-9,，.\-]*\])\s*$')
# 纯标点/省略号行（如 "..."、"。。"），至少 2 个标点字符
_JUNK_PUNCT_RE = re.compile(r'^\s*[.…。、，,·:：;；\-—_~!！?？]{2,}\s*$')


def _is_junk_fragment(t: str) -> bool:
    """判断孤立行是否为标签/坐标/标点碎片（模型格式崩溃产物）。

    模型在难识别的页面上偶发输出残缺标签与坐标数组，例如：
        image[0, 0, 999, 999]
        [119, 1, 2, 0, 3, 0
        [0, 200, 800, 400, 2000]
        [1]
    这些行没有完整 ``<BLOCK>type[coords]<FORK>文本`` 结构，不属于正文，
    必须在解析阶段整行丢弃，否则会原样出现在 Markdown 结果里。

    含任何文字字符的正常行不会被误判（正常表格行如 "1, 2, 3" 也无 [] 包围）。

    Args:
        t: 待判断的孤立行文本（已 strip）。

    Returns:
        True 表示是碎片行，应丢弃。
    """
    s = t.strip()
    if not s:
        return False
    return bool(
        _JUNK_LABEL_RE.match(s) or _JUNK_COORD_RE.match(s) or _JUNK_PUNCT_RE.match(s)
    )


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
      5. 孤儿行若是解析失败碎片（残缺标签、坐标数组、纯标点）整行丢弃。

    参数:
        raw_text: 模型原始输出（含 <BLOCK> 标签）。

    返回:
        清理后的文本。正常页面应逐字节不变（对合法文本零影响）。
    """
    # v2 格式：<BLOCK>type[<LEVELn>][x1,y1,x2,y2]<FORK>文本
    det_re = re.compile(r'<BLOCK>\s*([A-Za-z_][\w-]*)\s*(?:<LEVEL\d+>)?\s*(\[[^\]]+\])\s*<FORK>\s*(.*)')
    parsed = []
    for ln in raw_text.split('\n'):
        m = det_re.match(ln)
        if m:
            # group(2) 形如 "[918, 28, 928, 38]"；模型偶发输出全角逗号/异常
            # 坐标（如 "[884， 239, ...]"），统一用正则提取整数；
            # 提取不完整时按 -1 处理（不参与越界判断）
            nums = [int(x) for x in re.findall(r'-?\d+', m.group(2))]
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

        # 规则5：孤儿行若是块解析失败碎片（残缺标签/坐标数组/纯标点）→ 丢弃
        if p['orphan']:
            # 先剥离行首结构前缀（<PAGE>/<MERGE>）与重复 <BLOCK>，再判断：
            #   <PAGE><BLOCK>header[...]   → 剥后能完整解析 → 保留
            #   <BLOCK><BLOCK>text[...]    → 剥后能完整解析 → 保留
            #   <MERGE> 单独成行           → 剥后为空 → 保留（跨页合并语义）
            probe = re.sub(r'^(?:<PAGE>|<MERGE>)\s*', '', t)
            if probe.startswith('<BLOCK>'):
                probe = re.sub(r'^(?:<BLOCK>\s*)+', '<BLOCK>', probe)
            if not det_re.match(probe) and (
                probe.startswith('<BLOCK>') or _is_junk_fragment(probe)
            ):
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

    return '\n'.join(_cut_near_loop(out))


def _cut_near_loop(lines: list[str], size: int = 5, mod: float = 0.75,
                   min_rows: int = 25) -> list[str]:
    """截断整页近似重复（同一词组覆盖大多数行）。

    模型对读不出的页面会反复套同一句式，每行只微调几个字：
        本调查委员会将根据本调查结果，并将本调查委员会提交给本调查委员会。
        本调查委员会的结论将於本调查报告上发表，并将於本调查委员会上发表。
    ……每行不完全相同，行级去重删不掉，但从页级看，某个 5 字词组会覆盖
    绝大多数行。此函数从尾部向前收缩，找出这样的循环起点并截断。

    正常文档不会出现“同一 5 字词组覆盖 >=75% 的行”（标题/表格重复也不至于），
    因此对正常页面零影响。

    Args:
        lines: 已过行级去重的输出行（含 <BLOCK> 标签或正文）。
        size:  n-gram 长度。
        mod:   覆盖率阈值（同一 n-gram 出现在该比例以上的行即判循环）。
        min_rows: 至少多少行文本才启用（避免短页误判）。

    Returns:
        截断后的行列表（未检测到循环时原样返回）。
    """
    texts: list[str | None] = []
    for ln in lines:
        if '<FORK>' in ln:
            txt = re.sub(r'\s+', '', ln.split('<FORK>')[-1])
            texts.append(txt if len(txt) >= 8 else None)
        else:
            texts.append(None)
    idx = [i for i, t in enumerate(texts) if t]
    if len(idx) < min_rows:
        return lines
    rows = [texts[i] or "" for i in idx]

    # 从前往后找最早一个“尾部仍覆盖超标”的起点
    for s in range(len(rows) - min_rows + 1):
        sub = rows[s:]
        gram: dict[str, int] = {}
        for t in sub:
            for g in {t[k:k + size] for k in range(len(t) - size + 1)}:
                gram[g] = gram.get(g, 0) + 1
        if gram and max(gram.values()) >= mod * len(sub):
            logger.info(
                "检测到整页近似重复循环（%d 行起，共 %d 行）：已截断，保留前 %d 行内容",
                s, len(sub), idx[s],
            )
            return lines[:idx[s]]
    return lines


def _box_iou(a, b) -> float:
    """
    计算两个 [x1, y1, x2, y2] 矩形框的交并比（IoU）。

    坐标可以是任意同一量纲（本模块传 [0,999] 归一化坐标）。
    坐标长度不足 4 或面积为零时返回 0.0（不参与去重判断）。

    Args:
        a: 矩形 A 坐标序列
        b: 矩形 B 坐标序列

    Returns:
        0.0–1.0 的 IoU 值。
    """
    try:
        ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    except (TypeError, ValueError, IndexError):
        return 0.0

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def process_raw_output(raw_text: str, original_image_path: str, req_id: str) -> str:
    """
    将 v2 模型返回的 raw 文本处理为完整 Markdown。

    Unlimited-OCR-2 原始输出格式：
        <PAGE><BLOCK>header[76, 137, 183, 179]<FORK>"All the News..."
        <BLOCK>title<LEVEL1>[58, 262, 908, 362]<FORK>MEN WALK ON MOON
        <BLOCK>image[272, 551, 692, 956]<FORK>
        <MERGE>
        <BLOCK>text[699, 673, 911, 699]<FORK>HOUSTON, Monday, July 21—...

    标签含义：
        <BLOCK>type[<LEVELn>][x1, y1, x2, y2]<FORK>正文
        <PAGE>   分页符（batch 模式已按页拆分，这里仅作清理）
        <MERGE>  跨页合并标记：紧随其后的块与上一块属于同一段落

    类型映射（与 v1 保持一致）：
        header → #；title → ##（<LEVEL2> → ###）；subtitle → ###
        text / image_caption → 正文
        image → 按坐标裁剪，输出 ![](images/{req_id}/N.jpg)
        footer / page_number → 跳过

    坐标格式 [x1, y1, x2, y2] 归一化到 [0, 999]，按实际图片尺寸映射到像素坐标。

    alt 文本策略：
        记录最近遇到的 header/title/subtitle，后续 image 标签的 alt 自动使用该标题。
        若无标题，则降级为 "image"。

    Args:
        raw_text:            模型返回的原始文本
        original_image_path: 原始输入图片的路径（用于裁剪提取子图）
        req_id:              本次请求的唯一标识，用于图片子目录命名

    Returns:
        格式化后的 Markdown 字符串（含标题层级、图片引用和 alt 文本）
    """
    # 延迟导入，避免循环依赖
    from PIL import Image as PILImage

    # ── 第1步：去除 EOS 终止标记 ──
    stop_str = '<｜end▁of▁sentence｜>'
    if raw_text.endswith(stop_str):
        raw_text = raw_text[:-len(stop_str)]
    raw_text = raw_text.strip()

    # ── 第1.5步：幻觉护栏（后处理过滤循环/无中生有行） ──
    raw_text = strip_hallucinations(raw_text)

    # ── 第2步：加载原始图片（用于后续按坐标裁剪） ──
    # 图片宽高用于将 [0,999] 归一化坐标映射到实际像素坐标
    try:
        orig_img = PILImage.open(original_image_path)
        img_w, img_h = orig_img.size
    except Exception as e:
        logger.warning(
            "裁剪源图片打开失败: %s 异常=%s: %s",
            original_image_path, type(e).__name__, e,
        )
        orig_img = None
        img_w, img_h = 1, 1

    # ── 第3步：准备图片输出目录 ──
    img_dir = IMAGES_DIR / req_id
    img_dir.mkdir(parents=True, exist_ok=True)

    # 检测列表项/独立条目: 以数字序号、破折号、圆点等开头
    _listish_pattern = re.compile(r'^(\d{1,2}\s|[•\-\*\–\—]\s|\d+\.\s)')
    # v2 块格式：<BLOCK>type[<LEVELn>][x1,y1,x2,y2]<FORK>正文
    block_re = re.compile(
        r'<BLOCK>\s*([A-Za-z_][\w-]*)\s*(?:<LEVEL(\d+)>)?\s*(\[[^\]]+\])\s*<FORK>\s*(.*)'
    )

    # ── 第4步：逐行解析并构建 Markdown ──
    lines = raw_text.split('\n')
    result_lines = []         # 最终输出的 Markdown 行列表
    last_heading = ""         # 记录最近遇到的标题文本，作为后续图片的 alt
    image_idx = 0             # 图片序号，用于文件命名（N.jpg）
    seen_image_boxes = []     # 本页已输出图片的坐标，用于重复图片去重
    prev_type = ""            # 上一块类型
    prev_is_listish_flag = False  # 上一行是否为列表项（用于连续列表项合并）
    merge_next = False        # <MERGE>：下一块与上一块合并（不加空行）

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # <MERGE>：跨页合并标记（可能单独一行，也可能带后续块）
        if stripped.startswith('<MERGE>'):
            merge_next = True
            stripped = stripped[len('<MERGE>'):].strip()
            if not stripped:
                continue

        # <PAGE>：单图/未拆分输出中的分页前缀
        if stripped.startswith('<PAGE>'):
            stripped = stripped[len('<PAGE>'):].strip()
            if not stripped:
                continue

        # 模型偶发重复输出块前缀（如 <BLOCK><BLOCK>text[...]），合并为单个
        if stripped.startswith('<BLOCK>'):
            stripped = re.sub(r'^<BLOCK>\s*(?:<BLOCK>\s*)+', '<BLOCK>', stripped)

        m = block_re.search(stripped)

        if not m:
            # ── 无块标签的行（孤立行） ──
            line_clean = stripped
            # 丢弃标签/坐标/标点碎片行（模型格式崩溃产物，不是正文）
            if line_clean and _is_junk_fragment(line_clean):
                continue
            if line_clean:
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
                merge_next = False
            continue

        # ── 解析块标签 ──
        det_type = m.group(1).strip()
        level = m.group(2)
        coords_str = m.group(3)
        text_after = m.group(4).strip()

        mapping = DET_TYPE_MAP.get(det_type)
        if mapping is None:
            # 未知类型 → 视为 text
            if text_after:
                # v2 每个 <BLOCK> 是独立段落，段落之间需要空行
                # （标准 Markdown 中单个换行只是软换行，渲染时会并成同一段）。
                if not merge_next and prev_type not in ("",) and not prev_is_listish_flag:
                    result_lines.append("")
                result_lines.append(text_after)
                prev_type = "text"
                prev_is_listish_flag = False
            merge_next = False
            continue

        prefix, keep_text = mapping
        # 标题层级微调：v2 用 <LEVEL1>/<LEVEL2> 标层级，LEVEL2 降为三级标题
        if det_type == "title" and level == "2":
            prefix = "### "

        if det_type == "image":
            # ── 图片处理（与 v1 相同的裁剪/alt/多子图逻辑） ──
            if prev_type not in ("", "image"):
                result_lines.append("")
            try:
                # 兼容模型偶发的全角逗号坐标（如 "[884， 239, ...]"）
                coords = eval(coords_str.replace('，', ','))
                # 如果是单个坐标组 [100, 50, 300, 200]，包装为 [[100, 50, 300, 200]]
                if coords and isinstance(coords[0], (int, float)):
                    coords = [coords]
            except Exception as e:
                logger.warning(
                    "图片坐标解析失败: req_id=%s 坐标=%s 异常=%s: %s",
                    req_id, coords_str, type(e).__name__, e,
                )
                coords = []

            # 记录实际保存成功的子图文件名（裁剪失败的图片不会生成链接）
            saved_files: list[str] = []
            dup_skipped = 0  # 因重复被跳过的子图数（用于区分日志级别）

            for ci, c in enumerate(coords):
                # ── 重复图片去重 ──
                # v2 模型在双栏/大图版面偶发把同一张图输出两次（坐标几乎相同，
                # 两次之间还可能夹着正文），不去重会得到两张一模一样的图片。
                if IMAGE_DEDUP_IOU > 0 and any(
                    _box_iou(c, b) >= IMAGE_DEDUP_IOU for b in seen_image_boxes
                ):
                    logger.info(
                        "重复图片跳过: req_id=%s 坐标=%s（与已输出图片 IoU ≥ %.2f）",
                        req_id, c, IMAGE_DEDUP_IOU,
                    )
                    dup_skipped += 1
                    continue

                fname = f"{image_idx}{('_' + str(ci)) if len(coords) > 1 else ''}.jpg"
                try:
                    # 将 [0,999] 归一化坐标映射到实际像素坐标
                    x1 = int(c[0] / 999 * img_w)
                    y1 = int(c[1] / 999 * img_h)
                    x2 = int(c[2] / 999 * img_w)
                    y2 = int(c[3] / 999 * img_h)

                    if orig_img is not None and x2 > x1 and y2 > y1:
                        cropped = orig_img.crop((x1, y1, x2, y2))
                        cropped.save(str(img_dir / fname))
                        saved_files.append(fname)
                        # 只记录实际输出到 Markdown 的图片
                        seen_image_boxes.append(tuple(c))
                    else:
                        logger.warning(
                            "图片裁剪跳过: req_id=%s 文件=%s 坐标=%s 原图=%s 原因=%s",
                            req_id, fname, c,
                            "已加载" if orig_img is not None else "缺失",
                            "坐标无效(x2<=x1 或 y2<=y1)" if x2 <= x1 or y2 <= y1 else "未知",
                        )
                except Exception as e:
                    logger.warning(
                        "图片裁剪异常: req_id=%s 文件=%s 坐标=%s 原图=%s 异常=%s: %s",
                        req_id, fname, c, original_image_path, type(e).__name__, e,
                    )
                    continue

            # 只有实际保存成功的图片才生成 Markdown 链接（避免死链 404）
            if not saved_files:
                if dup_skipped == len(coords):
                    # 整块都是重复图 → 去重的正常结果，不是异常
                    logger.info(
                        "整块图片重复，已跳过: req_id=%s 坐标=%s",
                        req_id, coords_str,
                    )
                else:
                    logger.warning(
                        "图片裁剪全部失败，跳过输出: req_id=%s 坐标=%s",
                        req_id, coords_str,
                    )
                merge_next = False
                continue

            # alt 文本：使用最近遇到的标题，无标题则降级为 "image"
            alt = last_heading if last_heading else "image"
            for ci, fname in enumerate(saved_files):
                if len(saved_files) == 1:
                    result_lines.append(f"![{alt}](images/{req_id}/{fname})")
                else:
                    result_lines.append(f"![{alt} ({ci+1})](images/{req_id}/{fname})")

            image_idx += 1
            prev_type = "image"
            prev_is_listish_flag = False
            merge_next = False
            continue

        if keep_text:
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
                    prev_is_listish_flag = False
                else:
                    # 正文段落（text / image_caption）
                    cur_is_listish = bool(_listish_pattern.match(text_after))
                    prev_was_listish = bool(prev_type == "text" and prev_is_listish_flag)
                    if cur_is_listish:
                        if not prev_was_listish:
                            result_lines.append("")  # 普通文本 → 列表，加空行
                        result_lines.append("- " + text_after)
                        prev_is_listish_flag = True
                    else:
                        # v2 每个 <BLOCK>text 是一个独立段落，段间必须用空行分隔，
                        # 否则标准 Markdown 渲染时会被合并成同一段（单换行不算换段）。
                        # 例外：<MERGE> 跨页续写、连续列表项不加空行。
                        if not merge_next and prev_type not in ("",) and not prev_was_listish:
                            result_lines.append("")
                        result_lines.append(md_line)
                        prev_is_listish_flag = False
                    prev_type = "text"
                merge_next = False

        # 注意：非 image 且 keep_text=False 的类型（page_number / footer）
        # 直接跳过，不输出任何内容

    # ── 第5步：清理并返回 ──
    result = '\n'.join(result_lines)
    # 清理残留的分页/合并标记（跨页表格等场景可能未完全消解）
    result = result.replace('<MERGE>', '').replace('<PAGE>', '')
    # 替换 LaTeX 数学模式转义字符为标准符号
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
        # 下游引擎空闲超时后被停止；该值保留字段兼容
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

    停止下游 llama-server 并释放显存；下次请求到达时自动重新拉起。
    持 _infer_lock 执行：若此刻有推理在跑则等待其完成后再停止，
    避免打断进行中的请求。
    """
    with _infer_lock:
        _stop_engine()
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
            # PDF 转图 DPI：实测 150 与 300 DPI 的 OCR 结果一致（相似度 99.4%，
            # 字符数差 1 个）——模型内部会把整页缩到固定 tile 网格（A4 约 448×672，
            # 共 6 块），分辨率超过网格上限后的细节会被丢弃，150 DPI 已足够且
            # 转图/解码开销更小。若后续遇到极小字（如批注、脚注）丢失可调回 300。
            pdf_dpi = 150
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

                # 单图请求 → llama-server 使用模型默认的完整视图模式
                    raw_text = llama_chat(prompt, [img_path])
                    page_md = process_raw_output(raw_text, str(orig_copy), f"{req_id}/page_{page_idx}")
                    if page_md and page_md != "[empty]":
                        all_processed.append(page_md)

                text = "\n\n---\n\n".join(all_processed) if all_processed else "[empty]"

            elif use_batch:
                # ═══════════════════════════════════════════════════════
                # 批量模式: 多张图一次请求（等价原 infer_multi base 1024）
                # 输出以 <PAGE> 分隔；每张图约 273 image token（base 模式）。
                # 每批图片数受 LLAMA_MAX_IMAGES_PER_REQ 限制（引擎 context 上限）。
                # ═══════════════════════════════════════════════════════
                total_pages = len(imgs)
                batch_size = min(MAX_PAGES_PER_BATCH, LLAMA_MAX_IMAGES_PER_REQ)
                all_processed = []
                global_page_idx = 0

                for batch_start in range(0, total_pages, batch_size):
                    batch_end = min(batch_start + batch_size, total_pages)
                    batch_imgs = imgs[batch_start:batch_end]

                    raw_output = llama_chat(prompt, batch_imgs)

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
