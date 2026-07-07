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
import sys
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
import warnings
from pathlib import Path
from typing import Optional

# 屏蔽 Transformers generate() 内部关于 attention_mask/pad_token_id 的冗余警告
# 模型使用 images_seq_mask 管理视觉 token 的注意力，不需要标准 attention_mask
# 实测不影响输出质量，仅减少日志噪音
warnings.filterwarnings("ignore", message=".*attention_mask.*")
warnings.filterwarnings("ignore", message=".*pad_token_id.*")

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

# PDF/多图分批处理：每批最多处理多少页
# 20页 × 257 token/页 ≈ 5140 image tokens，留足文本输出空间，避免 OOM
MAX_PAGES_PER_BATCH = int(os.environ.get("MAX_PAGES_PER_BATCH", "20"))

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
                        # 本地绝对路径
                        fpath = url
                    if fpath:
                        temps.append(fpath)
                        imgs.append(fpath)

    # 组装 prompt
    # 多图时使用 "Multi page parsing." 提示词，单图时使用 "Free OCR."
    if len(imgs) > 1:
        default_prompt = "<image>\nMulti page parsing."
    else:
        default_prompt = "<image>\nFree OCR."

    prompt = "\n".join(parts) if parts else default_prompt
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    return prompt, imgs, temps


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

    # ── 第2步：加载原始图片（用于后续按坐标裁剪） ──
    # 图片宽高用于将 [0,999] 归一化坐标映射到实际像素坐标
    # 例如：坐标 [100, 50, 300, 200] 在 800×600 图片上对应像素区域 (80, 30, 240, 120)
    try:
        orig_img = PILImage.open(original_image_path)
        img_w, img_h = orig_img.size
    except Exception:
        # 如果原图加载失败，后面裁剪会跳过，但不阻塞 Markdown 文本生成
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
            except Exception:
                # 坐标解析失败（格式异常），跳过此图片
                coords = []

            # 遍历可能的多组坐标（一个 image 区域可能包含多张子图）
            for ci, c in enumerate(coords):
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
                        cropped.save(str(img_dir / f"{image_idx}{suffix}.jpg"))
                except Exception:
                    # 单张图片裁剪失败不影响其他图片
                    continue

            # ── 生成 alt 文本 ──
            # 策略：使用最近遇到的标题（由 header/title/subtitle 标签记录）
            # 如果前面没有标题，降级为 "image"
            alt = last_heading if last_heading else "image"

            # ── 构建 Markdown 图片引用 ──
            # 单图: ![标题文本](images/req_id/0.jpg)
            # 多图: ![标题文本 (1)](images/req_id/0_0.jpg)
            #       ![标题文本 (2)](images/req_id/0_1.jpg)
            if len(coords) == 1:
                result_lines.append(
                    f"![{alt}](images/{req_id}/{image_idx}.jpg)"
                )
            else:
                for ci in range(len(coords)):
                    result_lines.append(
                        f"![{alt} ({ci+1})](images/{req_id}/{image_idx}_{ci}.jpg)"
                    )

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

    支持单图和多图两种模式:
    - 单图 (1 张 image_url): 使用 gundam 模式（crop_mode=True, image_size=640）
    - 多图 (≥2 张 image_url): 使用 base 模式（image_size=1024），一次推理处理所有页
      输出页间用 <PAGE> 分隔，每页独立提取图片和 Markdown 结构

    也支持 PDF 文件: 传入 .pdf 文件路径，服务端自动转换为多图后推理。

    单 GPU 串行执行，同一时间只处理一个请求。
    """
    import io as _io

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
                prompt = "<image>\nMulti page parsing."

        # ── 应用 max_pages 限制（仅对多图/PDF 生效） ──
        if req.max_pages is not None and req.max_pages > 0 and len(imgs) > req.max_pages:
            imgs = imgs[:req.max_pages]

        # ── 第2步：按需加载模型 ──
        _ensure_loaded()
        _touch_used()

        req_id = uuid.uuid4().hex[:12]

        with _infer_lock:
            if _model is None:
                return error_response(503, "模型未加载", "server_error")

            # 决定使用哪种推理路径
            use_batch = (len(imgs) > 1 and req.page_mode == "batch")

            if len(imgs) == 1 or req.page_mode == "single":
                # ═══════════════════════════════════════════════════════
                # 逐张模式: model.infer() gundam 640 crop_mode
                # 高质量、快速。多图时逐张循环处理，结果用 --- 合并。
                # ═══════════════════════════════════════════════════════
                all_processed = []
                for page_idx, img_path in enumerate(imgs):
                    orig_copy = IMAGES_DIR / req_id / f"_page_{page_idx}.png"
                    orig_copy.parent.mkdir(parents=True, exist_ok=True)
                    from PIL import Image as PILImage
                    PILImage.open(img_path).save(str(orig_copy))

                    raw_text = _model.infer(
                        _tokenizer,
                        prompt=prompt,
                        image_file=img_path,
                        output_path='/tmp/unused',
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        max_length=32768,
                        no_repeat_ngram_size=35,
                        ngram_window=128,
                        save_results=False,
                        eval_mode=True,
                    )
                    page_md = process_raw_output(raw_text, str(orig_copy), f"{req_id}/page_{page_idx}")
                    if page_md and page_md != "[empty]":
                        all_processed.append(page_md)

                text = "\n\n---\n\n".join(all_processed) if all_processed else "[empty]"

            elif use_batch:
                # ═══════════════════════════════════════════════════════
                # 批量模式: model.infer_multi() base 1024 跨页连贯
                # ═══════════════════════════════════════════════════════
                total_pages = len(imgs)
                batch_size = MAX_PAGES_PER_BATCH
                all_processed = []
                global_page_idx = 0

                for batch_start in range(0, total_pages, batch_size):
                    batch_end = min(batch_start + batch_size, total_pages)
                    batch_imgs = imgs[batch_start:batch_end]

                    old_stdout = sys.stdout
                    sys.stdout = _io.StringIO()
                    try:
                        raw_output, _ = _model.infer_multi(
                            _tokenizer,
                            prompt=prompt,
                            image_files=batch_imgs,
                            output_path='/tmp/unused',
                            image_size=1024,
                            max_length=32768,
                            no_repeat_ngram_size=35,
                            ngram_window=1024,
                            save_results=False,
                        )
                    finally:
                        sys.stdout = old_stdout

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
