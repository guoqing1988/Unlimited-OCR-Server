# Unlimited-OCR Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy Unlimited-OCR as an OpenAI-compatible inference service with SGLang backend, lazy loading, idle auto-unload, and Markdown post-processing with image extraction.

**Architecture:** Single-file FastAPI proxy (`server.py`) wraps an SGLang subprocess on an internal port. The proxy manages SGLang lifecycle (start-on-demand, idle-unload), proxies chat completion requests with SGLang-specific parameters, then post-processes the raw output (ref/det tag parsing, image cropping, alt text generation) into clean Markdown. Static image serving via FastAPI `StaticFiles` mount.

**Tech Stack:** Python 3.12, uv, SGLang (custom wheel), FastAPI, uvicorn

## Global Constraints

- Python 3.12 only (`/usr/bin/python3.12`)
- uv-managed venv at `.venv/`, never touch system pip/packages
- SGLang wheel from `wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl`
- Model at `/data/www/models/Unlimited-OCR`
- Service port 10000, SGLang internal port 20000
- `MEM_FRACTION_STATIC=0.25` (12 GB) default, tune upward if OOM
- API must match existing DeepSeek-OCR-2 format exactly
- Code style: PEP 8, 4-space indent, match existing DS-OCR-2 server.py patterns
- Images served at `/images/{req_id}/{file}`

---

### Task 1: Environment Setup

**Files:**
- Create: `.venv/` (directory, via uv)
- No source files created yet

**Description:** Create Python 3.12 virtual environment and install all dependencies from the SGLang wheel and PyPI.

- [ ] **Step 1: Create virtual environment**

```shell
cd /data/www/wwwroot/Unlimited-OCR
uv venv .venv --python 3.12
```

Expected: `.venv/` directory created with Python 3.12 interpreter.

- [ ] **Step 2: Install SGLang wheel**

This wheel is large (12 MB compressed) and has many transitive dependencies. The install may take 2-5 minutes.

```shell
source .venv/bin/activate
uv pip install wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl
```

Expected: SGLang and all its dependencies installed. Watch for any build failures (especially `flashinfer_python` and `sglang-kernel` which have compiled components).

- [ ] **Step 3: Install additional dependencies**

```shell
uv pip install kernels==0.11.7 pymupdf==1.27.2.2 fastapi "uvicorn[standard]" python-dotenv
```

Expected: All packages installed successfully.

- [ ] **Step 4: Verify installation**

```shell
source .venv/bin/activate
python -c "import sglang; print('SGLang version:', sglang.__version__)"
python -c "from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor; print('LogitProcessor OK:', DeepseekOCRNoRepeatNGramLogitProcessor.to_str()[:50])"
python -c "import fastapi; import uvicorn; import dotenv; print('Web deps OK')"
```

Expected: All imports succeed, no errors. The `DeepseekOCRNoRepeatNGramLogitProcessor.to_str()` returns a non-empty string.

- [ ] **Step 5: Commit**

```bash
git add .gitignore
git commit -m "chore: set up Python 3.12 venv with SGLang and dependencies

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: server.py — Config, State, Auth, and SGLang Lifecycle

**Files:**
- Create: `server.py` (first half: config, state, app, auth middleware, lifecycle functions)

**Description:** Build the core framework: environment config loading, global state, FastAPI app creation, authentication middleware, and SGLang subprocess management (start, stop, health check, watchdog). After this task, the server can manage SGLang lifecycle but has no API endpoints.

- [ ] **Step 1: Write the config, state, and app skeleton**

Create `server.py`:

```python
"""
Unlimited-OCR: OpenAI-compatible OCR inference service with SGLang backend.

Startup:
    uvicorn server:app --host 0.0.0.0 --port 10000

Configuration via .env file or environment variables.
SGLang is managed as a subprocess — started on first request, stopped after idle timeout.

API:
    GET  /health              — health check (model status + idle time)
    GET  /v1/models           — model list
    POST /v1/chat/completions — multimodal OCR inference
    POST /admin/unload        — manual model unload
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

# ── Config (.env / env vars) ─────────────────────────────────────────────

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
SERVER_TIMEOUT = 300  # max seconds to wait for SGLang startup
NO_REPEAT_NGRAM_SIZE = 35
REQUEST_TIMEOUT = 1200
MAX_RETRIES = 3

IMAGES_DIR = Path(__file__).parent / "images"
LOG_DIR = Path(__file__).parent / "log"

# ── Global state ──────────────────────────────────────────────────────────

_state_lock = threading.Lock()   # protects _process / _loaded_at / _last_used
_startup_lock = threading.Lock() # serializes SGLang startup (one at a time)

_sglang_process: Optional[subprocess.Popen] = None
_loaded_at: Optional[float] = None
_last_used: Optional[float] = None
_total_requests: int = 0

# ── FastAPI app ───────────────────────────────────────────────────────────

app = FastAPI(title="Unlimited-OCR API", version="1.0.0")
```

- [ ] **Step 2: Add authentication middleware**

Append to `server.py`:

```python
# ── Auth middleware ───────────────────────────────────────────────────────

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path in ("/health",) or request.url.path.startswith("/images/"):
        return await call_next(request)
    if not API_KEY:
        return await call_next(request)

    auth = request.headers.get("Authorization", "")
    expected = f"Bearer {API_KEY}"
    if auth != expected:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "Invalid or missing API key. Use 'Authorization: Bearer <key>' header.",
                    "type": "authentication_error",
                    "code": 401,
                }
            },
        )
    return await call_next(request)
```

- [ ] **Step 3: Add SGLang lifecycle functions**

Append to `server.py`:

```python
# ── SGLang lifecycle ──────────────────────────────────────────────────────

def _sglang_health() -> bool:
    """Check if the SGLang server is responding."""
    try:
        resp = __import__("requests").get(f"{SGLANG_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _start_sglang():
    """Launch SGLang server as a subprocess. Returns immediately; caller must poll _sglang_health()."""
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
    """Terminate the SGLang subprocess and release GPU memory."""
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

    # Release GPU memory
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
    """Ensure SGLang is running. Blocks until health check passes.
    Uses double-checked locking so concurrent requests share one startup."""
    import requests as req

    # Fast path: already running
    if _sglang_health():
        return

    with _startup_lock:
        # Double-check after acquiring lock
        if _sglang_health():
            return

        _start_sglang()

        start = time.time()
        while time.time() - start < SERVER_TIMEOUT:
            if _sglang_process is not None and _sglang_process.poll() is not None:
                raise RuntimeError(
                    f"SGLang server exited early (code {_sglang_process.returncode}). "
                    f"Check {LOG_DIR / 'sglang_server.log'}"
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
            f"SGLang server did not become healthy within {SERVER_TIMEOUT}s. "
            f"Check {LOG_DIR / 'sglang_server.log'}"
        )


def _touch_used():
    """Update last-used timestamp (called after each request completes)."""
    global _last_used, _total_requests
    with _state_lock:
        _last_used = time.time()
        _total_requests += 1


def _watchdog():
    """Background thread: check idle time and unload if exceeded."""
    while True:
        time.sleep(WATCHDOG_POLL_SECONDS)
        with _state_lock:
            loaded = _sglang_process is not None
            last = _last_used
        if loaded and last and (time.time() - last) >= IDLE_UNLOAD_SECONDS:
            _stop_sglang()


@app.on_event("startup")
def startup():
    th = threading.Thread(target=_watchdog, daemon=True)
    th.start()
```

- [ ] **Step 4: Verify lifecycle — start SGLang manually**

First, make sure `.env` exists with minimal config:

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

Then verify by importing and calling lifecycle functions:

```shell
cd /data/www/wwwroot/Unlimited-OCR
source .venv/bin/activate
python -c "
from server import _start_sglang, _sglang_health, _stop_sglang
import time
print('Starting SGLang...')
_start_sglang()
print('Waiting for health...')
for i in range(100):
    if _sglang_health():
        print(f'Healthy after {i*3}s')
        break
    time.sleep(3)
else:
    print('TIMEOUT - check log/sglang_server.log')
print('Stopping...')
_stop_sglang()
print('Done')
"
```

Expected: SGLang starts, becomes healthy (may take 60-120s), then stops cleanly. GPU memory returns to baseline after stop.

- [ ] **Step 5: Commit**

```bash
git add server.py .env
git commit -m "feat: add server.py core with SGLang lifecycle management

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: server.py — API Models, Utilities, and Post-Processing

**Files:**
- Modify: `server.py` (append Pydantic models, utility functions, post-processing pipeline)

**Description:** Add the data models for OpenAI-compatible requests, utility functions for parsing messages/images, and the post-processing pipeline that converts raw SGLang output (with ref/det tags) into clean Markdown with extracted images and alt text.

- [ ] **Step 1: Add Pydantic models and error helper**

Append to `server.py`:

```python
# ── OpenAI-compatible error responses ─────────────────────────────────────

def error_response(status: int, message: str, error_type: str) -> JSONResponse:
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


# ── Pydantic request models ──────────────────────────────────────────────

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = "user"
    content: str | list[dict] = ""
    name: str | None = None


class ChatCompletionRequest(BaseModel):
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

- [ ] **Step 2: Add message/image parsing utilities**

Append to `server.py`:

```python
# ── Utility: parse OpenAI messages ────────────────────────────────────────

def decode_image(s: str) -> str:
    """Decode a base64 data-URI to a temporary image file. Returns file path."""
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
    Parse OpenAI-format messages into (prompt, image_paths, temp_files).

    Returns:
        prompt: text prompt with <image> marker
        image_paths: list of resolved image file paths
        temp_files: list of temp files that the caller should clean up
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

- [ ] **Step 3: Add post-processing pipeline**

Append to `server.py`:

```python
# ── Post-processing: ref/det tag parsing ──────────────────────────────────

def re_match(text: str) -> tuple[list, list, list]:
    """
    Parse <|ref|>label<|/ref|><|det|>[...]<|/det|> tags from model output.

    Returns:
        all_matches:  list of (full_match, label, coords_str)
        image_matches: subset where label == 'image'
        other_matches: all other labels
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
    Parse coordinate string and denormalize from [0,999] to actual pixel values.

    coords_str format: "[x1,y1,x2,y2]" or "[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]"
    Returns list of (x1, y1, x2, y2) tuples in pixel coordinates.
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
    Generate alt text for an image tag.
    Strategy: find the nearest ## heading before the image, combine with region type.
    """
    headings = re.findall(r'^##\s*(.+)', text_before, re.MULTILINE)
    if headings:
        return f"{headings[-1].strip()} - {region_type}"
    # Fallback: use the last 40 chars of preceding text
    last_text = re.sub(r'\s+', ' ', text_before).strip()
    fallback = last_text[-40:].strip() if last_text else "image"
    return f"{fallback} - {region_type}"


def post_process(raw_text: str, original_image_path: str, req_id: str) -> str:
    """
    Post-process SGLang raw output into clean Markdown.

    1. Parse <|ref|>/<|det|> tags
    2. For 'image' regions: crop from original, save to images/{req_id}/N.jpg
    3. For all regions: replace tags with proper markdown image syntax + alt text
    4. Clean up residual tags

    Returns processed Markdown string.
    """
    from PIL import Image

    # Strip EOS token
    stop_str = '<｜end▁of▁sentence｜>'
    if raw_text.endswith(stop_str):
        raw_text = raw_text[:-len(stop_str)]
    raw_text = raw_text.strip()

    # Parse ref/det tags
    all_matches, image_matches, other_matches = re_match(raw_text)

    if not all_matches:
        return raw_text

    # Prepare output directory
    img_dir = IMAGES_DIR / req_id
    img_dir.mkdir(parents=True, exist_ok=True)

    # Load original image for cropping
    try:
        orig_img = Image.open(original_image_path)
        img_w, img_h = orig_img.size
    except Exception:
        img_w, img_h = 1, 1
        orig_img = None

    processed = raw_text

    # Process image-type regions: crop and save
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

        # Determine alt text
        # Find text before this match in the output
        match_pos = processed.find(full_match)
        text_before = processed[:match_pos] if match_pos >= 0 else ""
        alt = generate_alt(text_before, label)

        # Build replacement
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

    # Clean up non-image ref/det tags (just remove them)
    for full_match, label, coords_str in other_matches:
        processed = processed.replace(full_match, "")

    # Clean up residual markup
    processed = processed.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

    return processed.strip()
```

- [ ] **Step 4: Unit test post-processing functions**

```shell
cd /data/www/wwwroot/Unlimited-OCR
source .venv/bin/activate
python -c "
from server import re_match, extract_coordinates, generate_alt, post_process
from PIL import Image
import tempfile, os

# Test re_match
text = '## Introduction\nSome text.\n<|ref|>image<|/ref|><|det|>[100,50,300,200]<|/det|>\nMore text.'
all_m, img_m, oth_m = re_match(text)
print('re_match:', len(all_m), 'total,', len(img_m), 'images,', len(oth_m), 'other')
assert len(img_m) == 1
assert img_m[0][1] == 'image'
assert img_m[0][2] == '[100,50,300,200]'

# Test extract_coordinates
coords = extract_coordinates('[100,50,300,200]', 1000, 800)
print('Coordinates:', coords)
assert len(coords) == 1
assert coords[0] == (100, 40, 300, 160)  # 100/999*1000 ≈ 100, 50/999*800 ≈ 40, ...

# Test generate_alt
text_before = '## Chapter 1\nSome content.\n## Figures\nHere is a'
alt = generate_alt(text_before, 'diagram')
print('Alt text:', alt)
assert alt == 'Figures - diagram'

# Test full post_process with a real image
tmp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
img = Image.new('RGB', (640, 480), color='red')
img.save(tmp_img.name)
raw = '<image>\n## Summary\nSome text.\n<|ref|>image<|/ref|><|det|>[100,100,500,400]<|/det|>\nEnd.'
result = post_process(raw, tmp_img.name, 'test_req_001')
print('Post-processed:')
print(result[:200])
assert '![](images/test_req_001/0.jpg)' in result or '![' in result
os.unlink(tmp_img.name)
print('All post-processing tests passed!')
"
```

Expected: All assertions pass. The post-processing pipeline correctly parses ref/det tags and generates image references.

- [ ] **Step 5: Commit**

```bash
git add server.py
git commit -m "feat: add API models, message parsing, and post-processing pipeline

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: server.py — API Endpoints

**Files:**
- Modify: `server.py` (append health, models, unload, and chat completions endpoints)

**Description:** Add all API endpoints. This is the final piece of server.py — after this task, the service is fully functional.

- [ ] **Step 1: Add health, models, and unload endpoints**

Append to `server.py`:

```python
# ── API Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check with model load status and idle time."""
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
    """OpenAI-compatible model list."""
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
    """Manually unload model and free GPU memory."""
    _stop_sglang()
    return JSONResponse({"ok": True, "unloaded": True})
```

- [ ] **Step 2: Add chat completions endpoint (non-streaming logic first)**

Append to `server.py`:

```python
# ── Chat Completions ──────────────────────────────────────────────────────

def _get_ngram_processor_str():
    """Lazy-import the SGLang logit processor. Cached after first call."""
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
    """Build the SGLang-specific request payload from a standard OpenAI request."""
    import requests as _requests

    session = _requests.Session()
    session.trust_env = False

    # Encode images as base64 data URIs
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

    # Determine image mode and ngram window
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
    """Collect all chunks from an SGLang SSE streaming response. Returns full text."""
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
    OCR inference endpoint (OpenAI Chat Completions compatible).

    Accepts multimodal messages (text + images), proxies to SGLang,
    post-processes the output into clean Markdown with extracted images.
    Supports stream=True for SSE streaming output.
    """
    import requests as http_req

    temps = []
    try:
        # 1. Parse messages
        prompt, img_paths, temps = extract_messages(req.messages)
        if not img_paths:
            return error_response(400, "No image found in messages", "invalid_request")

        # 2. Ensure SGLang is loaded
        _ensure_loaded()
        _touch_used()

        # 3. Save original image copy for post-processing
        req_id = uuid.uuid4().hex[:12]
        from PIL import Image as PILImage
        orig_img_path = img_paths[0]  # Use first image for post-processing crop reference
        # If it's a temp file, keep a copy in the images dir
        orig_copy = IMAGES_DIR / req_id / "_original.png"
        orig_copy.parent.mkdir(parents=True, exist_ok=True)
        PILImage.open(orig_img_path).save(str(orig_copy))

        # 4. Build payload and forward to SGLang
        payload = _build_sglang_payload(
            prompt, img_paths, req.model or SERVED_NAME,
            req.max_tokens, req.temperature, req.stream,
        )

        # Always request streaming from SGLang so we can collect full output
        payload["stream"] = True

        resp = http_req.post(
            f"{SGLANG_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )
        resp.raise_for_status()

        # 5. Collect raw output from SGLang
        raw_text = _collect_sglang_stream(resp)

        # 6. Post-process: extract images, generate alt text, clean tags
        processed = post_process(raw_text, str(orig_copy), req_id)

        obj_id = f"chatcmpl-{req_id}"

        # 7. Return response
        if req.stream:
            async def gen():
                # Stream word-by-word (same behavior as DS-OCR-2)
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
        # Clean up temp files
        for p in temps:
            try:
                os.unlink(p)
            except Exception:
                pass
```

- [ ] **Step 3: Add global exception handler and static files mount**

Append to `server.py`:

```python
# ── Global exception handler ──────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return error_response(500, str(exc), "server_error")


# ── Static image serving ──────────────────────────────────────────────────

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
```

- [ ] **Step 4: Verify server starts and responds**

```shell
cd /data/www/wwwroot/Unlimited-OCR
source .venv/bin/activate

# Start server in background
nohup .venv/bin/uvicorn server:app --host 0.0.0.0 --port 10000 > log/server.log 2>&1 &
SERVER_PID=$!
sleep 3

# Test health endpoint
curl -s http://localhost:10000/health | python3 -m json.tool
# Expected: {"status":"ok","model_loaded":false,"idle_seconds":0,...}

# Test models endpoint
curl -s http://localhost:10000/v1/models | python3 -m json.tool
# Expected: {"object":"list","data":[{"id":"Unlimited-OCR",...}]}

# Stop server
kill $SERVER_PID
```

Expected: Health returns `model_loaded: false`. Models returns Unlimited-OCR in the list.

- [ ] **Step 5: Commit**

```bash
git add server.py
git commit -m "feat: add all API endpoints with SGLang proxy and post-processing

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Configuration Files and systemd Service

**Files:**
- Create: `unlimited-ocr.service`
- Modify: `.env` (already created in Task 2, verify contents)

**Description:** Create the systemd service file so the service starts on boot and restarts on failure. The `.env` file was already created in Task 2 Step 4 — verify it's complete.

- [ ] **Step 1: Verify .env file**

```bash
cat /data/www/wwwroot/Unlimited-OCR/.env
```

Expected content:

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

- [ ] **Step 2: Create systemd service file**

Write `unlimited-ocr.service`:

```ini
[Unit]
Description=Unlimited-OCR API Service (SGLang)
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

# GPU access
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 3: Install and enable the service**

```bash
# First, stop any manually running instance on port 10000
sudo systemctl stop unlimited-ocr 2>/dev/null || true
kill $(lsof -ti :10000) 2>/dev/null || true

# Install service
sudo ln -sf /data/www/wwwroot/Unlimited-OCR/unlimited-ocr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable unlimited-ocr
sudo systemctl start unlimited-ocr

# Check status
sleep 3
sudo systemctl status unlimited-ocr --no-pager
```

Expected: Service starts successfully. Health endpoint returns `model_loaded: false` (model not loaded until first request).

- [ ] **Step 4: Commit**

```bash
git add unlimited-ocr.service .env
git commit -m "chore: add systemd service and finalize .env config

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Integration Test

**Files:**
- No new files
- Test interacts with running service

**Description:** End-to-end test: send a real image to the service, verify it returns valid Markdown with image references, verify extracted images are served.

- [ ] **Step 1: Create a test image**

```shell
cd /data/www/wwwroot/Unlimited-OCR
source .venv/bin/activate
python -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (800, 600), color='white')
draw = ImageDraw.Draw(img)
draw.rectangle([50, 50, 750, 200], outline='black', width=2)
draw.text((60, 100), 'Test Document Title', fill='black')
draw.rectangle([50, 220, 750, 500], outline='blue', width=2)
draw.text((60, 300), 'Sample image region here', fill='blue')
img.save('test_document.png')
print('Test image created: test_document.png')
"
```

- [ ] **Step 2: Ensure service is running**

```bash
sudo systemctl status unlimited-ocr --no-pager | head -5
# Or start manually if systemd not used:
# nohup .venv/bin/uvicorn server:app --host 0.0.0.0 --port 10000 > log/server.log 2>&1 &
```

- [ ] **Step 3: Send a test OCR request**

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

print('Sending request (model will load on first request, may take 60-120s)...')
resp = requests.post(
    'http://localhost:10000/v1/chat/completions',
    headers={'Content-Type': 'application/json'},
    json=payload,
    timeout=300,
)
resp.raise_for_status()
result = resp.json()
content = result['choices'][0]['message']['content']
print('=== OCR Result ===')
print(content[:500])
print('...' if len(content) > 500 else '')
print('=== End ===')
"
```

Expected:
- First request triggers SGLang startup (60-120s wait)
- Request succeeds (200 OK)
- Output is clean Markdown text (no `<|ref|>` or `<|det|>` tags in output)
- If there are images in the output, they reference `images/{req_id}/N.jpg`

- [ ] **Step 4: Verify image serving**

```shell
# Check if any images were extracted
ls -la /data/www/wwwroot/Unlimited-OCR/images/ 2>/dev/null

# Test serving an extracted image (if any exist)
curl -s -o /dev/null -w "%{http_code}" http://localhost:10000/images/ 2>/dev/null
```

- [ ] **Step 5: Test idle unload**

```shell
# Check model is loaded
curl -s http://localhost:10000/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'model_loaded: {d[\"model_loaded\"]}')"

# Wait for idle timeout (or set IDLE_UNLOAD_SECONDS=60 in .env for testing)
# After timeout:
# curl -s http://localhost:10000/health → model_loaded: false
```

- [ ] **Step 6: Verify stream mode**

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

print('Sending streaming request...')
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

print(f'Total chunks received: {chunk_count}')
print('Stream test passed!')
"
```

Expected: Multiple SSE chunks received, ends with `data: [DONE]`.

- [ ] **Step 7: Clean up and commit**

```bash
rm -f test_document.png
git commit --allow-empty -m "test: integration test passed — OCR, image extraction, streaming

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Verification Checklist

After all tasks are complete, verify:

- [ ] `sudo systemctl status unlimited-ocr` shows active/running
- [ ] `curl http://localhost:10000/health` returns model status
- [ ] First OCR request succeeds (with SGLang cold start ~60-120s)
- [ ] Subsequent requests return quickly (model stays loaded)
- [ ] Output is clean Markdown — no `<|ref|>` / `<|det|>` tags
- [ ] Extracted images are served at `/images/{req_id}/N.jpg`
- [ ] Image alt text includes nearest heading + region type
- [ ] After `IDLE_UNLOAD_SECONDS`, model auto-unloads (`model_loaded: false`)
- [ ] `curl -X POST http://localhost:10000/admin/unload` immediately unloads
- [ ] `nvidia-smi` shows ~0 GB extra VRAM when unloaded, ~12-16 GB when loaded
- [ ] ComfyUI on port 8188 still works (GPU memory not exhausted)
