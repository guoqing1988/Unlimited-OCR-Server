"""server.py 两个修复点的单元测试。

覆盖内容：
1. process_raw_output：图片裁剪失败（坐标无效/原图缺失）时不生成死链图片链接
   —— 修复前：链接无条件生成 → 客户端 404
2. extract_messages：本地绝对路径不加入 temps
   —— 修复前：本地路径被 finally 块 unlink 删除（误删调用方文件）
"""

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

import server

# 测试用 req_id 前缀，便于结束后清理
TEST_PREFIX = "test_fix_"


@pytest.fixture(autouse=True)
def _cleanup_test_dirs():
    """每个测试结束后清理测试产生的 images 子目录与临时文件。"""
    yield
    for d in server.IMAGES_DIR.iterdir():
        if d.is_dir() and d.name.startswith(TEST_PREFIX):
            import shutil
            shutil.rmtree(d, ignore_errors=True)


def _make_test_image(width: int = 999, height: int = 999) -> str:
    """生成一张纯色测试图，返回临时文件路径。"""
    img = Image.new("RGB", (width, height), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    import tempfile
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    f.write(buf.read())
    f.close()
    return f.name


def _run_process(raw_text: str, img_path: str, req_id: str) -> str:
    """调用 process_raw_output 并返回 Markdown。"""
    return server.process_raw_output(raw_text, img_path, req_id)


# ────────────────────────────────────────────────────────────────────────────
# process_raw_output：图片裁剪与链接生成
# ────────────────────────────────────────────────────────────────────────────

def test_valid_coords_generates_link_and_file(tmp_path):
    """正常坐标：生成图片链接，且裁剪文件真实存在。"""
    img_path = _make_test_image(999, 999)
    req_id = f"{TEST_PREFIX}valid"
    raw = (
        "<|det|>title [10, 10, 500, 60]<|/det|>标题\n"
        "<|det|>image [100, 100, 500, 500]<|/det|>\n"
    )
    md = _run_process(raw, img_path, req_id)
    assert "![标题](images/test_fix_valid/0.jpg)" in md, md
    assert (server.IMAGES_DIR / req_id / "0.jpg").exists()


def test_invalid_coords_no_image_link(tmp_path):
    """坐标无效（x1==x2，宽为0）：不生成任何图片链接（修复核心）。"""
    img_path = _make_test_image(999, 999)
    req_id = f"{TEST_PREFIX}invalid"
    raw = (
        "<|det|>title [10, 10, 500, 60]<|/det|>标题\n"
        "<|det|>image [200, 200, 200, 500]<|/det|>\n"
    )
    md = _run_process(raw, img_path, req_id)
    assert "![标题]" not in md, md
    # 文本内容仍保留
    assert "标题" in md
    assert not (server.IMAGES_DIR / req_id / "0.jpg").exists()


def test_missing_original_image_no_link(tmp_path):
    """原图加载失败（路径不存在）：不生成图片链接。"""
    req_id = f"{TEST_PREFIX}missing"
    raw = "<|det|>image [100, 100, 500, 500]<|/det|>\n"
    md = _run_process(raw, "/nonexistent/xxx.png", req_id)
    assert "![image]" not in md, md
    assert not (server.IMAGES_DIR / req_id / "0.jpg").exists()


def test_malformed_coords_no_link(tmp_path):
    """坐标元素不足（裁剪时 IndexError）：不输出任何图片链接。"""
    img_path = _make_test_image(999, 999)
    req_id = f"{TEST_PREFIX}malformed"
    # 只有 3 个坐标值，c[3] 越界 → 裁剪异常被吞，无文件产出
    raw = "<|det|>image [100, 100, 400]<|/det|>\n"
    md = _run_process(raw, img_path, req_id)
    assert "![image]" not in md, md
    assert not (server.IMAGES_DIR / req_id).exists() or \
        not any((server.IMAGES_DIR / req_id).iterdir())


def test_crop_failure_logged(caplog):
    """裁剪失败必须记录 warning 日志（便于线上排查"无图"问题）。"""
    img_path = _make_test_image(999, 999)
    req_id = f"{TEST_PREFIX}logcheck"
    with caplog.at_level("WARNING", logger="unlimited-ocr"):
        # 坐标无效（x1==x2）→ 触发裁剪跳过日志
        raw = "<|det|>image [200, 200, 200, 500]<|/det|>\n"
        _run_process(raw, img_path, req_id)
    assert any("图片裁剪" in r.message for r in caplog.records), caplog.text


def test_after_failed_image_idx_continues(tmp_path):
    """失败的 image 不占用序号：后续成功的图片编号连续。"""
    img_path = _make_test_image(999, 999)
    req_id = f"{TEST_PREFIX}seq"
    raw = (
        "<|det|>image [100, 100, 200, 200]<|/det|>\n"   # 有效 → 0.jpg
        "<|det|>image [300, 300, 300, 300]<|/det|>\n"   # 无效 → 跳过
        "<|det|>image [400, 400, 500, 500]<|/det|>\n"   # 有效 → 1.jpg
    )
    md = _run_process(raw, img_path, req_id)
    assert "images/test_fix_seq/0.jpg" in md, md
    assert "images/test_fix_seq/1.jpg" in md, md
    assert "images/test_fix_seq/2.jpg" not in md, md
    assert (server.IMAGES_DIR / req_id / "0.jpg").exists()
    assert (server.IMAGES_DIR / req_id / "1.jpg").exists()


# ────────────────────────────────────────────────────────────────────────────
# extract_messages：本地路径不进入 temps（防误删）
# ────────────────────────────────────────────────────────────────────────────

def test_local_path_not_in_temps(tmp_path):
    """本地绝对路径图片：不加入 temps，服务端不应删除调用方文件。"""
    img_path = _make_test_image(100, 100)
    messages = [
        server.Message(content=[
            {"type": "text", "text": "<image>\nFree OCR."},
            {"type": "image_url", "image_url": {"url": img_path}},
        ])
    ]
    prompt, imgs, temps = server.extract_messages(messages)
    assert img_path in imgs
    assert img_path not in temps
    # 文件必须保持存在（修复前会被 finally 删除）
    import os
    assert os.path.exists(img_path)
    os.unlink(img_path)


def test_base64_image_in_temps(tmp_path):
    """base64 图片：解码产物是服务端临时文件，应加入 temps 以便清理。"""
    img = Image.new("RGB", (50, 50), (10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    messages = [
        server.Message(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ])
    ]
    prompt, imgs, temps = server.extract_messages(messages)
    assert len(imgs) == 1
    assert len(temps) == 1
    assert Path(temps[0]).exists()
    Path(temps[0]).unlink(missing_ok=True)
