"""server_gguf.py（llama.cpp GGUF 后端适配层）单元测试。

覆盖内容：
1. _build_engine_cmd：llama-server 启动参数（--jinja / -np 1 / -c / --alias）
2. llama_chat：请求构造——默认指令不随文本发送（否则模板会把文本放到图片前，
   导致多页只输出第一页）；自定义指令追加在图片之后；DRY 防重复参数齐全；
   引擎错误时抛 RuntimeError
3. process_raw_output：v2 的 <BLOCK>type[box]<FORK>text 格式解析为与原版一致的
   Markdown（标题层级、图片裁剪链接、footer 跳过、<MERGE> 清理、全角逗号容错）
4. strip_hallucinations：循环行/越界行过滤，且能容忍非数字坐标

说明：全部为纯逻辑测试（mock 网络），不依赖 9706 llama-server 或 GPU，
可在无模型环境离线运行。
"""

import io
import json
import tempfile
import unittest.mock as mock
from pathlib import Path

from PIL import Image

import server_gguf as sg


def _make_image_path(width: int = 200, height: int = 300) -> str:
    """生成一张纯色测试图，返回临时文件路径。"""
    img = Image.new("RGB", (width, height), (50, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    f.write(buf.getvalue())
    f.close()
    return f.name


class _FakeResp:
    """urllib.request.urlopen 的假响应。"""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 引擎启动参数
# ═══════════════════════════════════════════════════════════════════════════

def test_build_engine_cmd_basic():
    """启动命令必须带 --jinja、单 slot、指定上下文与端口、统一 alias。"""
    cmd = sg._build_engine_cmd()
    assert sg.LLAMA_SERVER_BIN in cmd
    assert "--jinja" in cmd
    assert "-np" in cmd and cmd[cmd.index("-np") + 1] == "1"
    assert "-c" in cmd and cmd[cmd.index("-c") + 1] == str(sg.LLAMA_CTX)
    assert "--mmproj" in cmd
    assert "--alias" in cmd and cmd[cmd.index("--alias") + 1] == sg.ENGINE_MODEL_NAME


# ═══════════════════════════════════════════════════════════════════════════
# llama_chat 请求构造
# ═══════════════════════════════════════════════════════════════════════════

def test_llama_chat_default_prompt_not_sent():
    """官方默认指令不显式发送：模型只认模板补的 "Multi page parsing."。

    实测：显式发 "document parsing." 会让 v2 模型进入错误解析路径并产生幻觉
    （同一张问卷图：模板默认 40 行正确 vs document parsing. 43 行全幻觉）。
    """
    for prompt in ("<image>document parsing.", "<image>Multi page parsing."):
        with mock.patch.object(sg.urllib.request, "urlopen", return_value=_FakeResp(
            {"choices": [{"message": {"content": "<PAGE><BLOCK>text[0,0,1,1]<FORK>hi"}}]}
        )) as m:
            out = sg.llama_chat(prompt, [_make_image_path()])
        assert out.startswith("<PAGE>")
        payload = json.loads(m.call_args[0][0].data)
        content = payload["messages"][0]["content"]
        assert content[0]["type"] == "image_url"
        # 内容里只有图片，没有默认指令文本
        assert all(c["type"] == "image_url" for c in content), content
        assert payload["skip_special_tokens"] is False
        assert payload["dry_multiplier"] == sg.DRY_MULTIPLIER
        assert payload["dry_allowed_length"] == sg.DRY_ALLOWED_LENGTH
        assert payload["model"] == sg.ENGINE_MODEL_NAME


def test_llama_chat_custom_prompt_appended_after_images():
    """自定义指令放在图片之后，避免模板把文本渲染到图片前。"""
    with mock.patch.object(sg.urllib.request, "urlopen", return_value=_FakeResp(
        {"choices": [{"message": {"content": "ok"}}]}
    )) as m:
        sg.llama_chat("<image>请提取表格", [_make_image_path()])
    payload = json.loads(m.call_args[0][0].data)
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[-1]["type"] == "text"
    assert content[-1]["text"] == "请提取表格"


def test_llama_chat_engine_error_raises():
    """引擎不可达时必须抛出 RuntimeError（由上层转 500）。"""
    import urllib.error
    with mock.patch.object(sg.urllib.request, "urlopen",
                           side_effect=urllib.error.URLError("boom")):
        try:
            sg.llama_chat("Multi page parsing.", [_make_image_path()])
        except RuntimeError as e:
            assert "llama-server" in str(e)
        else:
            raise AssertionError("expected RuntimeError")


# ═══════════════════════════════════════════════════════════════════════════
# process_raw_output：v2 标签 → Markdown
# ═══════════════════════════════════════════════════════════════════════════

def test_process_raw_output_v2_markdown(tmp_path):
    """header/title/subtitle/text 映射、footer 跳过、image 裁剪链接生成。"""
    old = sg.IMAGES_DIR
    sg.IMAGES_DIR = tmp_path
    try:
        img = _make_image_path()
        raw = (
            "<PAGE><BLOCK>header[0, 0, 999, 50]<FORK>MAIN TITLE\n"
            "<BLOCK>title<LEVEL1>[0, 60, 500, 100]<FORK>Section One\n"
            "<BLOCK>title<LEVEL2>[0, 110, 500, 140]<FORK>Sub Section\n"
            "<BLOCK>text[0, 160, 999, 200]<FORK>Hello world.\n"
            "<BLOCK>image[100, 220, 400, 500]<FORK>\n"
            "<BLOCK>footer[0, 960, 999, 990]<FORK>page footer\n"
        )
        md = sg.process_raw_output(raw, img, "unit/page_0")
        assert md.startswith("# MAIN TITLE")
        assert "## Section One" in md
        assert "### Sub Section" in md
        assert "Hello world." in md
        assert "page footer" not in md
        # alt 取最近遇到的标题（此例为 title<LEVEL2> 的 Sub Section）
        assert "![Sub Section](images/unit/page_0/0.jpg)" in md
        assert (tmp_path / "unit" / "page_0" / "0.jpg").is_file()
    finally:
        sg.IMAGES_DIR = old


def test_process_raw_output_merge_marker_cleared(tmp_path):
    """<MERGE> 标记必须从输出中清除，跨页文本连续输出。"""
    old = sg.IMAGES_DIR
    sg.IMAGES_DIR = tmp_path
    try:
        img = _make_image_path()
        raw = (
            "<BLOCK>text[0, 0, 999, 50]<FORK>first part\n"
            "<MERGE>\n"
            "<BLOCK>text[0, 60, 999, 100]<FORK>continued part\n"
        )
        md = sg.process_raw_output(raw, img, "unit/page_1")
        assert "<MERGE>" not in md
        assert "<PAGE>" not in md
        assert md.index("first part") < md.index("continued part")
    finally:
        sg.IMAGES_DIR = old


def test_process_raw_output_fullwidth_comma(tmp_path):
    """模型偶发全角逗号坐标时不得崩溃（跳过该图即可）。"""
    old = sg.IMAGES_DIR
    sg.IMAGES_DIR = tmp_path
    try:
        img = _make_image_path()
        raw = "<BLOCK>image[884， 239, 900, 300]<FORK>\n<BLOCK>text[0, 0, 100, 20]<FORK>ok\n"
        md = sg.process_raw_output(raw, img, "unit/page_2")
        assert "ok" in md
    finally:
        sg.IMAGES_DIR = old


def test_process_raw_output_image_failure_no_dead_link(tmp_path):
    """坐标无效导致裁剪失败时，不输出图片链接（避免 404）。"""
    old = sg.IMAGES_DIR
    sg.IMAGES_DIR = tmp_path
    try:
        img = _make_image_path()
        raw = "<BLOCK>image[500, 500, 100, 100]<FORK>\n<BLOCK>text[0, 0, 100, 20]<FORK>text only\n"
        md = sg.process_raw_output(raw, img, "unit/page_3")
        assert "![" not in md
        assert "text only" in md
    finally:
        sg.IMAGES_DIR = old


# ═══════════════════════════════════════════════════════════════════════════
# 幻觉护栏
# ═══════════════════════════════════════════════════════════════════════════

def test_strip_hallucinations_drops_loop_and_out_of_page():
    """行内循环与越界 y 坐标行应被丢弃，正常行保留。"""
    raw = (
        "<BLOCK>text[0, 100, 200, 120]<FORK>Normal line.\n"
        "<BLOCK>text[0, 100, 200, 120]<FORK>The first, The first, The first, The first\n"
        "<BLOCK>text[0, 996, 200, 999]<FORK>Out of page line\n"
    )
    out = sg.strip_hallucinations(raw)
    assert "Normal line." in out
    assert "The first, The first, The first" not in out
    assert "Out of page line" not in out


def test_strip_hallucinations_tolerates_bad_coords():
    """坐标含全角逗号或非数字时不得抛异常。"""
    raw = "<BLOCK>text[884， 239, 900, 300]<FORK>abc\n"
    assert "abc" in sg.strip_hallucinations(raw)


# ═══════════════════════════════════════════════════════════════════════════
# 重复图片去重（v2 双栏/大图版面会把同一张图输出两次）
# ═══════════════════════════════════════════════════════════════════════════

def test_box_iou():
    """IoU：同一框=1，无交集=0，部分重叠介于两者之间。"""
    assert sg._box_iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0
    assert sg._box_iou([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0
    part = sg._box_iou([0, 0, 100, 100], [0, 0, 100, 50])
    assert 0.4 < part < 0.6
    assert sg._box_iou([0, 0], [1, 1, 2, 2]) == 0.0  # 坐标不足 4 个不崩溃


def test_process_raw_output_dedup_duplicate_image(tmp_path):
    """同一张大图被模型输出两次（坐标几乎相同）时，只保留一张。"""
    old = sg.IMAGES_DIR
    sg.IMAGES_DIR = tmp_path
    try:
        img = _make_image_path(400, 600)
        raw = (
            "<PAGE><BLOCK>image[279, 302, 724, 967]<FORK>\n"
            "<BLOCK>text[54, 461, 268, 517]<FORK>左侧正文\n"
            "<BLOCK>image[279, 302, 724, 957]<FORK>\n"
            "<BLOCK>text[743, 457, 959, 608]<FORK>右侧正文\n"
        )
        md = sg.process_raw_output(raw, img, "unit/dup")
        assert md.count("![") == 1, md
        assert "左侧正文" in md and "右侧正文" in md
        files = list((tmp_path / "unit" / "dup").glob("*.jpg"))
        assert len(files) == 1, files
    finally:
        sg.IMAGES_DIR = old


def test_process_raw_output_keeps_distinct_images(tmp_path):
    """不同位置的两张图不应被去重。"""
    old = sg.IMAGES_DIR
    sg.IMAGES_DIR = tmp_path
    try:
        img = _make_image_path(400, 600)
        raw = (
            "<BLOCK>image[10, 10, 200, 300]<FORK>\n"
            "<BLOCK>image[500, 400, 900, 900]<FORK>\n"
        )
        md = sg.process_raw_output(raw, img, "unit/distinct")
        assert md.count("![") == 2, md
    finally:
        sg.IMAGES_DIR = old


def test_process_raw_output_paragraph_blank_line(tmp_path):
    """v2 每个 text 块是独立段落：块间必须是空行，否则 Markdown 渲染会并成一段。"""
    old = sg.IMAGES_DIR
    sg.IMAGES_DIR = tmp_path
    try:
        img = _make_image_path()
        raw = (
            "<BLOCK>text[0, 0, 999, 50]<FORK>第一段\n"
            "<BLOCK>text[0, 60, 999, 100]<FORK>第二段\n"
        )
        md = sg.process_raw_output(raw, img, "unit/para")
        assert "第一段\n\n第二段" in md, repr(md)

        # <MERGE> 表示跨页续写，同段落内不插空行
        raw_merge = (
            "<BLOCK>text[0, 0, 999, 50]<FORK>前半句\n"
            "<MERGE>\n"
            "<BLOCK>text[0, 60, 999, 100]<FORK>后半句\n"
        )
        md2 = sg.process_raw_output(raw_merge, img, "unit/para2")
        assert "前半句\n后半句" in md2, repr(md2)
    finally:
        sg.IMAGES_DIR = old


def test_is_junk_fragment():
    """模型格式崩溃产生的碎片行应被识别；正常正文不能被误判。"""
    junk = [
        "[119, 1, 2, 0, 3, 0",
        "image[0, 0, 999, 999]",
        "[0, 0, 110, 200, 120, 200, 400, 400, 800, 800, 2000, 2000]",
        "[0, 200, 800, 400, 2000]",
        "[1]",
        "...",
        "。。",
        "<FORK>",
    ]
    for s in junk:
        assert sg._is_junk_fragment(s), s

    normal = [
        "1, 2, 3",
        "[1] Smith et al.",
        "普通正文一行。",
        "The quick brown fox.",
    ]
    for s in normal:
        assert not sg._is_junk_fragment(s), s


def test_process_raw_output_drops_junk_fragments(tmp_path):
    """坐标数组/残缺标签/纯标点行不得出现在 Markdown 结果中。"""
    old = sg.IMAGES_DIR
    sg.IMAGES_DIR = tmp_path
    try:
        img = _make_image_path()
        raw = (
            "<BLOCK>text[0, 0, 999, 50]<FORK>正文内容\n"
            "image[0, 0, 999, 999]\n"
            "[119, 1, 2, 0, 3, 0\n"
            "[0, 200, 800, 400, 2000]\n"
            "[1]\n"
            "[1]\n"
            "...\n"
        )
        md = sg.process_raw_output(raw, img, "unit/junk")
        assert "正文内容" in md, md
        for bad in ("119, 1, 2", "image[0", "[1]", "..."):
            assert bad not in md, (bad, md)
    finally:
        sg.IMAGES_DIR = old


def test_process_raw_output_duplicate_block_prefix(tmp_path):
    """模型偶发输出 <BLOCK><BLOCK>text[...]，应合并前缀后正常解析。"""
    old = sg.IMAGES_DIR
    sg.IMAGES_DIR = tmp_path
    try:
        img = _make_image_path()
        raw = "<BLOCK><BLOCK>text[0, 0, 999, 50]<FORK>双前缀正文\n"
        md = sg.process_raw_output(raw, img, "unit/dupblock")
        assert "双前缀正文" in md, md
        assert "<BLOCK>" not in md, md
    finally:
        sg.IMAGES_DIR = old


def test_stream_loop_detected():
    """流式早停：整行重复达阈值时判定循环；正常文档与短输出不得误判。"""
    # 同一行重复 4 次 → 循环
    loop = ["<BLOCK>text[0, 0, 999, 50]<FORK>Unlimited OCR Works"] * 25
    assert sg._stream_loop_detected(loop)

    # 行数不足阈值（正常短页）→ 不判定
    short = ["<BLOCK>text[0, 0, 999, 50]<FORK>同一行"] * 10
    assert not sg._stream_loop_detected(short)

    # 正常多行内容 → 不判定
    import random

    random.seed(0)
    normal = [f"<BLOCK>text[0, {i}, 999, {i + 10}]<FORK>第 {i} 段不同的正文内容" for i in range(40)]
    assert not sg._stream_loop_detected(normal)
