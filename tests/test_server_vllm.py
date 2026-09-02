"""server_vllm.py（vLLM 后端适配层）单元测试。

覆盖内容：
1. image_to_data_uri：本地图片正确转 data URI（MIME 推断）
2. vllm_chat：请求构造含 skip_special_tokens=False（关键！），
   且引擎返回非 200 / 不可达时正确抛 RuntimeError
3. process_raw_output 集成：vLLM 输出的 <|det|> 原始格式能正确
   还原为与 Transformers 版一致的 Markdown（无死链、标题层级正确）
4. 容器管理辅助（_container_status / _stop_container 的状态机，
   通过 mock docker 命令验证不抛异常）

说明：所有测试均为纯逻辑测试（mock 网络与 docker），
不依赖 9706 vLLM 服务或真实容器，可在无 GPU 环境离线运行。
"""

import io
import json
import tempfile
import unittest.mock as mock
from pathlib import Path

from PIL import Image

import server_vllm as sv

TEST_PREFIX = "test_vllm_"


def _make_image_path(width: int = 200, height: int = 300, fmt: str = "PNG") -> str:
    """生成一张纯色测试图，返回临时文件路径。"""
    img = Image.new("RGB", (width, height), (50, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    f = tempfile.NamedTemporaryFile(suffix=f".{fmt.lower()}", delete=False)
    f.write(buf.getvalue())
    f.close()
    return f.name


def test_image_to_data_uri_png():
    """PNG 图片应编码为 image/png data URI。"""
    path = _make_image_path(fmt="PNG")
    try:
        uri = sv.image_to_data_uri(path)
        assert uri.startswith("data:image/png;base64,")
        # base64 部分应能解码回图片
        import base64
        data = base64.b64decode(uri.split(",", 1)[1])
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        Path(path).unlink(missing_ok=True)


def test_image_to_data_uri_jpeg():
    """JPEG 图片应编码为 image/jpeg data URI（魔数识别）。"""
    path = _make_image_path(fmt="JPEG")
    try:
        uri = sv.image_to_data_uri(path)
        assert uri.startswith("data:image/jpeg;base64,")
    finally:
        Path(path).unlink(missing_ok=True)


def test_vllm_chat_payload_keeps_det_tokens():
    """vllm_chat 请求必须带 skip_special_tokens=False（保留 <|det|> 标签）。

    若该参数缺失，vLLM 默认剥掉 <|det|> / <|/det|> 特殊 token，
    process_raw_output 将无法解析 —— 这是适配层正确性的核心。
    """
    img_path = _make_image_path()
    try:
        with mock.patch.object(sv.urllib.request, "urlopen") as m_urlopen:
            resp = mock.MagicMock()
            resp.read.return_value = json.dumps({
                "choices": [{"message": {"content": "<|det|>text [1,2,3,4]<|/det|>hi"}}],
            }).encode()
            m_urlopen.return_value.__enter__.return_value = resp

            out = sv.vllm_chat("<image>document parsing.", [img_path])

        # 验证请求体构造
        req = m_urlopen.call_args.args[0]
        payload = json.loads(req.data)
        assert payload["skip_special_tokens"] is False
        assert payload["vllm_xargs"] == {"ngram_size": 35, "window_size": 128}, (
            "必须携带 vllm_xargs 启用引擎侧 NGram 防重复，否则处理器被静默禁用"
        )
        assert payload["messages"][0]["content"][0]["text"] == "<image>document parsing."
        assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:")
        # 输出原样返回
        assert out == "<|det|>text [1,2,3,4]<|/det|>hi"
    finally:
        Path(img_path).unlink(missing_ok=True)


def test_vllm_chat_engine_error_raises():
    """引擎返回非 200 时应抛出 RuntimeError（供上层转 503/500）。"""
    img_path = _make_image_path()
    try:
        with mock.patch.object(sv.urllib.request, "urlopen") as m_urlopen:
            err = sv.urllib.error.HTTPError(
                "http://x/v1/chat/completions", 500, "err", None, None)
            m_urlopen.side_effect = err
            try:
                sv.vllm_chat("p", [img_path])
                assert False, "应抛出 RuntimeError"
            except RuntimeError as exc:
                assert "500" in str(exc)
    finally:
        Path(img_path).unlink(missing_ok=True)


def _det_raw_text() -> str:
    """构造一段 vLLM 输出（带 <|det|> 标签），等价真实引擎返回。"""
    return (
        "<|det|>header [20, 20, 280, 40]<|/det|>TITLE ONE\n"
        "<|det|>text [20, 60, 300, 90]<|/det|>第一段正文内容测试。\n"
        "<|det|>title [30, 120, 150, 140]<|/det|>小标题\n"
        "<|det|>image [20, 160, 200, 260]<|/det|>\n"
        "<|det|>text [20, 300, 300, 330]<|/det|>第二段正文。\n"
        "<|det|>page_number [500, 900, 600, 950]<|/det|>12\n"
    )


def test_process_raw_output_with_vllm_format(tmp_path):
    """vLLM 的 <|det|> 原始输出经 process_raw_output 应还原为结构正确的 Markdown。

    验证点：标题映射为 #/##、图片被裁剪保存且生成非死链引用、
    页码跳过、正文段落保留。
    """
    # 原图：999 宽高方便坐标直接映射（适配层裁剪坐标 /999*宽高）
    src = _make_image_path(999, 999)
    req_id = TEST_PREFIX + "format"
    try:
        md = sv.process_raw_output(_det_raw_text(), src, req_id)
        assert "# TITLE ONE" in md        # header → # 一级标题
        assert "## 小标题" in md           # title → ## 二级标题
        assert "第一段正文内容测试。" in md  # 正文保留
        assert "page_number" not in md    # 页码跳过
        assert "![小标题](images/" in md   # image 生成 alt 引用（用最近标题）
        # 实际裁剪文件存在（非死链）
        import re
        links = re.findall(r"!\[[^\]]*\]\(images/([^)]+)\)", md)
        assert links, "应有图片链接"
        for link in links:
            assert (sv.IMAGES_DIR / link).exists(), f"图片文件缺失: {link}"
    finally:
        Path(src).unlink(missing_ok=True)


def test_process_raw_output_crop_failure_no_deadlink(tmp_path):
    """图片坐标无效时不应生成死链（与原版修复行为一致）。"""
    src = _make_image_path(999, 999)
    req_id = TEST_PREFIX + "nodead"
    raw = (
        "<|det|>text [20, 20, 280, 40]<|/det|>some text\n"
        "<|det|>image [100, 100, 50, 50]<|/det|>\n"  # x2<x1, y2<y1 → 无效
    )
    try:
        md = sv.process_raw_output(raw, src, req_id)
        assert "![" not in md, "无效坐标不应生成图片链接"
    finally:
        Path(src).unlink(missing_ok=True)


def test_extract_messages_local_path_not_deleted():
    """本地绝对路径图片不加入 temps（不误删调用方文件），与 server.py 一致。"""
    img_path = _make_image_path()
    try:
        msg = sv.Message(role="user", content=[
            {"type": "image_url", "image_url": {"url": img_path}},
        ])
        prompt, imgs, temps = sv.extract_messages([msg])
        assert img_path in imgs
        assert img_path not in temps  # 本地路径不清理
        assert "<image>" in prompt
    finally:
        Path(img_path).unlink(missing_ok=True)


def test_container_status_running(mocker_capsys=None):
    """容器 inspect 返回 running 时应标记容器运行中且不抛异常。"""
    with mock.patch.object(sv, "_docker", return_value="running") as m_docker:
        st = sv._container_status()
        assert st == "running"
        m_docker.assert_called_once()
    with mock.patch.object(sv, "_docker", return_value="exited"):
        st = sv._container_status()
        assert st == "exited"


def test_container_status_missing_when_docker_error():
    """docker inspect 失败（容器不存在）应返回 missing。"""
    with mock.patch.object(sv, "_docker", side_effect=RuntimeError("No such container")):
        st = sv._container_status()
        assert st == "missing"


# ─────────────────────────────────────────────────────────────
# 幻觉护栏 strip_hallucinations 测试
# ─────────────────────────────────────────────────────────────

def test_guard_keeps_normal_text():
    """正常 det 文本行（含相似结构）不应被误删。"""
    raw = (
        "<|det|>title [98, 39, 350, 63]<|/det|>天外怪客大鬧香港？\n"
        "<|det|>text [33, 131, 205, 203]<|/det|>和諧式飛機有幾樣特點\n"
        "<|det|>text [29, 226, 201, 279]<|/det|>一、它縮短旅途時間\n"
        "<|det|>page_number [918, 28, 928, 38]<|/det|>5\n"
    )
    assert sv.strip_hallucinations(raw) == raw


def test_guard_drops_repeated_lines():
    """同一文本行重复 >=2 次时应只保留首次。"""
    raw = (
        "<|det|>text [399, 249, 486, 255]<|/det|>hello world line\n"
        "<|det|>text [399, 255, 486, 261]<|/det|>hello world line\n"
        "<|det|>text [399, 261, 486, 267]<|/det|>hello world line\n"
    )
    out = sv.strip_hallucinations(raw)
    assert out.count("hello world line") == 1
    assert out.count("<|det|>") == 1


def test_guard_drops_halluc_markers():
    """含已知幻觉模板句的行（quick brown fox / 无文字声明 / 财务模板）应删除。"""
    raw = (
        "<|det|>text [33, 131, 205, 203]<|/det|>正常中文段落內容\n"
        "<|det|>text [399, 315, 486, 321]<|/det|>The quick brown fox jumps over the lazy dog.\n"
        "<|det|>text [399, 300, 486, 999]<|/det|>The image contains no text. The horizontal lines are stylistic\n"
        "<|det|>text [399, 300, 486, 999]<|/det|>(1) 2017年1月1日，公司与关联方发生的交易金额为人民币4,000万元。\n"
    )
    out = sv.strip_hallucinations(raw)
    assert "正常中文段落內容" in out
    assert "quick brown fox" not in out
    assert "image contains no text" not in out
    assert "2017年" not in out


def test_guard_drops_out_of_page_lines():
    """y1 >= 995（页面底部外虚构行）与 y 跨度 >= 500（整栏无文字说明）应删除。"""
    raw = (
        "<|det|>text [399, 987, 486, 999]<|/det|>tail line ok\n"
        "<|det|>text [399, 999, 486, 999]<|/det|>phantom bottom line\n"
        "<|det|>text [399, 0, 486, 999]<|/det|>whole column nonsense\n"
    )
    out = sv.strip_hallucinations(raw)
    assert "phantom bottom line" not in out
    assert "whole column nonsense" not in out
    assert "tail line ok" in out


def test_guard_cuts_inline_loop():
    """单行内循环（大小写混排）应截断到循环起点。"""
    raw = (
        "<|det|>text [399, 249, 486, 255]<|/det|>the first. The first, the first, the first, the first, the first\n"
    )
    out = sv.strip_hallucinations(raw)
    assert "the first" in out
    # 行内循环截断后不应再有 3 连重复
    seg = out.split("<|/det|>")[1]
    assert seg.count("the first, the first, the first") == 0


def test_guard_drops_orphan_det_fragments():
    """孤儿行若以残缺 det 标签开头（解析失败碎片）应删除。"""
    raw = (
        "<|det|>text [33, 131, 205, 203]<|/det|>正常內容\n"
        "<|det|>text [399, 999, 486, 999]]\n"   # 括号残缺 → 孤儿行
    )
    out = sv.strip_hallucinations(raw)
    assert "正常內容" in out
    assert "999]]" not in out


def test_guard_empty_and_plain_text():
    """空文本与无 det 的普通孤立行不受影响。"""
    assert sv.strip_hallucinations("") == ""
    plain = "Just a bare line without det tags\n\nanother line"
    assert sv.strip_hallucinations(plain) == plain


def test_guard_cuts_cjk_inline_loop():
    """中文 4 字单元行内循环（罪狀/圖謀刺殺类）应截断。"""
    raw = (
        "<|det|>text [33, 249, 205, 300]<|/det|>罪狀第三條：圖謀刺殺圖謀刺殺圖謀刺殺圖謀刺殺圖謀刺殺\n"
    )
    out = sv.strip_hallucinations(raw)
    seg = out.split("<|/det|>")[1]
    # 应截到首个单元结束，不再有多连重复
    assert "罪狀第三條：圖謀刺殺" in seg
    assert "刺殺圖謀刺殺圖謀" not in seg
