<p align="center">
  <img src="assets/baidu.png" width="40%" alt="Baidu Inc." />
</p>

<hr>

<h1 align="center">Unlimited-OCR 推理服务</h1>

<div align="center">
  <a href="https://github.com/baidu/Unlimited-OCR">
    <img alt="GitHub(上游)" src="https://img.shields.io/badge/GitHub-上游仓库-181717?logo=github&logoColor=white" />
  </a>
  <a href="https://huggingface.co/baidu/Unlimited-OCR">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107?color=ffc107&logoColor=white" />
  </a>
  <a href="https://arxiv.org/abs/2606.23050">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Unlimited OCR Works-b31b1b?logo=arxiv&logoColor=white" />
  </a>
</div>

<p align="center">
    <img src="assets/Unlimited-OCR.png" width="1000" alt="Unlimited OCR overview" />
</p>

Unlimited-OCR 是百度推出的单次长文档解析（One-shot Long-horizon Parsing）OCR 模型，
基于 DeepSeek-OCR 进一步优化。本仓库为**本地部署版推理服务**，提供 OpenAI 兼容的
HTTP API（单图 / 多图 / PDF 解析，Markdown 输出 + 图片自动提取）。

> 本仓库 fork 自 [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)，在其模型与推理
> 示例基础上增加了完整的服务化部署：HTTP API、认证、PDF 转图、Markdown 后处理、图片裁剪、
> 三后端（llama.cpp GGUF / vLLM FP8 / Transformers）、systemd 托管与自动启停。

---

## 快速开始（llama.cpp GGUF，当前生产）

完整部署见 [DEPLOY_GGUF.md](DEPLOY_GGUF.md)，API 用法见 [SERVER.md](SERVER.md)。

> 需要**打了 Unlimited-OCR-2 补丁**的 llama.cpp（主线不支持 `unlimited-ocr2` 架构），
> 模型为 [AtomicChat/Unlimited-OCR-2-GGUF](https://huggingface.co/AtomicChat/Unlimited-OCR-2-GGUF)
> 的 Q8_0 + mmproj。

```bash
# 0) 编译补丁版 llama-server（详细步骤见 DEPLOY_GGUF.md）
cd /data1/www/llama-ocr2
git checkout b11913 && git apply /tmp/unlimited-ocr2.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j 48 --target llama-server llama-mtmd-cli

# 1) Python 虚拟环境（适配层不加载 torch）
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python fastapi "uvicorn[standard]" pydantic \
  python-dotenv python-multipart requests pillow pymupdf

# 2) 配置 .env（端口 / 密钥 / GPU）
cat > .env <<'EOF'
HOST=0.0.0.0
PORT=18001
API_KEY=ocr-xxxxxxxx
LLAMA_GPU=7
LLAMA_CTX=32768
EOF

# 3) systemd 启动（开机自启）
sudo cp unlimited-ocr-gguf.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now unlimited-ocr-gguf
```

调用示例：

```bash
curl -s http://localhost:18001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <你的密钥>" \
  -d '{
    "model": "Unlimited-OCR",
    "messages": [{
      "role": "user",
      "content": [{"type": "image_url", "image_url": {"url": "/data/docs/page.png"}}]
    }]
}' | python3 -m json.tool
```

支持的输入：base64 图片、本地图片/PDF 路径、HTTP URL；输出为 Markdown，
文档内图片自动裁剪并保存到 `images/{req_id}/`，经 `/images/` 静态路径提供。

> 不传 prompt 时由模型内嵌 chat template 自动生成官方指令；自定义指令可作为
> text 块传入（服务端会放到图片之后）。

## 架构

```
Client ──► :18001 (systemd unlimited-ocr-gguf, FastAPI server_gguf.py)
                │
                └─ HTTP ──► llama-server 子进程 :9706
                              (补丁版 llama.cpp + Q8_0 GGUF + DASH attention)
```

| 组件 | 说明 |
|------|------|
| `server_gguf.py` | 适配层（当前生产）：认证 / PDF 转图 / v2 标签解析 / Markdown 后处理 / 图片裁剪 / 引擎生命周期 |
| `llama-server` 子进程 | 推理引擎：补丁版 llama.cpp + `Unlimited-OCR-2-Q8_0.gguf` + mmproj |
| `server_vllm.py` | 上一版适配层（vLLM FP8，保留可回退） |
| `server.py` | Transformers 版（原生产，已停用，保留可回退） |

**生命周期**：引擎不常驻。首个请求到达时自动拉起 llama-server（冷启动约 5s），
空闲 900s 后自动停止释放显存；模型文件保留在页缓存，下次启动更快。

## 文档导航

| 文档 | 内容 |
|------|------|
| [SERVER.md](SERVER.md) | API 完整使用文档（接口、输入方式、示例、Prompt 参考） |
| [DEPLOY_GGUF.md](DEPLOY_GGUF.md) | llama.cpp GGUF 部署文档（方案 C，**当前生产**） |
| [DEPLOY_VLLM.md](DEPLOY_VLLM.md) | vLLM FP8 部署文档（方案 A，保留可回退） |
| [DEPLOY.md](DEPLOY.md) | Transformers 部署文档（方案 B，原始方案，保留可回退） |
| [CLAUDE.md](CLAUDE.md) | 项目说明与开发约定（供 AI 编码助手阅读） |
| `tests/` | 单元测试（`pytest tests/`） |

## 常用运维命令

```bash
# 服务状态 / 日志
sudo systemctl status unlimited-ocr-gguf
sudo journalctl -u unlimited-ocr-gguf -f
tail -f /data1/www/Unlimited-OCR/log/llama-server.log

# 健康检查（status: ok=就绪 / stopped=已空闲卸载 / degraded=异常）
curl http://localhost:18001/health | python3 -m json.tool

# 手动停止引擎（释放显存，下次请求自动拉起）
curl -X POST http://localhost:18001/admin/unload -H "Authorization: Bearer <密钥>"

# 回退 vLLM FP8 版（先停 GGUF 服务让出端口）
sudo systemctl disable --now unlimited-ocr-gguf
sudo systemctl enable --now unlimited-ocr-vllm
```

## 实测性能（RTX 4090，Q8_0）

| 维度 | Transformers bf16（旧） | vLLM FP8（上一版） | llama.cpp GGUF（当前） |
|------|:---:|:---:|:---:|
| 单页耗时 | ~9s | ~3.3s | ~5s |
| 活跃显存 | ~8.6 GB | ~8 GB | 6.0 GB（4k ctx）/ 8.4 GB（32k ctx） |
| 空闲显存 | 0（进程内卸载） | 0（容器停止） | 0（子进程停止） |
| 冷启动 | ~5-10s | ~30-40s | ~5s |
| 并发 | 单请求 | 多请求 | 单请求（`-np 1`） |

## Release（上游模型发布记录）

- [2026/07/03] 🤝 感谢百度云团队支持，模型已上线[百度智能云](https://cloud.baidu.com/doc/OCR/s/fmr1p39gb)。
- [2026/06/28] 🤝 感谢 [vLLM 社区](https://github.com/vllm-project/vllm)与 Tianyu Guo，模型支持 vLLM 推理。
- [2026/06/24] 🤝 感谢 [AK](https://x.com/_akhaliq) 制作的 demo，已上线 [Hugging Face Spaces](https://huggingface.co/spaces/baidu/Unlimited-OCR)。
- [2026/06/23] 📄 论文已发布在 [arXiv](https://arxiv.org/abs/2606.23050)。
- [2026/06/23] 🤝 感谢 [ModelScope 社区](https://github.com/modelscope)，模型已上线 [ModelScope](https://modelscope.cn/models/PaddlePaddle/Unlimited-OCR)。
- [2026/06/22] 🚀 发布 [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)，将 [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) 向前推进一步。

## 可视化

<img src="assets/long-horizon-ocr.gif" width="100%" alt="Long-horizon OCR demo" />

## 致谢

感谢 [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)、[DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2)、[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 提供的宝贵模型与思路。

## 引用

```bibtex
@misc{yin2026unlimitedocrworks,
      title={Unlimited OCR Works},
      author={Youyang Yin and Huanhuan Liu and YY and Qunyi Xie and Chaorun Liu and Shiqi Yang and Shaohua Wang and Zhanlong Liu and Hao Zou and Jinyue Chen and Shu Wei and Jingjing Wu and Mingxin Huang and Zhen Wu and Guibin Wang and Tengyu Du and Lei Jia},
      year={2026},
      eprint={2606.23050},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.23050},
}
```
