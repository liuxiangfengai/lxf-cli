# lxf-cli

本地离线 AI 命令行工具集。当前子命令：

- `lxf-cli video2txt <视频/音频/目录> <输出目录>` —— 音视频转带时间戳 txt

完全离线、免费、无限次。

## 安装

```bash
# 推荐：在专用 venv 中安装（含 ASR 引擎依赖）
pip install "lxf-cli[asr]" .
# 或开发模式（源码改动即时生效）
pip install -e ".[asr]"
```

安装后即可使用 `lxf-cli` 命令。

## 模型

模型文件较大（约 1.1GB），不打包进 pip 包。默认从以下位置自动查找：

1. 环境变量 `LXF_MODELS_DIR` 指向的目录
2. 项目目录下的 `models/`
3. `~/Desktop/lxf_cli/models/`（本地开发默认）

每个模型是独立子目录，例如 SenseVoice：
`models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/`

可用 `lxf-cli model path` 查看实际使用的模型路径。

## 开发：新增一个子命令

在 `src/lxf_cli/commands/` 下新建 `xxx.py`：

```python
import click

@click.command("xxx")
@click.argument("input")
def xxx(input):
    """一句话说明"""
    click.echo(f"处理 {input}")

def register(cli):
    cli.add_command(xxx)
```

文件放入 `commands/` 目录即自动注册，无需改动入口代码。
