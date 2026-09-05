"""video2txt：视频/音频 → 带时间戳 txt。

示例：lxf-cli video2txt 视频.mp4 输出目录/
"""
from __future__ import annotations

import glob
import os
import sys
import time

import click

from lxf_cli.core import audio
from lxf_cli.core.asr import SenseVoiceEngine, Segment


def _segments_to_txt(segs: list[Segment]) -> str:
    lines = [f"[{audio.fmt_ts(s.start)} → {audio.fmt_ts(s.end)}] {s.text}" for s in segs]
    full = "".join(s.text for s in segs)
    return "\n".join(lines) + f"\n\n──── 全文合并 ────\n{full}\n"


def _process_one(engine: SenseVoiceEngine, media_path: str, out_dir: str) -> None:
    stem = os.path.splitext(os.path.basename(media_path))[0]
    t0 = time.time()
    segs = engine.transcribe(media_path)
    elapsed = time.time() - t0

    if not segs:
        click.echo(f"  ✗ {os.path.basename(media_path)}: 未识别到语音")
        return

    txt_path = os.path.join(out_dir, stem + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(_segments_to_txt(segs))

    if segs[0].start == 0.0 and len(segs) == 1:
        mode = "短媒体"
    else:
        mode = f"长媒体 VAD 分 {len(segs)} 段"
    click.echo(f"  ✓ {os.path.basename(media_path)} ({mode}, 耗时 {elapsed:.1f}s)")
    click.echo(f"    → {txt_path}")


@click.command("video2txt")
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path(file_okay=False))
@click.option("--language", default="zh", show_default=True,
              help="识别语言：zh/en/yue/ja/ko/auto")
@click.option("--threads", default=4, show_default=True, help="推理线程数")
def video2txt(input: str, output: str, language: str, threads: int) -> None:
    """视频/音频 → 带时间戳 txt（输入可为文件或目录）"""
    os.makedirs(output, exist_ok=True)
    click.echo(f"输出目录: {output}")

    engine = SenseVoiceEngine(num_threads=threads, language=language)

    if os.path.isdir(input):
        files = sorted(
            f for f in glob.glob(os.path.join(input, "*")) if audio.is_media_file(f)
        )
        if not files:
            click.echo(f"✗ 目录中没有可识别的媒体文件: {input}")
            sys.exit(1)
    else:
        files = [input]

    for f in files:
        _process_one(engine, f, output)
    click.echo("完成 ✓")


def register(cli) -> None:
    cli.add_command(video2txt)
