"""音频处理：统一转 16kHz 单声道 wav + 读取。"""
from __future__ import annotations

import subprocess
import tempfile
import wave

import numpy as np

TARGET_SR = 16000
MEDIA_EXTS = (
    ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg",
    ".mp4", ".mov", ".m4v", ".mkv", ".ts", ".webm", ".avi",
)


def is_media_file(path: str) -> bool:
    return path.lower().endswith(MEDIA_EXTS)


def ensure_16k_mono_wav(path: str) -> tuple[str, bool]:
    """任意媒体 → 16kHz 单声道 wav。

    返回 (wav 路径, 是否临时文件)。已是目标格式则原样返回；
    否则用 ffmpeg 转成临时文件（调用方负责删除）。
    """
    if path.lower().endswith(".wav"):
        try:
            with wave.open(path, "rb") as wf:
                if wf.getframerate() == TARGET_SR and wf.getnchannels() == 1:
                    return path, False
        except Exception:
            pass
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", str(TARGET_SR), "-ac", "1", "-f", "wav", tmp],
        check=True, capture_output=True,
    )
    return tmp, True


def read_wave(path: str) -> np.ndarray:
    """读 16k 单声道 wav → float32 [-1,1]"""
    with wave.open(path, "rb") as wf:
        data = wf.readframes(wf.getnframes())
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def wave_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def fmt_ts(sec: float) -> str:
    """mm:ss"""
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"
