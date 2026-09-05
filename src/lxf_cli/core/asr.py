"""ASR 引擎封装（当前实现：sherpa-onnx + SenseVoice）。

核心概念：
- 短媒体（<=31s）直接整段识别；
- 长媒体先用 silero-vad 切段，再逐段识别，返回带时间戳的结果。

未来如接入其它引擎（whisper.cpp / FunASR 等），保持
`transcribe(media_path) -> list[Segment]` 这一接口即可。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lxf_cli.core import audio
from lxf_cli.core.models import MODEL_SUBDIR, require_model_dir

SHORT_MEDIA_LIMIT = 31.0  # 秒；留 1s 余量给 SenseVoice 30s 红线
MAX_VAD_SEGMENT = 20      # VAD 单段最大秒数（远低于 30s 上限）


@dataclass
class Segment:
    """一段识别结果"""
    start: float          # 秒
    end: float            # 秒
    text: str


class SenseVoiceEngine:
    """基于 sherpa-onnx + SenseVoice-Small 的离线识别引擎。"""

    def __init__(self, num_threads: int = 4, language: str = "zh"):
        import sherpa_onnx  # 延迟导入，允许未装 asr 依赖时仍可使用其它子命令

        self._sr = audio.TARGET_SR
        model_dir = require_model_dir(MODEL_SUBDIR)
        model = str(model_dir / "model.int8.onnx")
        tokens = str(model_dir / "tokens.txt")
        vad_model = str(model_dir / "silero_vad.onnx")

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model, tokens=tokens, num_threads=num_threads,
            use_itn=True, language=language,
        )

        # VAD 配置
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = vad_model
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = 0.5
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.silero_vad.max_speech_duration = MAX_VAD_SEGMENT
        cfg.sample_rate = self._sr
        cfg.num_threads = 2
        self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)

        self._model_dir: Path = model_dir

    # ---- 内部：整段识别 ----
    def _transcribe_waveform(self, samples: np.ndarray) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self._sr, samples)
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()

    # ---- 内部：VAD 切段识别 ----
    def _transcribe_segmented(self, samples: np.ndarray) -> list[Segment]:
        window_size = self._vad.config.silero_vad.window_size

        self._vad.reset()
        for i in range(0, len(samples), window_size):
            self._vad.accept_waveform(samples[i:i + window_size])
        self._vad.flush()  # 冲刷尾部缓冲，确保最后一段也被吐出

        segs: list[Segment] = []
        while not self._vad.empty():
            seg = self._vad.front          # 属性而非方法
            start = seg.start / self._sr   # start 单位是采样点
            seg_samples = np.asarray(seg.samples, dtype=np.float32)
            if len(seg_samples) == 0:
                self._vad.pop()
                continue
            text = self._transcribe_waveform(seg_samples)
            end = start + len(seg_samples) / self._sr
            if text:
                segs.append(Segment(start, end, text))
            self._vad.pop()
        return segs

    # ---- 对外统一接口 ----
    def transcribe(self, media_path: str) -> list[Segment]:
        """识别单个媒体文件，返回有序 Segment 列表（可能为空）。"""
        wav_path, is_tmp = audio.ensure_16k_mono_wav(media_path)
        try:
            duration = audio.wave_duration(wav_path)
            samples = audio.read_wave(wav_path)
            if duration <= SHORT_MEDIA_LIMIT:
                text = self._transcribe_waveform(samples)
                return [Segment(0.0, duration, text)] if text else []
            return self._transcribe_segmented(samples)
        finally:
            if is_tmp:
                import os
                if os.path.exists(wav_path):
                    os.remove(wav_path)
