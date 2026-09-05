"""模型路径定位。

模型文件较大，不随 pip 包分发；支持三种查找方式（按优先级）：
1. 环境变量 LXF_MODELS_DIR 指向的目录
2. 源码树向上找 models/（开发模式，模型放项目根 models/）
3. ~/.cache/lxf-cli/models/（pip 安装后的用户缓存位置）

每个模型是独立子目录。本模块只负责"找到模型目录"，不负责下载。
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_DIR = "LXF_MODELS_DIR"
MODEL_SUBDIR = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []

    # 1. 环境变量（最高优先级，可指向任意含 models/ 子目录的位置）
    env = os.environ.get(ENV_DIR)
    if env:
        roots.append(Path(env).expanduser())

    # 2. 开发模式：从本文件位置逐级向上找 "models" 目录
    #    src/lxf_cli/core/models.py → 项目根/models 会在第 3 层命中
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        cand = parent / "models"
        if cand.is_dir():
            roots.append(cand)
            break

    # 3. 用户缓存目录（pip 安装后通常落在这里）
    roots.append(Path.home() / ".cache" / "lxf-cli" / "models")

    return roots


def model_dir(name: str = MODEL_SUBDIR) -> Path | None:
    """按优先级返回首个存在的模型目录；找不到返回 None。"""
    for root in _candidate_roots():
        cand = root / name
        if cand.is_dir():
            return cand
    return None


def require_model_dir(name: str = MODEL_SUBDIR) -> Path:
    d = model_dir(name)
    if d is None:
        raise FileNotFoundError(
            f"找不到模型目录 {name!r}。\n"
            "请将模型放入源码树 models/ 下、或设置环境变量 "
            f"{ENV_DIR} 指向含该子目录的目录。"
        )
    return d
