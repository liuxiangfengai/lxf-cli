"""lxf-cli 入口：命令组 + 子命令自动发现。

新增子命令：在 lxf_cli/commands/ 下新建 <名字>.py，
文件内定义 `register(cli)` 函数（见 video2txt.py 示例），
即可自动挂载，无需改动本文件。
"""
from __future__ import annotations

import importlib
import pkgutil

import click

from lxf_cli import commands


def _discover_commands(cli: click.Group) -> None:
    """扫描 commands 包内所有模块，调用其 register(cli)。"""
    for mod in pkgutil.iter_modules(commands.__path__):
        module = importlib.import_module(f"lxf_cli.commands.{mod.name}")
        if hasattr(module, "register"):
            module.register(cli)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="lxf-cli", prog_name="lxf-cli")
def cli() -> None:
    """lxf 本地 AI 命令行工具集。"""


def main() -> None:
    _discover_commands(cli)
    cli()


if __name__ == "__main__":
    main()
