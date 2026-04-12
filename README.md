# RK3588 离线 TTS 交付工程

本项目用于在 RK3588 / A588 开发板上完成离线 TTS 的工程化交付，当前包含：

- 板卡网络初始化脚本
- PaddleSpeech ARM Linux 运行包构建与上板脚本
- 单元测试入口与历史测试产物目录
- 指标、差距、部署说明与选型报告
- 本地构建产物与板端冒烟回传结果

## 目录结构

```text
.
├─ README.md
├─ pyproject.toml
├─ uv.lock
├─ config/
│  ├─ examples/
│  └─ local/
├─ docs/
│  ├─ architecture/
│  ├─ delivery/
│  ├─ reports/
│  └─ requirements/
├─ scripts/
│  ├─ board/
│  ├─ delivery/
│  │  ├─ paddlespeech_tts_armlinux/
│  │  └─ templates/
│  └─ release/
└─ artifacts/
   ├─ cache/
   ├─ releases/
   ├─ source-bundles/
│  └─ test-runs/
   └─ runtime/
```

## 入口说明

- 推荐通过 uv run 执行 Python 入口；对于包化的 delivery CLI，推荐使用 python -m 形式。
- 实际实现位于 scripts/board、scripts/delivery 和 scripts/release。
- scripts/delivery/paddlespeech_tts_armlinux/ 按 source bundle、runtime build、upload/deploy、cli 拆分了 delivery 实现。
- scripts/delivery/templates/ 保存运行包、SDK、CMake 与 shell 生成模板，避免在 Python 入口中内嵌大量原生源码。

## Python 环境

- 项目虚拟环境与依赖锁定统一由 uv 管理。
- 首次进入仓库后先执行：

```powershell
uv sync
```

- 后续直接执行 uv run python scripts/... 下的实现脚本，或通过 uv run python -m 调用包入口，无需手动激活 .venv。

常用命令：

```powershell
uv run python -m scripts.delivery.paddlespeech_tts_armlinux build
uv run python -m scripts.delivery.paddlespeech_tts_armlinux upload --source-ip 169.254.46.223
uv run python -m tests
uv run python scripts/board/set_board_static_ipv4.py
uv run python scripts/release/package_release.py --version v1.0.0 --include-runtime-bundle --include-evidence
```

## 单元测试

- tests/ 下统一维护 scripts/ 目录 Python 入口的单元测试。
- 全量单测统一通过包入口执行，避免每个测试文件重复维护独立启动样板。

```powershell
uv run python -m tests
uv run python -m unittest tests.test_package_release
```

## 文档位置

- 部署方案：docs/architecture/PaddleSpeech_RK3588_RKNN部署说明.md
- 项目指标：docs/requirements/项目指标.md
- 指标差距：docs/requirements/项目指标差距清单.md
- 选型报告：docs/reports/免费商用离线TTS技术选型详细汇报报告.md
- 交付清单：docs/delivery/交付清单.md
- 部署手册：docs/delivery/部署手册.md
- 测试手册：docs/delivery/测试手册.md
- 验收手册：docs/delivery/验收手册.md
- 回退手册：docs/delivery/回退手册.md
- 发布手册：docs/delivery/发布手册.md

## 配置约定

- 本地敏感连接信息放在 config/local/
- 示例模板放在 config/examples/
- 不建议把板卡密码、源地址等信息写入可交付文档

推荐配置文件：

- config/local/board.local.env
- config/local/delivery.local.env
- config/local/tts_test_plan.json

配置优先级：

1. 命令行参数
2. 进程环境变量
3. config/local/*.env
4. 非敏感内置默认值

## 产物约定

- 下载缓存：artifacts/cache/
- 离线源码包：artifacts/source-bundles/
- 运行包与回传结果：artifacts/runtime/
- 批量测试报告：artifacts/test-runs/

当前默认运行包目录：

- artifacts/runtime/paddlespeech_tts_armlinux_runtime/

当前运行包内的二进制 SDK 形态：

- bin/rkvoice_tts_demo
- bin/paddlespeech_tts_demo
- lib/librkvoice_tts.so
- lib/librkvoice_tts.a
- include/rkvoice_tts_api.h
- examples/c_api_demo.c

当前后端支持状态：

- CPU 后端：可用
- RKNN 后端：预留接口，当前运行包尚未编译进 NPU 推理实现

当前默认冒烟结果目录：

- artifacts/runtime/paddlespeech_tts_armlinux_runtime/output/

当前默认发布目录：

- artifacts/releases/

当前默认测试计划模板：

- config/examples/tts_test_plan.example.json

当前默认测试报告目录：

- artifacts/test-runs/

发布脚本会在发布包根目录额外生成：

- RELEASE_MANIFEST.md
- RELEASE_NOTES.md

默认发布说明模板：

- docs/delivery/发布说明模板.md