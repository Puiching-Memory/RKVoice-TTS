# RK3588 离线 ASR + TTS 交付工程

本项目用于在 RK3588 / A588 开发板上完成离线语音能力的工程化交付，当前默认主线基于 sherpa-onnx，包含：

- sherpa-onnx RK3588 运行包下载、组装、上板与冒烟
- ASR 的 CPU/ONNX 基线与 RKNN/NPU 验证路径
- 中文 TTS 的 CPU/ONNX 基线运行包
- 板卡网络初始化脚本、发布打包脚本与单元测试入口
- 指标、差距、部署说明与选型报告

历史 PaddleSpeech TTSArmLinux 交付链仍保留在 scripts/delivery/paddlespeech_tts_armlinux/，作为中文 CPU 基线与对照路径。

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
│  │  ├─ sherpa_onnx_rk3588/
│  │  └─ templates/
│  ├─ testing/
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
- 默认交付主线位于 scripts/delivery/sherpa_onnx_rk3588/，按 source bundle、runtime assemble、upload/deploy、cli 拆分实现。
- scripts/delivery/paddlespeech_tts_armlinux/ 保留为历史 CPU 基线，不再是默认入口。
- scripts/delivery/templates/ 保存运行包 shell 模板，避免在 Python 入口中内嵌大段板端脚本。

## Python 环境

- 项目虚拟环境与依赖锁定统一由 uv 管理。
- 首次进入仓库后先执行：

```powershell
uv sync
```

- 后续直接执行 uv run python scripts/... 下的实现脚本，或通过 uv run python -m 调用包入口，无需手动激活 .venv。

常用命令：

```powershell
uv run python -m scripts.delivery.sherpa_onnx_rk3588 download
uv run python -m scripts.delivery.sherpa_onnx_rk3588 build
uv run python -m scripts.delivery.sherpa_onnx_rk3588 all --source-ip 169.254.46.223
uv run python -m tests
uv run python -m scripts.testing.rkvoice_report
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

## 综合测试报告

- 统一报告入口会执行 unittest discovery，并汇总当前运行包 output/ 下的 smoke / profile 证据、测试计划 JSON 和 docs/requirements/项目指标.md，输出 HTML 与 JSON 报告。
- 默认报告目录形如 artifacts/test-runs/rkvoice-report-时间戳/，其中包含：

```text
index.html
report.json
assets/
```

- 默认命令：

```powershell
uv run python -m scripts.testing.rkvoice_report
```

- 常用参数：

```powershell
uv run python -m scripts.testing.rkvoice_report --skip-unittests
uv run python -m scripts.testing.rkvoice_report --runtime-dir artifacts/runtime/sherpa_onnx_rk3588_runtime
uv run python -m scripts.testing.rkvoice_report --fail-on-requirement-failures
```

- 报告中的 flame-style heatmap 基于 rknn_profile.log 或 profile-samples.csv 的采样数据绘制，用于工程排查与趋势观察，不等同于 perf 调用栈火焰图。

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
- 新主线默认使用 RKVOICE_* 环境变量；为兼容历史流程，delivery CLI 仍接受旧的 TTS_* 环境变量

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

- artifacts/runtime/sherpa_onnx_rk3588_runtime/

当前默认运行包形态：

- bin/sherpa-onnx-offline
- bin/sherpa-onnx-offline-tts
- lib/libsherpa-onnx-c-api.so
- models/asr/cpu/sense-voice/
- models/asr/rknn/sense-voice-rk3588-20s/
- models/tts/vits-icefall-zh-aishell3/
- run_asr.sh
- run_tts.sh
- smoketest.sh

当前后端支持状态：

- ASR CPU/ONNX：可用
- ASR RKNN/NPU：可用，要求板端提供兼容版本的 librknnrt.so
- TTS CPU/ONNX：可用
- TTS RKNN/NPU：不是当前默认交付目标

默认 TTS 模型仅作为技术基线，商业交付前应再次核验上游模型与数据许可。

当前默认冒烟结果目录：

- artifacts/runtime/sherpa_onnx_rk3588_runtime/output/

当前默认发布目录：

- artifacts/releases/

当前默认测试计划模板：

- config/examples/tts_test_plan.example.json

当前默认测试报告目录：

- artifacts/test-runs/

当前默认综合报告目录形态：

- artifacts/test-runs/rkvoice-report-时间戳/index.html
- artifacts/test-runs/rkvoice-report-时间戳/report.json
- artifacts/test-runs/rkvoice-report-时间戳/assets/

发布脚本会在发布包根目录额外生成：

- RELEASE_MANIFEST.md
- RELEASE_NOTES.md

默认发布说明模板：

- docs/delivery/发布说明模板.md