# RK3588 离线 ASR + TTS 交付工程

本项目用于在 RK3588 / A588 开发板上完成离线语音能力的工程化交付，包含两条主线：

- ASR：基于 sherpa-onnx，支持 CPU/ONNX 基线、RKNN/NPU 验证路径与流式识别
- TTS：基于 MeloTTS-RKNN2，Python + RKNN 离线中文语音合成
- 板卡网络初始化脚本、发布打包脚本与单元测试入口
- 指标、差距、部署说明与选型报告

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
│  │  ├─ melotts_rknn2/
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
- ASR 和 TTS 交付管线统一位于 scripts/delivery/，通过 `asr` / `tts` 子命令区分管线，共享 config、remote、shared 基础设施。
- ASR 管线 (sherpa-onnx RK3588) 按 source bundle → 本地 RKNN 导出 → runtime assemble → upload/deploy 流程实现。
- TTS 管线 (MeloTTS-RKNN2) 提供 Python + RKNN TTS 运行包，build 时准备离线 wheelhouse，upload/all 时在板端自动安装到运行包内的 pydeps/。
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
uv run python -m scripts.delivery asr download
uv run python -m scripts.delivery asr build
uv run python -m scripts.delivery asr all --source-ip 169.254.46.223
uv run python -m scripts.delivery tts download
uv run python -m scripts.delivery tts build
uv run python -m scripts.delivery tts all --source-ip 169.254.46.223
uv run python -m tests
uv run python -m scripts.testing.rkvoice_report
uv run python scripts/board/prepare_rknn_debug_bridge.py
uv run python scripts/testing/rknn_toolkit2_profile_in_docker.py --prepare-board-debug-bridge
uv run python scripts/board/set_board_static_ipv4.py
uv run python scripts/release/package_release.py --version v1.0.0 --include-runtime-bundle --include-evidence
uv run python scripts/release/package_release_in_docker.py --version v1.0.0 --include-runtime-bundle --include-evidence
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
uv run python -m scripts.testing.rkvoice_report --runtime-dir artifacts/runtime/rkvoice_runtime
uv run python -m scripts.testing.rkvoice_report --fail-on-requirement-failures
```

- 报告会优先解析 output/ 下的 rknn_eval_perf.txt、rknn_query_perf_detail.txt、rknn_perf_run.json、rknn_memory_profile.txt 等官方 RKNN profiler 证据；若缺失，再回退到 rknn_runtime.log 与 rknpu/load、profile-samples.csv 生成的时序热力图。
- 热力图仅用于观察运行期负载与进程采样趋势，不等同于 perf 调用栈火焰图。

## 文档位置

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
- runtime 根目录默认使用 RKVOICE_* 环境变量；TTS 仍保留 RKVOICE_MELO_* 变量用于源码包和测试文本配置

推荐配置文件：

- config/local/board.local.env
- config/local/delivery.local.env
- config/local/tts_test_plan.json

Unified runtime 根目录优先使用以下名字：

- RKVOICE_RUNTIME_DIR
- RKVOICE_REMOTE_DIR

MeloTTS-RKNN2 仍保留以下专用变量用于源码包与测试文本：

- RKVOICE_MELO_STAGE_DIR
- RKVOICE_MELO_TTS_TEXT

MeloTTS-RKNN2 额外支持以下运行期变量：

- RKVOICE_PYTHON_BIN
- RKVOICE_PYDEPS_DIR
- RKVOICE_PYTHON_DEPS_LOG

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

- artifacts/runtime/rkvoice_runtime/
- artifacts/runtime/rkvoice_runtime/asr/ （ASR）
- artifacts/runtime/rkvoice_runtime/tts/ （TTS）

当前 ASR 运行包形态：

- bin/sherpa-onnx
- lib/libsherpa-onnx-c-api.so
- models/asr/streaming-rknn/streaming-zipformer-rk3588-small/
- run_asr.sh
- smoketest.sh
- tools/profile_asr_inference.sh

当前后端支持状态：

- ASR streaming RKNN/NPU：默认交付入口；build 时基于上游 ONNX 源模型在本机或 Docker 中导出 RKNN，板端要求提供兼容版本的 librknnrt.so
- TTS MeloTTS-RKNN2：交付主线，要求板端提供 python3；运行包会自带离线 wheelhouse 并在 upload/all 时自动安装 onnxruntime、soundfile、cn2an、inflect 和 rknn-toolkit-lite2 到 pydeps/

MeloTTS-RKNN2 上游镜像当前标注为 AGPL-3.0，商业交付前应完成法务评估。

当前默认冒烟结果目录：

- artifacts/runtime/rkvoice_runtime/asr/output/
- artifacts/runtime/rkvoice_runtime/tts/output/

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