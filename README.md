# RK3588 离线 TTS 交付工程

本项目用于在 RK3588 / A588 开发板上完成离线 TTS 的工程化交付，当前包含：

- 板卡网络初始化脚本
- PaddleSpeech ARM Linux 运行包构建与上板脚本
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
│  └─ release/
└─ artifacts/
   ├─ cache/
   ├─ releases/
   ├─ source-bundles/
   └─ runtime/
```

## 入口说明

- 推荐通过 uv run 直接执行 scripts 目录下的 Python 入口。
- 实际实现位于 scripts/board、scripts/delivery 和 scripts/release。

## Python 环境

- 项目虚拟环境与依赖锁定统一由 uv 管理。
- 首次进入仓库后先执行：

```powershell
uv sync
```

- 后续直接执行 uv run python scripts/... 下的实现脚本，无需手动激活 .venv。

常用命令：

```powershell
uv run python scripts/delivery/prepare_paddlespeech_tts_armlinux.py build
uv run python scripts/delivery/prepare_paddlespeech_tts_armlinux.py upload --source-ip 169.254.46.223
uv run python scripts/board/set_board_static_ipv4.py
./scripts/release/package_release.ps1 -Version v1.0.0 -IncludeRuntimeBundle -IncludeEvidence
```

## 文档位置

- 部署方案：docs/architecture/PaddleSpeech_RK3588_RKNN部署说明.md
- 项目指标：docs/requirements/项目指标.md
- 指标差距：docs/requirements/项目指标差距清单.md
- 选型报告：docs/reports/免费商用离线TTS技术选型详细汇报报告.md
- 交付清单：docs/delivery/交付清单.md
- 部署手册：docs/delivery/部署手册.md
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

配置优先级：

1. 命令行参数
2. 进程环境变量
3. config/local/*.env
4. 非敏感内置默认值

## 产物约定

- 下载缓存：artifacts/cache/
- 离线源码包：artifacts/source-bundles/
- 运行包与回传结果：artifacts/runtime/

当前默认运行包目录：

- artifacts/runtime/paddlespeech_tts_armlinux_runtime/

当前默认冒烟结果目录：

- artifacts/runtime/paddlespeech_tts_armlinux_runtime/output/

当前默认发布目录：

- artifacts/releases/

发布脚本会在发布包根目录额外生成：

- RELEASE_MANIFEST.md
- RELEASE_NOTES.md

默认发布说明模板：

- docs/delivery/发布说明模板.md