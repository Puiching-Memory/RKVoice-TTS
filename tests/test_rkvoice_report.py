from __future__ import annotations

import json
import wave
from pathlib import Path

from scripts.testing.rkvoice_report import build_report
from tests.test_support import WorkspaceTestCase


class RKVoiceReportTests(WorkspaceTestCase):
    def _create_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _create_binary(self, path: Path, size_bytes: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"0" * size_bytes)

    def _create_wav(self, path: Path, *, frame_rate: int = 16000, duration_ms: int = 250) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame_count = int(frame_rate * duration_ms / 1000)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(frame_rate)
            wav_file.writeframes(b"\x00\x00" * frame_count)

    def _create_workspace(self, workspace_root: Path) -> Path:
        runtime_dir = workspace_root / "artifacts" / "runtime" / "sherpa_onnx_rk3588_runtime"
        output_dir = runtime_dir / "output"

        self._create_file(
            workspace_root / "docs" / "requirements" / "项目指标.md",
            """# 项目指标

## 1. 语音合成 TTS 指标

### 1.1 端侧合成延迟

- NPU 加速下，单句中文语音合成端到端延迟 ≤ 150 ms。

### 1.2 合成准确率

- 通信指令、数字、字母发音准确无歧义。

### 1.3 模型与稳定性

- 支持 7×24 小时连续稳定运行。

### 1.4 运行模式

- 纯离线端侧运行，无需网络、无需云端接口。

## 2. 语音识别 ASR 指标

### 2.1 端侧识别延迟

- 流式语音识别端到端延迟 ≤ 200 ms。

### 2.3 模型与稳定性

- 支持流式识别。

## 3. 系统整体指标

- 支持 RK3588 NPU 硬件加速，CPU 占用可控。
- 系统整体全程离线独立工作，无任何外部网络依赖。
""",
        )
        self._create_file(
            workspace_root / "config" / "examples" / "tts_test_plan.example.json",
            json.dumps(
                {
                    "name": "demo-plan",
                    "description": "report test plan",
                    "defaults": {"repeat": 3},
                    "cases": [
                        {"id": "domain-1", "name": "domain sample", "category": "domain", "sentence": "频点播报"},
                        {"id": "stability-1", "name": "stability sample", "category": "stability", "sentence": "长稳测试"},
                    ],
                },
                ensure_ascii=False,
            ),
        )
        self._create_file(
            runtime_dir / "run_tts.sh",
            "#!/bin/bash\nexec ./bin/sherpa-onnx-offline-tts --vits-model=./models/tts/vits-icefall-zh-aishell3/model.onnx\n",
        )
        self._create_file(
            runtime_dir / "run_asr.sh",
            "#!/bin/bash\nexec ./bin/sherpa-onnx-offline --provider=rknn ./models/asr/rknn/sense-voice-rk3588-20s/test_wavs/zh.wav\n",
        )
        self._create_file(runtime_dir / "bin" / "sherpa-onnx-offline", "binary\n")
        self._create_file(runtime_dir / "bin" / "sherpa-onnx-offline-tts", "binary\n")
        self._create_binary(runtime_dir / "models" / "tts" / "vits-icefall-zh-aishell3" / "model.onnx", 1024)
        self._create_binary(runtime_dir / "models" / "asr" / "cpu" / "sense-voice" / "model.int8.onnx", 1024)
        self._create_binary(runtime_dir / "models" / "asr" / "rknn" / "sense-voice-rk3588-20s" / "model.rknn", 1024)
        self._create_file(
            output_dir / "smoke_test_summary.log",
            """[0/3] Board capability snapshot
[1/3] CPU TTS smoke test
Elapsed seconds: 0.353 s
Audio duration: 3.030 s
Real-time factor (RTF): 0.353/3.030 = 0.117
The text is: 你好，欢迎使用离线语音合成服务。. Speaker ID: 66
[2/3] CPU ASR smoke test
Elapsed seconds: 0.343 s
Real time factor (RTF): 0.343 / 5.592 = 0.061
[3/3] RKNN ASR smoke test
Elapsed seconds: 0.855 s
Real time factor (RTF): 0.855 / 5.592 = 0.153
{"text":"开放时间早上九点至下午五点"}
""",
        )
        self._create_file(
            output_dir / "rknn_profile.log",
            """=== 2026-04-12 18:44:15 ===
NPU load:  Core0: 52%, Core1:  0%, Core2:  0%,
=== 2026-04-12 18:44:15 ===
NPU load:  Core0: 48%, Core1:  0%, Core2:  0%,
""",
        )
        self._create_file(
            output_dir / "board_profile_capabilities.txt",
            """== memory ==
内存：      7.7Gi       914Mi       4.1Gi

== rknn runtime ==
librknnrt version: 2.3.2 (demo)
""",
        )
        self._create_wav(output_dir / "smoke_test_tts.wav")
        self._create_file(
            workspace_root / "tests" / "test_demo.py",
            """import unittest


class DemoTests(unittest.TestCase):
    def test_alpha(self):
        self.assertTrue(True)

    def test_beta(self):
        self.assertEqual(1 + 1, 2)
""",
        )
        self._create_file(workspace_root / "tests" / "__init__.py", "")
        return runtime_dir

    def test_build_report_generates_dashboard_json_audio_and_heatmap(self) -> None:
        with self.temp_dir("rkvoice_report_") as temp_dir:
            workspace_root = temp_dir / "workspace"
            output_root = temp_dir / "reports"
            runtime_dir = self._create_workspace(workspace_root)

            result = build_report(
                workspace_root=workspace_root,
                output_root=output_root,
                runtime_dir=runtime_dir,
                run_unittests=True,
            )

            self.assertTrue(result.html_path.exists())
            self.assertTrue(result.json_path.exists())
            self.assertTrue((result.report_dir / "assets" / "smoke_test_tts.wav").exists())
            self.assertTrue((result.report_dir / "assets" / "asr-rknn-heatmap.svg").exists())

            payload = json.loads(result.json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["unit_tests"]["total"], 2)
            self.assertEqual(payload["unit_tests"]["passed"], 2)
            self.assertEqual(payload["summary"]["verdict"], "fail")

            requirements = {item["requirement"]: item for item in payload["requirements"]["items"]}
            self.assertEqual(requirements["NPU 加速下，单句中文语音合成端到端延迟 ≤ 150 ms。"]["status"], "fail")
            self.assertEqual(requirements["支持流式识别。"]["status"], "fail")
            self.assertEqual(requirements["系统整体全程离线独立工作，无任何外部网络依赖。"]["status"], "pass")
            self.assertEqual(requirements["通信指令、数字、字母发音准确无歧义。"]["status"], "partial")

            html_report = result.html_path.read_text(encoding="utf-8")
            self.assertIn("audio controls", html_report)
            self.assertIn("ASR RKNN Flame-Style Heatmap", html_report)