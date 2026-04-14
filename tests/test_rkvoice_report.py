from __future__ import annotations

import json
import wave
from pathlib import Path

from scripts.testing.rkvoice_report import build_report, parse_rknn_memory_profile, parse_rknn_perf_text
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

    def _create_workspace(self, workspace_root: Path) -> tuple[Path, Path]:
        runtime_dir = workspace_root / "artifacts" / "runtime" / "rkvoice_runtime"
        asr_runtime_dir = runtime_dir / "asr"
        tts_runtime_dir = runtime_dir / "tts"
        asr_output_dir = asr_runtime_dir / "output"
        tts_output_dir = tts_runtime_dir / "output"

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
            tts_runtime_dir / "run_tts.sh",
            '#!/bin/bash\nexec python3 "$SCRIPT_DIR/melotts_rknn.py" --decoder-model "$SCRIPT_DIR/decoder.rknn" --encoder-model "$SCRIPT_DIR/encoder.onnx"\n',
        )
        self._create_binary(tts_runtime_dir / "decoder.rknn", 2048)
        self._create_binary(tts_runtime_dir / "encoder.onnx", 1024)
        self._create_file(
            asr_runtime_dir / "run_asr.sh",
            '#!/bin/bash\nexec ./bin/sherpa-onnx --provider=rknn --encoder=./models/asr/streaming-rknn/streaming-zipformer-rk3588-small/encoder.rknn --decoder=./models/asr/streaming-rknn/streaming-zipformer-rk3588-small/decoder.rknn --joiner=./models/asr/streaming-rknn/streaming-zipformer-rk3588-small/joiner.rknn --tokens=./models/asr/streaming-rknn/streaming-zipformer-rk3588-small/tokens.txt\n',
        )
        self._create_file(asr_runtime_dir / "bin" / "sherpa-onnx", "binary\n")
        self._create_file(asr_runtime_dir / "bin" / "sherpa-onnx-offline", "binary\n")
        self._create_binary(asr_runtime_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "encoder.rknn", 44 * 1024 * 1024)
        self._create_binary(asr_runtime_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "decoder.rknn", 8 * 1024 * 1024)
        self._create_binary(asr_runtime_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "joiner.rknn", 7 * 1024 * 1024)
        self._create_file(asr_runtime_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "tokens.txt", "tokens\n")
        self._create_file(
            asr_output_dir / "smoke_test_summary.log",
            """[0/5] Board capability snapshot
[1/5] Streaming ASR (RKNN) smoke test
Elapsed seconds: 0.125 s
Real time factor (RTF): 0.125 / 5.592 = 0.022
    {"text":"开放时间早上九点至下午五点","tokens":["开","放","时","间","早","上","九","点","至","下","午","五","点"],"timestamps":[0.0,0.2,0.4,0.6,0.9,1.1,1.3,1.5,1.8,2.1,2.4,2.7,3.0]}
[2/5] Completed
""",
        )
        self._create_file(
            tts_output_dir / "smoke_test_summary.log",
            """[0/5] Board capability snapshot
[1/5] RKNN TTS cold start
load models take 1850.99ms
encoder run take 59.97ms
Sentence[6] Slice[0]: decoder run take 193.74ms
Sentence[11] Slice[1]: decoder run take 191.75ms
Sentence[16] Slice[2]: decoder run take 191.61ms
[2/5] RKNN TTS warm run
load models take 132.40ms
encoder run take 53.40ms
Sentence[6] Slice[0]: decoder run take 174.96ms
Sentence[11] Slice[1]: decoder run take 172.33ms
Sentence[16] Slice[2]: decoder run take 171.01ms
[3/5] RKNN TTS profile run
load models take 140.20ms
encoder run take 54.10ms
Sentence[6] Slice[0]: decoder run take 194.96ms
Sentence[11] Slice[1]: decoder run take 192.33ms
Sentence[16] Slice[2]: decoder run take 194.01ms
[4/5] Completed
""",
        )
        self._create_file(
            tts_output_dir / "profile-samples.csv",
            """elapsed_ms,rss_kb,vm_size_kb,threads,state,utime_ticks,stime_ticks,npu_core0_percent,npu_core1_percent,npu_core2_percent
10,102400,200000,8,R,1,0,63,0,0
60,204800,220000,8,R,2,1,41,0,0
""",
        )
        self._create_file(
            tts_output_dir / "rknn_runtime.log",
            """I RKNN: layer decoder MACs utilization 31% bandwidth occupation 14%
""",
        )
        self._create_file(
            asr_output_dir / "rknn_profile.log",
            """=== 2026-04-12 18:44:15 ===
NPU load:  Core0: 52%, Core1:  0%, Core2:  0%,
=== 2026-04-12 18:44:15 ===
NPU load:  Core0: 48%, Core1:  0%, Core2:  0%,
""",
        )
        self._create_file(
            asr_output_dir / "rknn_eval_perf.txt",
            """===================================================================================================================
                            Performance
===================================================================================================================
ID   OpType           DataType Target InputShape                                   OutputShape            DDR Cycles     NPU Cycles     Total Cycles   Time(us)       MacUsage(%)    WorkLoad(0/1/2)-ImproveTherical        Task Number    Lut Number     RW(KB)         FullName
1    InputOperator    UINT8    CPU    \\                                            (1,3,224,224)          0              0              0              7              \\              0.0%/0.0%/0.0% - Up:0.0%               0              0              147.00         InputOperator:data
2    ConvRelu         UINT8    NPU    (1,3,224,224),(32,3,3,3),(32)                (1,32,112,112)         94150          10584          94150          428            2.47           100.0%/0.0%/0.0% - Up:0.0%             3              0              543.75         conv1

Total Operator Elapsed Time(us): 14147
Total Memory RW Amount(MB): 0
Operator Time-Consuming Ranking:
OpType           Call Number     CPU Time(us)    NPU Time(us)    Total Time(us)    Time Ratio(%)
ConvRelu         36              0               9338            9338              66.0
InputOperator    1               7               0               7                 0.04
===================================================================================================================
""",
        )
        self._create_file(
            asr_output_dir / "rknn_perf_run.json",
            json.dumps({"run_duration_us": 12345}, ensure_ascii=False),
        )
        self._create_file(
            asr_output_dir / "rknn_memory_profile.txt",
            """======================================================
            Memory Profile Info Dump
======================================================
NPU model memory detail(bytes):
    Total Weight Memory: 3.53 MiB
    Total Internal Tensor Memory: 1.67 MiB
    Total Memory: 5.66 MiB

INFO: When evaluating memory usage, we need consider
the size of model, current model size is: 4.08 MiB
======================================================
""",
        )
        self._create_file(
            asr_output_dir / "rknn_runtime.log",
            """I RKNN: layer conv1 MACs utilization 72% bandwidth occupation 31%
I RKNN: layer conv2 MACs utilization 61% bandwidth occupation 28%
""",
        )
        self._create_file(
            asr_output_dir / "board_profile_capabilities.txt",
            """== memory ==
内存：      7.7Gi       914Mi       4.1Gi

== rknn runtime ==
librknnrt version: 2.3.2 (demo)
""",
        )
        self._create_wav(tts_output_dir / "smoke_test_tts.wav")
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
        return runtime_dir, tts_runtime_dir

    def test_build_report_generates_dashboard_json_and_distinguishes_tts_runs(self) -> None:
        with self.temp_dir("rkvoice_report_") as temp_dir:
            workspace_root = temp_dir / "workspace"
            output_root = temp_dir / "reports"
            runtime_dir, _ = self._create_workspace(workspace_root)

            result = build_report(
                workspace_root=workspace_root,
                output_root=output_root,
                runtime_dir=runtime_dir,
                run_unittests=True,
            )

            self.assertTrue(result.html_path.exists())
            self.assertTrue(result.json_path.exists())
            self.assertTrue((result.report_dir / "assets" / "smoke_test_tts.wav").exists())
            self.assertTrue((result.report_dir / "assets" / "asr-rknpu-load-heatmap.svg").exists())
            self.assertTrue((result.report_dir / "assets" / "tts-profile-heatmap.svg").exists())

            payload = json.loads(result.json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["unit_tests"]["total"], 2)
            self.assertEqual(payload["unit_tests"]["passed"], 2)
            self.assertEqual(payload["runtime"]["tts_backend"], "melotts-rknn")
            self.assertEqual(payload["runtime"]["tts_model_name"], "MeloTTS-RKNN2")
            self.assertIsNotNone(payload["runtime"]["tts_decoder_rknn_size_mib"])
            self.assertIsNotNone(payload["runtime"]["tts_encoder_onnx_size_mib"])
            self.assertNotIn("asr_mode", payload["runtime"])
            self.assertEqual(payload["summary"]["verdict"], "fail")
            self.assertEqual(payload["observed"]["rknn_profile_source"], "eval_perf")
            self.assertEqual(payload["observed"]["tts_elapsed_basis"], "warm_run")
            self.assertNotIn("asr_mode", payload["observed"])
            self.assertEqual(payload["plan"]["case_count"], 2)
            self.assertEqual(len(payload["plan"]["cases"]), 2)
            self.assertAlmostEqual(payload["observed"]["tts_cold_start_elapsed_ms"], 637.0, places=3)
            self.assertAlmostEqual(payload["observed"]["tts_warm_run_elapsed_ms"], 572.0, places=3)
            self.assertAlmostEqual(payload["observed"]["tts_profile_run_elapsed_ms"], 635.0, places=3)
            self.assertAlmostEqual(payload["observed"]["tts_elapsed_ms"], 572.0, places=3)
            self.assertEqual(payload["observed"]["asr_streaming_rknn_first_unit_timestamp_ms"], 0.0)
            self.assertAlmostEqual(payload["observed"]["asr_streaming_rknn_per_unit_latency_ms"], 125.0 / 13.0, places=3)
            self.assertEqual(payload["observed"]["asr_streaming_rknn_final_result_latency_ms"], 125.0)
            self.assertEqual(payload["observed"]["asr_streaming_rknn_unit_count"], 13)
            self.assertEqual(payload["observed"]["asr_streaming_rknn_latency_unit"], "字")
            self.assertAlmostEqual(payload["observed"]["tts_max_rss_mib"], 200.0, places=3)
            self.assertEqual(payload["observed"]["npu_peak_source"], "tts_profile_csv")
            self.assertAlmostEqual(payload["observed"]["npu_peak_percent"], 63.0, places=3)
            self.assertEqual(payload["observed"]["rknn_runtime_layer_log_count"], 3)
            self.assertEqual(payload["evidence"]["rknn_perf"]["operator_count"], 2)
            self.assertEqual(payload["evidence"]["rknn_perf"]["ranking"][0]["op_type"], "ConvRelu")
            self.assertAlmostEqual(payload["observed"]["rknn_run_duration_ms"], 12.345, places=3)
            self.assertAlmostEqual(payload["observed"]["rknn_total_memory_mib"], 5.66, places=2)

            requirements = {item["requirement"]: item for item in payload["requirements"]["items"]}
            self.assertEqual(requirements["NPU 加速下，单句中文语音合成端到端延迟 ≤ 150 ms。"]["status"], "fail")
            self.assertIn("warm run", requirements["NPU 加速下，单句中文语音合成端到端延迟 ≤ 150 ms。"]["observed"])
            self.assertEqual(requirements["流式语音识别端到端延迟 ≤ 200 ms。"]["status"], "pass")
            self.assertIn("单字", requirements["流式语音识别端到端延迟 ≤ 200 ms。"]["observed"])
            self.assertIn("首字时间戳", requirements["流式语音识别端到端延迟 ≤ 200 ms。"]["observed"])
            self.assertIn("最终结果耗时", requirements["流式语音识别端到端延迟 ≤ 200 ms。"]["observed"])
            self.assertEqual(requirements["支持流式识别。"]["status"], "pass")
            self.assertEqual(requirements["系统整体全程离线独立工作，无任何外部网络依赖。"]["status"], "pass")
            self.assertEqual(requirements["通信指令、数字、字母发音准确无歧义。"]["status"], "partial")
            self.assertEqual(requirements["支持 RK3588 NPU 硬件加速，CPU 占用可控。"]["status"], "pass")

            html_report = result.html_path.read_text(encoding="utf-8")
            self.assertIn("audio controls", html_report)
            self.assertIn("report-data", html_report)
            self.assertIn("RKVoice", html_report)
            self.assertIn("ASR RKNN NPU Load Heatmap", html_report)
            self.assertIn("ASR \\u9996\\u5b57\\u65f6\\u95f4\\u6233", html_report)
            self.assertIn("ASR \\u6700\\u7ec8\\u7ed3\\u679c\\u8017\\u65f6", html_report)
            self.assertIn("TTS Warm Run", html_report)
            self.assertIn("Toolkit2 eval_perf()", html_report)
            self.assertIn("TTS Profile Heatmap", html_report)
            self.assertIn('"cases": [{"id": "domain-1"', html_report)
            self.assertIn("domain sample", html_report)

    def test_build_report_keeps_tts_cold_start_display_only(self) -> None:
        with self.temp_dir("rkvoice_report_tts_cold_only_") as temp_dir:
            workspace_root = temp_dir / "workspace"
            output_root = temp_dir / "reports"
            runtime_dir, tts_runtime_dir = self._create_workspace(workspace_root)

            self._create_file(
                tts_runtime_dir / "output" / "smoke_test_summary.log",
                """[0/3] Board capability snapshot
[1/3] RKNN TTS cold start
load models take 900.00ms
encoder run take 33.00ms
Sentence[5] Slice[0]: decoder run take 111.00ms
[2/3] Completed
""",
            )
            for optional_name in ("profile-samples.csv", "rknn_runtime.log"):
                optional_path = tts_runtime_dir / "output" / optional_name
                if optional_path.exists():
                    optional_path.unlink()

            result = build_report(
                workspace_root=workspace_root,
                output_root=output_root,
                runtime_dir=runtime_dir,
                run_unittests=False,
            )

            payload = json.loads(result.json_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["observed"]["tts_cold_start_elapsed_ms"], 144.0, places=3)
            self.assertEqual(payload["observed"]["tts_elapsed_basis"], "")
            self.assertIsNone(payload["observed"]["tts_elapsed_ms"])

            requirements = {item["requirement"]: item for item in payload["requirements"]["items"]}
            latency_requirement = requirements["NPU 加速下，单句中文语音合成端到端延迟 ≤ 150 ms。"]
            self.assertEqual(latency_requirement["status"], "unknown")
            self.assertIn("warm/profile", latency_requirement["observed"])

    def test_build_report_falls_back_to_tts_profile_sampling_when_asr_profiling_is_missing(self) -> None:
        with self.temp_dir("rkvoice_report_tts_profile_") as temp_dir:
            workspace_root = temp_dir / "workspace"
            output_root = temp_dir / "reports"
            runtime_dir, _ = self._create_workspace(workspace_root)
            asr_output_dir = runtime_dir / "asr" / "output"

            for artifact_name in (
                "rknn_profile.log",
                "rknn_eval_perf.txt",
                "rknn_perf_run.json",
                "rknn_memory_profile.txt",
                "rknn_runtime.log",
            ):
                (asr_output_dir / artifact_name).unlink()
            self._create_file(
                runtime_dir / "tts" / "output" / "rknn_runtime.log",
                """RKNN Runtime Information, librknnrt version: 2.3.2
RKNN Driver Information, version: 0.9.8
""",
            )

            result = build_report(
                workspace_root=workspace_root,
                output_root=output_root,
                runtime_dir=runtime_dir,
                run_unittests=False,
            )

            payload = json.loads(result.json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["observed"]["rknn_profile_source"], "tts_profile_csv")
            self.assertAlmostEqual(payload["observed"]["tts_max_rss_mib"], 200.0, places=3)
            self.assertEqual(payload["observed"]["npu_peak_source"], "tts_profile_csv")
            self.assertAlmostEqual(payload["observed"]["npu_peak_percent"], 63.0, places=3)
            self.assertEqual(payload["observed"]["rknn_runtime_layer_log_count"], 0)
            self.assertEqual(payload["evidence"]["tts_profile"]["sample_count"], 2)
            self.assertIsNotNone(payload["evidence"]["assets"]["tts_profile_heatmap"])
            requirements = {item["requirement"]: item for item in payload["requirements"]["items"]}
            self.assertEqual(requirements["支持 RK3588 NPU 硬件加速，CPU 占用可控。"]["status"], "pass")

            html_report = result.html_path.read_text(encoding="utf-8")
            self.assertIn("TTS Profile Heatmap", html_report)
            self.assertIn("TTS profile-samples.csv", html_report)

    def test_build_report_ignores_failed_asr_smoke_latency(self) -> None:
        with self.temp_dir("rkvoice_report_asr_fail_") as temp_dir:
            workspace_root = temp_dir / "workspace"
            output_root = temp_dir / "reports"
            runtime_dir, _ = self._create_workspace(workspace_root)
            asr_output_dir = runtime_dir / "asr" / "output"

            self._create_file(
                asr_output_dir / "smoke_test_summary.log",
                """[0/5] Board capability snapshot
[1/5] Streaming ASR (RKNN) smoke test
Elapsed seconds: 0.499 s
Audio duration: 4.176 s
Real time factor (RTF): 0.499 / 4.176 = 0.119
ASR smoke test failed with exit code: 139
[2/5] Completed
""",
            )

            result = build_report(
                workspace_root=workspace_root,
                output_root=output_root,
                runtime_dir=runtime_dir,
                run_unittests=False,
            )

            payload = json.loads(result.json_path.read_text(encoding="utf-8"))
            self.assertIsNone(payload["observed"]["asr_streaming_rknn_elapsed_ms"])
            self.assertIsNone(payload["observed"]["asr_streaming_rknn_first_unit_timestamp_ms"])
            self.assertIsNone(payload["observed"]["asr_streaming_rknn_per_unit_latency_ms"])
            self.assertIsNone(payload["observed"]["asr_streaming_rknn_final_result_latency_ms"])
            self.assertIsNone(payload["observed"]["asr_streaming_rknn_rtf"])
            self.assertTrue(payload["evidence"]["smoke"]["asr_streaming_rknn"]["failed"])
            self.assertEqual(payload["evidence"]["smoke"]["asr_streaming_rknn"]["exit_code"], 139)

            html_report = result.html_path.read_text(encoding="utf-8")
            self.assertIn("Exit Code", html_report)

    def test_parse_rknn_perf_text_supports_toolkit2_232_format(self) -> None:
        with self.temp_dir("rkvoice_perf_text_") as temp_dir:
            perf_path = temp_dir / "rknn_eval_perf.txt"
            memory_path = temp_dir / "rknn_memory_profile.txt"
            self._create_file(
                perf_path,
                """CPU Current Frequency List:
    - 1800000
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                          Network Layer Information Table
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ID   OpType             DataType Target InputShape                               OutputShape            Cycles(DDR/NPU/Total)    Time(us)     MacUsage(%)          WorkLoad(0/1/2)      RW(KB)       FullName
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1    InputOperator      FLOAT16  CPU    \\                                       (1,333,560)            0/0/0                    5                                 0.0%/0.0%/0.0%       0            InputOperator:x
2    Conv               FLOAT16  NPU    (1,560,1,337),(1536,560,1,1),(1536)      (1,1536,1,337)         116785/608256/608256     733          77.24/0.00/0.00      100.0%/0.0%/0.0%     2054         Conv:/encoder/layer0
3    OutputOperator     FLOAT16  CPU    (1,337,25055)                            \\                      0/0/0                    557                               0.0%/0.0%/0.0%       16491        OutputOperator:logits
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Total Operator Elapsed Per Frame Time(us): 785166
Total Memory Read/Write Per Frame Size(KB): 1001510.19
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
                                 Operator Time Consuming Ranking Table
---------------------------------------------------------------------------------------------------
OpType             CallNumber   CPUTime(us)  GPUTime(us)  NPUTime(us)  TotalTime(us)  TimeRatio(%)
---------------------------------------------------------------------------------------------------
Transpose          2            236967       0            220          237187         30.21
Conv               141          0            0            64037        64037          8.16
---------------------------------------------------------------------------------------------------
Total                           237546       0            547620       785166
---------------------------------------------------------------------------------------------------
""",
            )
            self._create_file(
                memory_path,
                """======================================================
            Memory Profile Info Dump
======================================================
NPU model memory detail(bytes):
    Weight Memory: 443.51 MiB
    Internal Tensor Memory: 49.13 MiB
    Other Memory: 18.72 MiB
    Total Memory: 511.36 MiB

INFO: When evaluating memory usage, we need consider
the size of model, current model size is: 463.47 MiB
======================================================
""",
            )

            perf = parse_rknn_perf_text(perf_path)
            memory = parse_rknn_memory_profile(memory_path)

            self.assertEqual(perf["source"], "eval_perf")
            self.assertEqual(perf["operator_count"], 3)
            self.assertEqual(perf["npu_operator_count"], 1)
            self.assertAlmostEqual(perf["summary"]["total_operator_elapsed_time_us"], 785166.0, places=3)
            self.assertAlmostEqual(perf["summary"]["total_memory_rw_mb"], 1001510.19 / 1024.0, places=3)
            self.assertAlmostEqual(perf["summary"]["peak_mac_usage_percent"], 77.24, places=2)
            self.assertEqual(perf["ranking"][0]["op_type"], "Transpose")
            self.assertEqual(perf["ranking"][0]["gpu_time_us"], 0.0)
            self.assertAlmostEqual(memory["total_weight_mib"], 443.51, places=2)
            self.assertAlmostEqual(memory["total_internal_tensor_mib"], 49.13, places=2)
            self.assertAlmostEqual(memory["total_memory_mib"], 511.36, places=2)
            self.assertAlmostEqual(memory["model_size_mib"], 463.47, places=2)

    def test_parse_rknn_perf_text_ignores_empty_profiler_file(self) -> None:
        with self.temp_dir("rkvoice_perf_text_empty_") as temp_dir:
            perf_path = temp_dir / "rknn_eval_perf.txt"
            self._create_file(perf_path, "\n\n")

            perf = parse_rknn_perf_text(perf_path)

            self.assertEqual(perf, {})
