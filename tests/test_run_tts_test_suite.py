from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.testing.run_tts_test_suite import (  # noqa: E402
    IterationResult,
    ParsedLog,
    TestCase,
    build_report_payload,
    load_test_plan,
    parse_tts_log,
    render_html_report,
    render_summary_markdown,
    select_cases,
)


SAMPLE_LOG = """WARNING: Logging before InitGoogleLogging() is written to STDERR
[W  4/11 22:26:27.350 ...model_parser.cc:854 LoadModelFbsFromFile]
warning: the version of opt that transformed this model is not consistent with current Paddle-Lite version.
I20260411 22:26:28.215322 258821 main.cc:148] Inference time: 718.381 ms, WAV size (without header): 148800 bytes, WAV duration: 1812.5 ms, RTF: 0.231736
"""

SAMPLE_PROFILE_LOG = """I20260412 00:35:19.800000 389926 front_interface.cpp:159] Key: jieba_dict_path; Value: ./dict/jieba/jieba.dict.utf8
I20260412 00:35:20.135700 389926 main.cc:90] Start to segment sentences by punctuation
I20260412 00:35:20.135963 389926 main.cc:92] Segment sentences through punctuation successfully
I20260412 00:35:20.135977 389926 main.cc:95] Start to get the phoneme and tone id sequence of each sentence
I20260412 00:35:20.137629 389926 main.cc:102] After normalization sentence is: 短波电台指令测试。
I20260412 00:35:20.140767 389926 main.cc:113] Get the phoneme id sequence of each sentence successfully
[I  4/12  0:35:20.141 ...tts/Paddle-Lite/lite/core/device_info.cc:282 get_cpu_arch] Unknow cpu arch: 3339
[W  4/12  0:35:20.142 ...e-Lite/lite/model_parser/model_parser.cc:854 LoadModelFbsFromFile]
[W  4/12  0:35:20.286 ...e-Lite/lite/model_parser/model_parser.cc:854 LoadModelFbsFromFile]
I20260412 00:35:21.035920 389926 main.cc:148] Inference time: 735.755 ms, WAV size (without header): 148800 bytes, WAV duration: 3100 ms, RTF: 0.237340
"""


class ParseLogTests(unittest.TestCase):
    def test_parse_tts_log_extracts_metrics(self) -> None:
        parsed = parse_tts_log(SAMPLE_LOG)
        self.assertIsInstance(parsed, ParsedLog)
        self.assertAlmostEqual(parsed.latency_ms or 0.0, 718.381)
        self.assertAlmostEqual(parsed.rtf or 0.0, 0.231736)
        self.assertEqual(parsed.wav_size_bytes, 148800)
        self.assertEqual(parsed.wav_duration_ms, 1812.5)
        self.assertGreaterEqual(parsed.warning_count, 1)


class PlanTests(unittest.TestCase):
    def test_load_plan_merges_defaults_and_selects_by_tag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plan_path = Path(temp_dir) / "plan.json"
            payload = {
                "name": "demo-suite",
                "defaults": {
                    "repeat": 3,
                    "warmup": 1,
                    "tags": ["baseline"],
                },
                "cases": [
                    {
                        "id": "smoke-hello",
                        "name": "欢迎词",
                        "category": "smoke",
                        "sentence": "你好，欢迎使用离线语音合成服务。",
                        "tags": ["smoke"],
                    },
                    {
                        "id": "latency-short",
                        "name": "短句",
                        "category": "latency",
                        "sentence": "短波电台指令测试。",
                        "repeat": 5,
                        "tags": ["latency"],
                    },
                ],
            }
            plan_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            metadata, cases = load_test_plan(plan_path)
            self.assertEqual(metadata["name"], "demo-suite")
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0].repeat, 3)
            self.assertEqual(cases[0].warmup, 1)
            self.assertIn("baseline", cases[0].tags)
            selected = select_cases(cases, case_ids=set(), categories=set(), tags={"latency"})
            self.assertEqual([item.id for item in selected], ["latency-short"])


class ReportTests(unittest.TestCase):
    def test_build_report_payload_and_renderers(self) -> None:
        report_dir = WORKSPACE_ROOT / "artifacts" / "test-runs" / "unit-test-report"
        (report_dir / "logs").mkdir(parents=True, exist_ok=True)
        (report_dir / "audio").mkdir(parents=True, exist_ok=True)
        case = TestCase(
            id="latency-short",
            name="短句指令",
            sentence="短波电台指令测试。",
            category="latency",
            tags=("latency", "baseline"),
            repeat=2,
            warmup=1,
            latency_threshold_ms=800.0,
            rtf_threshold=0.3,
            max_warning_count=3,
            must_contain=(),
            must_not_contain=(),
            notes="用于验证报表输出",
        )
        results = [
            IterationResult(
                case_id=case.id,
                case_name=case.name,
                category=case.category,
                tags=case.tags,
                iteration=1,
                is_warmup=True,
                passed=True,
                exit_code=0,
                latency_ms=900.0,
                rtf=0.25,
                wav_size_bytes=1000,
                wav_duration_ms=100,
                warning_count=0,
                warning_lines=(),
                error_hints=(),
                failure_reasons=(),
                remote_log_path="/tmp/run-1.log",
                remote_wav_path="/tmp/run-1.wav",
                local_log_path=report_dir / "logs" / "run-1.log",
                local_wav_path=None,
                started_at_utc="2026-04-12T00:00:00+00:00",
                finished_at_utc="2026-04-12T00:00:01+00:00",
                elapsed_wall_ms=1000.0,
            ),
            IterationResult(
                case_id=case.id,
                case_name=case.name,
                category=case.category,
                tags=case.tags,
                iteration=1,
                is_warmup=False,
                passed=True,
                exit_code=0,
                latency_ms=720.0,
                rtf=0.22,
                wav_size_bytes=1000,
                wav_duration_ms=100,
                warning_count=0,
                warning_lines=(),
                error_hints=(),
                failure_reasons=(),
                remote_log_path="/tmp/run-2.log",
                remote_wav_path="/tmp/run-2.wav",
                local_log_path=report_dir / "logs" / "run-2.log",
                local_wav_path=report_dir / "audio" / "run-2.wav",
                started_at_utc="2026-04-12T00:00:02+00:00",
                finished_at_utc="2026-04-12T00:00:03+00:00",
                elapsed_wall_ms=1000.0,
            ),
            IterationResult(
                case_id=case.id,
                case_name=case.name,
                category=case.category,
                tags=case.tags,
                iteration=2,
                is_warmup=False,
                passed=False,
                exit_code=0,
                latency_ms=910.0,
                rtf=0.31,
                wav_size_bytes=1000,
                wav_duration_ms=100,
                warning_count=1,
                warning_lines=("warning",),
                error_hints=(),
                failure_reasons=("延迟 910.000 ms 超过阈值 800.000 ms",),
                remote_log_path="/tmp/run-3.log",
                remote_wav_path="/tmp/run-3.wav",
                local_log_path=report_dir / "logs" / "run-3.log",
                local_wav_path=None,
                started_at_utc="2026-04-12T00:00:04+00:00",
                finished_at_utc="2026-04-12T00:00:05+00:00",
                elapsed_wall_ms=1000.0,
            ),
        ]
        for item in results:
            item.local_log_path.parent.mkdir(parents=True, exist_ok=True)
            item.local_log_path.write_text(SAMPLE_PROFILE_LOG, encoding="utf-8")
        payload = build_report_payload(
            metadata={
                "name": "demo-suite",
                "description": "unit test suite",
                "plan_path": "config/examples/demo.json",
            },
            cases=[case],
            report_dir=report_dir,
            host="169.254.46.2",
            remote_dir="/root/tts/paddlespeech_tts_armlinux_runtime",
            source_ip="169.254.46.223",
            results=results,
            filters={"tags": ["latency"]},
        )
        self.assertEqual(payload["summary"]["cases_total"], 1)
        self.assertEqual(payload["summary"]["cases_passed"], 0)
        self.assertAlmostEqual(payload["summary"]["p95_latency_ms"], 900.5)
        self.assertTrue(payload["profiles"]["available"])
        self.assertIn("aggregate_profile", payload["profiles"])
        markdown = render_summary_markdown(payload)
        html_text = render_html_report(payload)
        self.assertIn("demo-suite", markdown)
        self.assertIn("火焰图可用: 是", markdown)
        self.assertIn("latency-short", markdown)
        self.assertIn("最慢用例 P95 时延", html_text)
        self.assertIn("板端推理火焰图", html_text)
        self.assertIn("代表轮次火焰图", html_text)
        self.assertIn("runtime.model_load", html_text)
        self.assertIn("runtime.inference_prepare", html_text)
        self.assertIn("frontend.build_phone_ids", html_text)
        self.assertIn("runtime.reported_inference", html_text)
        self.assertIn("用例总览", html_text)
        self.assertIn("输入文字 / 音频", html_text)
        self.assertIn("<audio", html_text)



if __name__ == "__main__":
    unittest.main()