from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import shutil
import sys
import unittest
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "artifacts" / "test-runs"
DEFAULT_RUNTIME_DIR = WORKSPACE_ROOT / "artifacts" / "runtime" / "sherpa_onnx_rk3588_runtime"
DEFAULT_REQUIREMENTS_PATH = WORKSPACE_ROOT / "docs" / "requirements" / "项目指标.md"
DEFAULT_LOCAL_PLAN_PATH = WORKSPACE_ROOT / "config" / "local" / "tts_test_plan.json"
DEFAULT_EXAMPLE_PLAN_PATH = WORKSPACE_ROOT / "config" / "examples" / "tts_test_plan.example.json"

ELAPSED_SECONDS_PATTERN = re.compile(r"Elapsed seconds:\s*([0-9.]+)\s*s")
AUDIO_DURATION_PATTERN = re.compile(r"Audio duration:\s*([0-9.]+)\s*s")
RTF_PATTERN = re.compile(r"Real[- ]time factor(?: \(RTF\))?:.*?=\s*([0-9.]+)")
TEXT_PATTERN = re.compile(r"The text is:\s*(.+?)\.\s*Speaker ID:")
CORE_LOAD_PATTERN = re.compile(r"Core([012]):\s*([0-9]+)%")
RKNN_VERSION_PATTERN = re.compile(r"librknnrt version:\s*(.+)")
MEMORY_TOTAL_PATTERN = re.compile(r"内存：\s*([0-9.]+)([GMK]i)\b")


class ReportBuildError(Exception):
    pass


@dataclass(frozen=True)
class TestCaseRecord:
    test_id: str
    status: str
    duration_s: float
    details: str = ""


@dataclass(frozen=True)
class UnittestSummary:
    total: int
    passed: int
    failed: int
    errors: int
    skipped: int
    expected_failures: int
    unexpected_successes: int
    duration_s: float
    success: bool
    output_text: str
    cases: tuple[TestCaseRecord, ...]


@dataclass(frozen=True)
class ReportBuildResult:
    report_dir: Path
    html_path: Path
    json_path: Path
    unittest_success: bool
    requirement_failures: int


class CollectingTextTestResult(unittest.TextTestResult):
    def __init__(self, stream: io.StringIO, descriptions: bool, verbosity: int) -> None:
        super().__init__(stream, descriptions, verbosity)
        self.case_records: list[TestCaseRecord] = []
        self._started_at: dict[int, float] = {}

    def startTest(self, test: unittest.case.TestCase) -> None:
        self._started_at[id(test)] = perf_counter()
        super().startTest(test)

    def addSuccess(self, test: unittest.case.TestCase) -> None:
        super().addSuccess(test)
        self._record(test, "passed")

    def addFailure(self, test: unittest.case.TestCase, err: Any) -> None:
        super().addFailure(test, err)
        self._record(test, "failed", err)

    def addError(self, test: unittest.case.TestCase, err: Any) -> None:
        super().addError(test, err)
        self._record(test, "error", err)

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        self._record(test, "skipped", details=reason)

    def addExpectedFailure(self, test: unittest.case.TestCase, err: Any) -> None:
        super().addExpectedFailure(test, err)
        self._record(test, "expected-failure", err)

    def addUnexpectedSuccess(self, test: unittest.case.TestCase) -> None:
        super().addUnexpectedSuccess(test)
        self._record(test, "unexpected-success")

    def _record(
        self,
        test: unittest.case.TestCase,
        status: str,
        err: Any | None = None,
        details: str = "",
    ) -> None:
        started_at = self._started_at.pop(id(test), perf_counter())
        rendered_details = details.rstrip()
        if err is not None:
            rendered_details = self._exc_info_to_string(err, test).rstrip()
        self.case_records.append(
            TestCaseRecord(
                test_id=str(test),
                status=status,
                duration_s=max(perf_counter() - started_at, 0.0),
                details=rendered_details,
            )
        )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def coerce_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bytes_to_mib(size_bytes: int | float | None) -> float | None:
    if size_bytes is None:
        return None
    return round(float(size_bytes) / (1024.0 * 1024.0), 3)


def format_number(value: float | None, *, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def format_status(status: str) -> str:
    labels = {
        "pass": "通过",
        "fail": "未通过",
        "partial": "部分满足",
        "unknown": "证据不足",
        "passed": "通过",
        "failed": "失败",
        "error": "错误",
        "skipped": "跳过",
        "expected-failure": "预期失败",
        "unexpected-success": "意外通过",
    }
    return labels.get(status, status)


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", value.strip())
    return cleaned.strip("-") or "artifact"


def relative_posix(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def file_size_bytes(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    return path.stat().st_size


def detect_tts_backend(run_tts_script: str) -> str:
    script_lower = run_tts_script.lower()
    if ".rknn" in script_lower or "provider=rknn" in script_lower:
        return "rknn"
    if "onnx" in script_lower or "sherpa-onnx-offline-tts" in script_lower:
        return "cpu-onnx"
    return "unknown"


def detect_asr_mode(run_asr_script: str) -> str:
    script_lower = run_asr_script.lower()
    if "stream" in script_lower and "offline" not in script_lower:
        return "streaming"
    if "offline" in script_lower:
        return "offline"
    return "unknown"


def maybe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def inspect_runtime(runtime_dir: Path) -> dict[str, Any]:
    run_tts_path = runtime_dir / "run_tts.sh"
    run_asr_path = runtime_dir / "run_asr.sh"
    run_tts_script = maybe_read_text(run_tts_path)
    run_asr_script = maybe_read_text(run_asr_path)

    tts_model_path = runtime_dir / "models" / "tts" / "vits-icefall-zh-aishell3" / "model.onnx"
    asr_cpu_model_path = runtime_dir / "models" / "asr" / "cpu" / "sense-voice" / "model.int8.onnx"
    asr_rknn_model_path = runtime_dir / "models" / "asr" / "rknn" / "sense-voice-rk3588-20s" / "model.rknn"

    return {
        "runtime_dir": str(runtime_dir),
        "tts_backend": detect_tts_backend(run_tts_script),
        "tts_supports_rknn": ".rknn" in run_tts_script.lower() or "provider=rknn" in run_tts_script.lower(),
        "asr_mode": detect_asr_mode(run_asr_script),
        "asr_supports_rknn": "provider=rknn" in run_asr_script.lower(),
        "tts_model_is_int8": "int8" in tts_model_path.name.lower(),
        "asr_cpu_model_is_int8": "int8" in asr_cpu_model_path.name.lower(),
        "tts_model_size_mib": bytes_to_mib(file_size_bytes(tts_model_path)),
        "asr_cpu_model_size_mib": bytes_to_mib(file_size_bytes(asr_cpu_model_path)),
        "asr_rknn_model_size_mib": bytes_to_mib(file_size_bytes(asr_rknn_model_path)),
        "models_total_size_mib": bytes_to_mib(directory_size_bytes(runtime_dir / "models")),
        "offline_ready": (runtime_dir / "bin").exists() and (runtime_dir / "models").exists(),
        "tts_model_name": tts_model_path.parent.name if tts_model_path.parent.exists() else "",
        "asr_cpu_model_name": asr_cpu_model_path.parent.name if asr_cpu_model_path.parent.exists() else "",
        "asr_rknn_model_name": asr_rknn_model_path.parent.name if asr_rknn_model_path.parent.exists() else "",
    }


def pick_first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def first_glob(path: Path, pattern: str) -> Path | None:
    for candidate in sorted(path.glob(pattern)):
        if candidate.exists():
            return candidate
    return None


def parse_smoke_log(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    sections: dict[str, dict[str, Any]] = {
        "tts": {"label": "CPU TTS smoke"},
        "asr_cpu": {"label": "CPU ASR smoke"},
        "asr_rknn": {"label": "RKNN ASR smoke"},
    }
    current_section: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[1/3] CPU TTS smoke test"):
            current_section = "tts"
            continue
        if line.startswith("[2/3] CPU ASR smoke test"):
            current_section = "asr_cpu"
            continue
        if line.startswith("[3/3] RKNN ASR smoke test"):
            current_section = "asr_rknn"
            continue
        if current_section is None:
            continue

        section = sections[current_section]
        elapsed_match = ELAPSED_SECONDS_PATTERN.search(line)
        if elapsed_match:
            section["elapsed_seconds"] = float(elapsed_match.group(1))
            continue
        audio_match = AUDIO_DURATION_PATTERN.search(line)
        if audio_match:
            section["audio_duration_seconds"] = float(audio_match.group(1))
            continue
        rtf_match = RTF_PATTERN.search(line)
        if rtf_match:
            section["rtf"] = float(rtf_match.group(1))
            continue
        text_match = TEXT_PATTERN.search(line)
        if text_match:
            section["text"] = text_match.group(1).strip()
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                section["result"] = json.loads(line)
            except json.JSONDecodeError:
                pass

    return sections


def parse_rknn_profile_log(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    current_timestamp = ""
    samples: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("===") and line.endswith("==="):
            current_timestamp = line.strip("= ")
            continue
        if "NPU load:" not in line:
            continue

        sample = {
            "timestamp": current_timestamp,
            "core0_percent": 0,
            "core1_percent": 0,
            "core2_percent": 0,
        }
        for core_index, percent in CORE_LOAD_PATTERN.findall(line):
            sample[f"core{core_index}_percent"] = int(percent)
        samples.append(sample)

    peak_percent = max((max(sample["core0_percent"], sample["core1_percent"], sample["core2_percent"]) for sample in samples), default=0)
    mean_percent = 0.0
    if samples:
        mean_percent = sum(
            max(sample["core0_percent"], sample["core1_percent"], sample["core2_percent"])
            for sample in samples
        ) / len(samples)

    return {
        "sample_count": len(samples),
        "peak_percent": peak_percent,
        "mean_percent": round(mean_percent, 2),
        "samples": samples,
    }


def parse_board_capabilities(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    content = path.read_text(encoding="utf-8")
    rknn_version_match = RKNN_VERSION_PATTERN.search(content)
    memory_total_match = MEMORY_TOTAL_PATTERN.search(content)

    board_info: dict[str, Any] = {
        "rknn_runtime_version": rknn_version_match.group(1).strip() if rknn_version_match else "",
        "memory_total": "",
    }
    if memory_total_match:
        board_info["memory_total"] = f"{memory_total_match.group(1)}{memory_total_match.group(2)}"
    return board_info


def parse_tts_profile_csv(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            samples.append(
                {
                    "elapsed_ms": int(float(row.get("elapsed_ms", "0") or 0)),
                    "rss_kb": int(float(row.get("rss_kb", "0") or 0)),
                    "vm_size_kb": int(float(row.get("vm_size_kb", "0") or 0)),
                    "threads": int(float(row.get("threads", "0") or 0)),
                    "utime_ticks": int(float(row.get("utime_ticks", "0") or 0)),
                    "stime_ticks": int(float(row.get("stime_ticks", "0") or 0)),
                    "npu_core0_percent": int(float(row.get("npu_core0_percent", "0") or 0)),
                    "npu_core1_percent": int(float(row.get("npu_core1_percent", "0") or 0)),
                    "npu_core2_percent": int(float(row.get("npu_core2_percent", "0") or 0)),
                }
            )

    max_rss_kb = max((sample["rss_kb"] for sample in samples), default=0)
    max_threads = max((sample["threads"] for sample in samples), default=0)
    peak_npu_percent = max(
        (
            max(sample["npu_core0_percent"], sample["npu_core1_percent"], sample["npu_core2_percent"])
            for sample in samples
        ),
        default=0,
    )
    return {
        "sample_count": len(samples),
        "duration_ms": max((sample["elapsed_ms"] for sample in samples), default=0),
        "max_rss_mib": bytes_to_mib(max_rss_kb * 1024),
        "max_threads": max_threads,
        "peak_npu_percent": peak_npu_percent,
        "samples": samples,
    }


def read_wav_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            duration_s = frame_count / frame_rate if frame_rate else 0.0
            return {
                "channels": wav_file.getnchannels(),
                "sample_width_bytes": wav_file.getsampwidth(),
                "frame_rate": frame_rate,
                "frame_count": frame_count,
                "duration_s": round(duration_s, 3),
                "size_mib": bytes_to_mib(path.stat().st_size),
            }
    except wave.Error:
        return {"size_mib": bytes_to_mib(path.stat().st_size)}


def resolve_plan_path(workspace_root: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise ReportBuildError(f"测试计划不存在：{explicit_path}")
        return explicit_path

    local_path = workspace_root / "config" / "local" / "tts_test_plan.json"
    if local_path.exists():
        return local_path

    example_path = workspace_root / "config" / "examples" / "tts_test_plan.example.json"
    if example_path.exists():
        return example_path
    return None


def load_plan_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    category_counts: dict[str, int] = {}
    acceptance_cases: list[dict[str, Any]] = []
    for case in cases:
        category = str(case.get("category", "uncategorized"))
        category_counts[category] = category_counts.get(category, 0) + 1
        if category == "acceptance":
            acceptance_cases.append(
                {
                    "id": case.get("id", ""),
                    "name": case.get("name", ""),
                    "latency_threshold_ms": case.get("latency_threshold_ms"),
                    "rtf_threshold": case.get("rtf_threshold"),
                    "notes": case.get("notes", ""),
                }
            )

    return {
        "path": str(path),
        "name": payload.get("name", ""),
        "description": payload.get("description", ""),
        "case_count": len(cases),
        "category_counts": category_counts,
        "acceptance_cases": acceptance_cases,
        "defaults": payload.get("defaults", {}),
    }


def parse_requirements(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ReportBuildError(f"项目指标文档不存在：{path}")

    section_title = ""
    subsection_title = ""
    items: list[dict[str, str]] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            section_title = line[3:].strip()
            subsection_title = ""
            continue
        if line.startswith("### "):
            subsection_title = line[4:].strip()
            continue
        if line.startswith("- "):
            items.append(
                {
                    "section": section_title,
                    "subsection": subsection_title,
                    "text": line[2:].strip(),
                }
            )
    return items


@contextmanager
def prepend_sys_path(path: Path):
    path_text = str(path)
    original = list(sys.path)
    if path_text in sys.path:
        sys.path.remove(path_text)
    sys.path.insert(0, path_text)
    try:
        yield
    finally:
        sys.path[:] = original


@contextmanager
def isolated_module_prefix(prefix: str):
    preserved = {
        name: module
        for name, module in sys.modules.items()
        if name == prefix or name.startswith(f"{prefix}.")
    }
    for name in list(preserved):
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(f"{prefix}."):
                sys.modules.pop(name, None)
        sys.modules.update(preserved)


def run_unittest_suite(workspace_root: Path, pattern: str) -> UnittestSummary:
    tests_dir = workspace_root / "tests"
    if not tests_dir.exists():
        raise ReportBuildError(f"单元测试目录不存在：{tests_dir}")

    stream = io.StringIO()
    with prepend_sys_path(workspace_root), isolated_module_prefix("tests"):
        suite = unittest.defaultTestLoader.discover(
            start_dir=str(tests_dir),
            pattern=pattern,
            top_level_dir=str(workspace_root),
        )
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=2,
            resultclass=CollectingTextTestResult,
            buffer=True,
        )
        started_at = perf_counter()
        result = runner.run(suite)
        duration_s = perf_counter() - started_at

    cases = tuple(result.case_records)
    failed = len(result.failures) + len(result.unexpectedSuccesses)
    passed = sum(1 for case in cases if case.status == "passed")
    return UnittestSummary(
        total=len(cases),
        passed=passed,
        failed=failed,
        errors=len(result.errors),
        skipped=len(result.skipped),
        expected_failures=len(result.expectedFailures),
        unexpected_successes=len(result.unexpectedSuccesses),
        duration_s=round(duration_s, 3),
        success=result.wasSuccessful(),
        output_text=stream.getvalue(),
        cases=cases,
    )


def copy_asset(source: Path | None, assets_dir: Path) -> str | None:
    if source is None or not source.exists() or not source.is_file():
        return None
    assets_dir.mkdir(parents=True, exist_ok=True)
    destination = assets_dir / source.name
    if destination.exists() and destination.resolve() != source.resolve():
        destination = assets_dir / f"{slugify(source.stem)}-{slugify(source.suffix)}{source.suffix}"
    shutil.copy2(source, destination)
    return relative_posix(destination, assets_dir.parent)


def evaluate_requirement(text: str, subsection: str, observed: dict[str, Any]) -> dict[str, str]:
    tts_elapsed_ms = observed.get("tts_elapsed_ms")
    tts_backend = observed.get("tts_backend", "unknown")
    asr_rknn_elapsed_ms = observed.get("asr_rknn_elapsed_ms")
    asr_mode = observed.get("asr_mode", "unknown")
    tts_max_rss_mib = observed.get("tts_max_rss_mib")
    tts_model_size_mib = observed.get("tts_model_size_mib")
    tts_model_is_int8 = bool(observed.get("tts_model_is_int8"))
    asr_cpu_model_size_mib = observed.get("asr_cpu_model_size_mib")
    models_total_size_mib = observed.get("models_total_size_mib")
    npu_peak_percent = observed.get("npu_peak_percent") or 0
    offline_ready = bool(observed.get("offline_ready"))
    tts_model_name = observed.get("tts_model_name", "")

    if "语音合成端到端延迟" in text:
        if tts_elapsed_ms is None:
            return {"status": "unknown", "observed": "缺少 TTS 冒烟或 profile 时延数据", "rationale": "当前报告没有可用的 TTS 延迟样本。"}
        if tts_backend != "rknn":
            return {
                "status": "fail",
                "observed": f"当前 TTS 后端为 {tts_backend}，单句 {tts_elapsed_ms:.0f} ms",
                "rationale": "指标明确要求 NPU 加速下 ≤ 150 ms，而当前交付主线仍是 CPU/ONNX TTS。",
            }
        if tts_elapsed_ms <= 150.0:
            return {"status": "pass", "observed": f"{tts_elapsed_ms:.0f} ms", "rationale": "已满足 150 ms 目标。"}
        return {"status": "fail", "observed": f"{tts_elapsed_ms:.0f} ms", "rationale": "虽为 NPU 路径，但仍超过 150 ms。"}

    if "普通话合成准确率" in text:
        return {"status": "unknown", "observed": "缺少带标注的 TTS 准确率测试集", "rationale": "当前报告只汇总冒烟和计划信息，没有自动化音素或文本对齐评分。"}

    if "通信指令、数字、字母发音准确无歧义" in text:
        if observed.get("plan_domain_case_count", 0) > 0:
            return {"status": "partial", "observed": f"已配置 {observed['plan_domain_case_count']} 条领域用例计划", "rationale": "测试计划覆盖了通信指令、数字和坐标样例，但当前报告未包含自动化主观/客观评分结果。"}
        return {"status": "unknown", "observed": "缺少领域词验证计划", "rationale": "当前报告没有可用的领域发音验证数据。"}

    if "主流方言合成" in text:
        if "aishell3" in tts_model_name.lower() or "zh" in tts_model_name.lower():
            return {"status": "fail", "observed": f"当前 TTS 模型为 {tts_model_name}", "rationale": "当前运行包只看到中文普通话基线模型，没有方言 TTS 交付证据。"}
        return {"status": "unknown", "observed": "未识别到 TTS 方言能力证据", "rationale": "当前报告没有可验证的方言 TTS 样本。"}

    if subsection == "1.3 模型与稳定性" and "模型体积 ≤ 100 MB" in text:
        if tts_model_size_mib is None:
            return {"status": "unknown", "observed": "缺少 TTS 模型文件", "rationale": "无法计算当前 TTS 模型体积。"}
        if tts_model_size_mib <= 100.0 and tts_model_is_int8:
            return {"status": "pass", "observed": f"{tts_model_size_mib:.1f} MiB", "rationale": "当前 TTS 模型体积满足指标，且文件名表明为 INT8 版本。"}
        if tts_model_size_mib <= 100.0:
            return {"status": "partial", "observed": f"{tts_model_size_mib:.1f} MiB", "rationale": "体积可能满足，但当前文件名未体现 INT8 量化证据。"}
        return {"status": "fail", "observed": f"{tts_model_size_mib:.1f} MiB", "rationale": "当前 TTS 模型体积超过 100 MiB 目标。"}

    if subsection == "1.3 模型与稳定性" and "运行内存占用 ≤ 300 MB" in text:
        if tts_max_rss_mib is None:
            return {"status": "unknown", "observed": "缺少 TTS profile RSS 采样", "rationale": "当前报告还没有可直接用于 TTS RSS 判断的 samples.csv。"}
        if tts_max_rss_mib <= 300.0:
            return {"status": "pass", "observed": f"峰值 RSS {tts_max_rss_mib:.1f} MiB", "rationale": "TTS profile 采样显示峰值内存在目标以内。"}
        return {"status": "fail", "observed": f"峰值 RSS {tts_max_rss_mib:.1f} MiB", "rationale": "TTS profile 采样显示峰值内存超出目标。"}

    if subsection == "1.3 模型与稳定性" and "7×24 小时" in text:
        if observed.get("plan_stability_case_count", 0) > 0:
            return {"status": "partial", "observed": f"已配置 {observed['plan_stability_case_count']} 条稳定性计划用例", "rationale": "仓库已有稳定性测试计划，但当前报告并未包含 7×24 连续运行证据。"}
        return {"status": "unknown", "observed": "缺少长稳执行记录", "rationale": "当前报告不包含长稳采样结果。"}

    if subsection == "1.4 运行模式" and "纯离线端侧运行" in text:
        if offline_ready:
            return {"status": "pass", "observed": "运行包自带本地二进制和模型", "rationale": "当前交付方式是本地运行包，不依赖在线接口。"}
        return {"status": "unknown", "observed": "缺少完整运行包证据", "rationale": "当前报告无法确认离线运行条件是否完整。"}

    if subsection == "2.1 端侧识别延迟" and "≤ 200 ms" in text:
        if asr_mode != "streaming":
            observed_text = "当前 ASR 入口为 offline"
            if asr_rknn_elapsed_ms is not None:
                observed_text += f"，RKNN 样本 {asr_rknn_elapsed_ms:.0f} ms"
            return {"status": "fail", "observed": observed_text, "rationale": "指标要求流式识别，而当前 run_asr.sh 和实测证据均为离线识别链路。"}
        if asr_rknn_elapsed_ms is None:
            return {"status": "unknown", "observed": "缺少流式 ASR 时延样本", "rationale": "未找到可用于 200 ms 判定的流式 ASR 证据。"}
        if asr_rknn_elapsed_ms <= 200.0:
            return {"status": "pass", "observed": f"{asr_rknn_elapsed_ms:.0f} ms", "rationale": "流式 ASR 延迟满足目标。"}
        return {"status": "fail", "observed": f"{asr_rknn_elapsed_ms:.0f} ms", "rationale": "流式 ASR 延迟超过目标。"}

    if subsection == "2.2 识别准确率":
        return {"status": "unknown", "observed": "缺少带标注的 ASR 评测集", "rationale": "当前报告仅汇总冒烟转写样例，不包含准确率统计。"}

    if subsection == "2.3 模型与稳定性" and "模型体积 ≤ 100 MB" in text:
        if asr_cpu_model_size_mib is None:
            return {"status": "unknown", "observed": "缺少 ASR 模型文件", "rationale": "无法计算当前 ASR 模型体积。"}
        if asr_cpu_model_size_mib <= 100.0:
            return {"status": "pass", "observed": f"CPU INT8 模型 {asr_cpu_model_size_mib:.1f} MiB", "rationale": "当前 CPU ASR 模型文件名和体积均符合 INT8 方向。"}
        return {"status": "fail", "observed": f"CPU INT8 模型 {asr_cpu_model_size_mib:.1f} MiB", "rationale": "当前 ASR 模型体积超过 100 MiB。"}

    if subsection == "2.3 模型与稳定性" and "支持流式识别" in text:
        if asr_mode == "streaming":
            return {"status": "pass", "observed": "检测到 streaming 入口", "rationale": "当前运行脚本使用流式识别链路。"}
        return {"status": "fail", "observed": f"当前 ASR 模式为 {asr_mode}", "rationale": "当前运行包仅暴露离线识别入口。"}

    if subsection == "2.3 模型与稳定性" and "7×24 小时" in text:
        return {"status": "unknown", "observed": "缺少 ASR 长稳记录", "rationale": "当前报告没有长时间连续 ASR 运行数据。"}

    if subsection == "2.4 运行模式" and "纯离线端侧运行" in text:
        if offline_ready:
            return {"status": "pass", "observed": "ASR 模型和二进制均为本地部署", "rationale": "当前运行包不依赖云端 ASR 接口。"}
        return {"status": "unknown", "observed": "缺少完整 ASR 运行包证据", "rationale": "无法确认离线运行条件。"}

    if "全链路闭环延迟" in text:
        if tts_elapsed_ms is not None and asr_rknn_elapsed_ms is not None:
            combined_ms = tts_elapsed_ms + asr_rknn_elapsed_ms
            status = "pass" if combined_ms <= 350.0 else "fail"
            return {
                "status": status,
                "observed": f"TTS {tts_elapsed_ms:.0f} ms + ASR RKNN {asr_rknn_elapsed_ms:.0f} ms = {combined_ms:.0f} ms",
                "rationale": "这里按现有冒烟样本做保守相加，且尚未包含真实闭环编解码开销。",
            }
        return {"status": "unknown", "observed": "缺少 TTS 或 ASR 延迟样本", "rationale": "当前报告无法对闭环延迟做完整判断。"}

    if "模型总存储占用 ≤ 200 MB" in text:
        if models_total_size_mib is None:
            return {"status": "unknown", "observed": "缺少 models 目录", "rationale": "无法计算整体模型体积。"}
        if models_total_size_mib <= 200.0:
            return {"status": "pass", "observed": f"{models_total_size_mib:.1f} MiB", "rationale": "当前 models 目录体积位于目标以内。"}
        return {"status": "fail", "observed": f"{models_total_size_mib:.1f} MiB", "rationale": "当前 models 目录体积超过 200 MiB 上限。"}

    if "典型工况内存占用 ≤ 500 MB" in text:
        if tts_max_rss_mib is None:
            return {"status": "unknown", "observed": "缺少系统级 RSS 采样", "rationale": "当前只有可选的 TTS profile 数据源，还没有完整系统级内存证据。"}
        if tts_max_rss_mib <= 500.0:
            return {"status": "partial", "observed": f"当前可见 TTS 峰值 RSS {tts_max_rss_mib:.1f} MiB", "rationale": "现有数据只覆盖 TTS 进程，不等于系统整体典型工况。"}
        return {"status": "fail", "observed": f"当前可见 TTS 峰值 RSS {tts_max_rss_mib:.1f} MiB", "rationale": "已知单一语音进程样本就超过 500 MiB，系统整体更不可能满足。"}

    if "支持 RK3588 NPU 硬件加速" in text:
        if npu_peak_percent > 0 and tts_backend != "rknn":
            return {"status": "partial", "observed": f"ASR RKNN 峰值 NPU load {npu_peak_percent}%", "rationale": "当前只有 ASR RKNN 命中 NPU，TTS 仍是 CPU 路径。"}
        if npu_peak_percent > 0:
            return {"status": "pass", "observed": f"峰值 NPU load {npu_peak_percent}%", "rationale": "当前采样显示核心能力已命中 RK3588 NPU。"}
        return {"status": "unknown", "observed": "未检测到 NPU 采样命中", "rationale": "当前报告缺少可证明的 NPU 负载样本。"}

    if "适配短波电台音频特性" in text:
        return {"status": "unknown", "observed": "缺少短波电台专项测试证据", "rationale": "当前报告未包含低信噪比或短波链路样本。"}

    if "低码率语音压缩模块" in text:
        return {"status": "unknown", "observed": "缺少低码率模块联调证据", "rationale": "当前仓库还没有压缩链路测试结果。"}

    if "主流方言的混合语音交互与播报" in text:
        return {"status": "fail", "observed": "当前 TTS 仍是普通话基线模型", "rationale": "系统级混合方言交互至少需要 TTS 和 ASR 同时具备方言验证证据。"}

    if "全程离线独立工作" in text:
        if offline_ready:
            return {"status": "pass", "observed": "现有运行包为本地模型 + 本地二进制", "rationale": "当前交付方式符合离线独立运行形态。"}
        return {"status": "unknown", "observed": "缺少完整离线证据", "rationale": "当前报告无法确认全部运行条件。"}

    return {"status": "unknown", "observed": "尚未定义自动判定规则", "rationale": "该指标需要后续补充专门测试或解析逻辑。"}


def build_requirement_assessments(requirements: list[dict[str, str]], observed: dict[str, Any]) -> list[dict[str, str]]:
    assessments: list[dict[str, str]] = []
    for item in requirements:
        evaluation = evaluate_requirement(item["text"], item["subsection"], observed)
        assessments.append(
            {
                "section": item["section"],
                "subsection": item["subsection"],
                "requirement": item["text"],
                "status": evaluation["status"],
                "observed": evaluation["observed"],
                "rationale": evaluation["rationale"],
            }
        )
    return assessments


def summarize_requirement_status(assessments: Sequence[dict[str, str]]) -> dict[str, int]:
    summary = {"pass": 0, "fail": 0, "partial": 0, "unknown": 0}
    for item in assessments:
        status = item["status"]
        if status not in summary:
            summary[status] = 0
        summary[status] += 1
    return summary


def heat_color(ratio: float) -> str:
    bounded = max(0.0, min(ratio, 1.0))
    red = int(243 + (183 - 243) * bounded)
    green = int(236 + (79 - 236) * bounded)
    blue = int(214 + (37 - 214) * bounded)
    return f"#{red:02x}{green:02x}{blue:02x}"


def render_heatmap_svg(
    *,
    title: str,
    subtitle: str,
    samples: Sequence[dict[str, Any]],
    rows: Sequence[tuple[str, str]],
    max_value: float,
) -> str:
    if not samples:
        return ""

    cell_width = 14
    cell_height = 22
    left_gutter = 120
    top_gutter = 58
    bottom_gutter = 30
    chart_width = max(len(samples) * cell_width, 200)
    width = left_gutter + chart_width + 20
    height = top_gutter + len(rows) * cell_height + bottom_gutter

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#fffaf0" />',
        '<stop offset="100%" stop-color="#f6ece5" />',
        '</linearGradient></defs>',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="18" fill="url(#bg)" />',
        f'<text x="20" y="28" font-size="18" font-family="Segoe UI Variable, Segoe UI, sans-serif" fill="#3e2a21">{html.escape(title)}</text>',
        f'<text x="20" y="46" font-size="11" font-family="Segoe UI Variable, Segoe UI, sans-serif" fill="#745b4f">{html.escape(subtitle)}</text>',
    ]

    for row_index, (label, field_name) in enumerate(rows):
        y = top_gutter + row_index * cell_height
        parts.append(
            f'<text x="20" y="{y + 15}" font-size="12" font-family="Segoe UI Variable, Segoe UI, sans-serif" fill="#6e4c3f">{html.escape(label)}</text>'
        )
        for sample_index, sample in enumerate(samples):
            value = coerce_float(sample.get(field_name)) or 0.0
            ratio = 0.0 if max_value <= 0 else min(value / max_value, 1.0)
            x = left_gutter + sample_index * cell_width
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 1}" height="{cell_height - 2}" rx="4" fill="{heat_color(ratio)}" />'
            )

    parts.append(
        f'<text x="{left_gutter}" y="{height - 10}" font-size="11" font-family="Segoe UI Variable, Segoe UI, sans-serif" fill="#7d675a">样本数 {len(samples)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def write_svg_asset(path: Path, markup: str) -> str | None:
    if not markup:
        return None
    write_text(path, markup)
    return relative_posix(path, path.parent.parent)


def collect_evidence(evidence_dir: Path, assets_dir: Path) -> dict[str, Any]:
    smoke_log_path = pick_first_existing([
        evidence_dir / "smoke_test_summary.log",
        evidence_dir / "smoke_test.log",
    ])
    board_capabilities_path = evidence_dir / "board_profile_capabilities.txt"
    rknn_profile_path = evidence_dir / "rknn_profile.log"
    tts_profile_csv_path = pick_first_existing([
        evidence_dir / "profile-samples.csv",
        evidence_dir / "tts-profile-samples.csv",
    ]) or first_glob(evidence_dir, "*profile*samples*.csv")
    tts_profile_log_path = pick_first_existing([
        evidence_dir / "profile.log",
        evidence_dir / "tts-profile.log",
    ])
    audio_path = pick_first_existing([
        evidence_dir / "profile.wav",
        evidence_dir / "smoke_test_tts.wav",
        evidence_dir / "smoke_test.wav",
    ]) or first_glob(evidence_dir, "*.wav")

    smoke_summary = parse_smoke_log(smoke_log_path)
    board_capabilities = parse_board_capabilities(board_capabilities_path)
    rknn_profile = parse_rknn_profile_log(rknn_profile_path)
    tts_profile = parse_tts_profile_csv(tts_profile_csv_path)
    wav_metadata = read_wav_metadata(audio_path)

    copied_assets = {
        "smoke_log": copy_asset(smoke_log_path, assets_dir),
        "board_capabilities": copy_asset(board_capabilities_path, assets_dir),
        "rknn_profile": copy_asset(rknn_profile_path, assets_dir),
        "tts_profile_csv": copy_asset(tts_profile_csv_path, assets_dir),
        "tts_profile_log": copy_asset(tts_profile_log_path, assets_dir),
        "audio": copy_asset(audio_path, assets_dir),
    }

    asr_heatmap_markup = render_heatmap_svg(
        title="ASR RKNN NPU Flame-Style Heatmap",
        subtitle="由 rknn_profile.log 采样数据生成，不是 perf 调用栈火焰图。",
        samples=rknn_profile.get("samples", []),
        rows=(
            ("Core0", "core0_percent"),
            ("Core1", "core1_percent"),
            ("Core2", "core2_percent"),
        ),
        max_value=100.0,
    )
    tts_heatmap_markup = render_heatmap_svg(
        title="TTS Profile Flame-Style Heatmap",
        subtitle="由 profile-samples.csv 采样的 RSS / CPU / NPU 指标生成。",
        samples=[
            {
                "rss": sample["rss_kb"] / 1024.0,
                "cpu_user": sample["utime_ticks"],
                "npu_peak": max(sample["npu_core0_percent"], sample["npu_core1_percent"], sample["npu_core2_percent"]),
            }
            for sample in tts_profile.get("samples", [])
        ],
        rows=(
            ("RSS MiB", "rss"),
            ("User CPU Ticks", "cpu_user"),
            ("NPU Peak %", "npu_peak"),
        ),
        max_value=max(
            [
                max((sample["rss_kb"] / 1024.0 for sample in tts_profile.get("samples", [])), default=0.0),
                max((sample["utime_ticks"] for sample in tts_profile.get("samples", [])), default=0.0),
                max(
                    (
                        max(sample["npu_core0_percent"], sample["npu_core1_percent"], sample["npu_core2_percent"])
                        for sample in tts_profile.get("samples", [])
                    ),
                    default=0.0,
                ),
            ]
        ),
    )

    copied_assets["asr_rknn_heatmap"] = write_svg_asset(assets_dir / "asr-rknn-heatmap.svg", asr_heatmap_markup)
    copied_assets["tts_profile_heatmap"] = write_svg_asset(assets_dir / "tts-profile-heatmap.svg", tts_heatmap_markup)

    return {
        "evidence_dir": str(evidence_dir),
        "smoke": smoke_summary,
        "board_capabilities": board_capabilities,
        "rknn_profile": rknn_profile,
        "tts_profile": tts_profile,
        "audio": wav_metadata,
        "assets": copied_assets,
    }


def build_observed_metrics(runtime_info: dict[str, Any], evidence: dict[str, Any], plan_summary: dict[str, Any]) -> dict[str, Any]:
    smoke = evidence.get("smoke", {})
    tts_smoke = smoke.get("tts", {})
    asr_rknn = smoke.get("asr_rknn", {})
    tts_profile = evidence.get("tts_profile", {})
    rknn_profile = evidence.get("rknn_profile", {})

    category_counts = plan_summary.get("category_counts", {})
    return {
        "tts_backend": runtime_info.get("tts_backend"),
        "asr_mode": runtime_info.get("asr_mode"),
        "tts_elapsed_ms": (tts_smoke.get("elapsed_seconds") or 0) * 1000.0 if tts_smoke.get("elapsed_seconds") is not None else None,
        "tts_audio_duration_s": tts_smoke.get("audio_duration_seconds"),
        "tts_rtf": tts_smoke.get("rtf"),
        "asr_rknn_elapsed_ms": (asr_rknn.get("elapsed_seconds") or 0) * 1000.0 if asr_rknn.get("elapsed_seconds") is not None else None,
        "asr_rknn_rtf": asr_rknn.get("rtf"),
        "tts_max_rss_mib": tts_profile.get("max_rss_mib"),
        "npu_peak_percent": max(rknn_profile.get("peak_percent", 0), tts_profile.get("peak_npu_percent", 0)),
        "tts_model_size_mib": runtime_info.get("tts_model_size_mib"),
        "tts_model_is_int8": runtime_info.get("tts_model_is_int8"),
        "tts_model_name": runtime_info.get("tts_model_name"),
        "asr_cpu_model_size_mib": runtime_info.get("asr_cpu_model_size_mib"),
        "models_total_size_mib": runtime_info.get("models_total_size_mib"),
        "offline_ready": runtime_info.get("offline_ready"),
        "plan_domain_case_count": category_counts.get("domain", 0),
        "plan_stability_case_count": category_counts.get("stability", 0),
    }


def determine_overall_verdict(requirement_summary: dict[str, int], unittest_summary: UnittestSummary | None) -> tuple[str, str]:
    if unittest_summary is not None and not unittest_summary.success:
        return "fail", "单元测试未通过，当前报告不能作为稳定基线。"
    if requirement_summary.get("fail", 0) > 0:
        return "fail", "当前证据表明项目仍不符合指标要求。"
    if requirement_summary.get("partial", 0) > 0:
        return "partial", "当前实现只满足部分指标，还需要补充能力和验证。"
    if requirement_summary.get("unknown", 0) > 0:
        return "unknown", "当前报告已生成，但仍有较多指标缺少可自动判定证据。"
    return "pass", "当前报告覆盖的指标均已满足。"


def render_card(title: str, value: str, subtitle: str, accent: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-accent" style="background:{accent}"></div>'
        f'<div class="metric-title">{html.escape(title)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-subtitle">{html.escape(subtitle)}</div>'
        '</div>'
    )


def render_status_pill(status: str) -> str:
    classes = {
        "pass": "pill pass",
        "fail": "pill fail",
        "partial": "pill partial",
        "unknown": "pill unknown",
    }
    class_name = classes.get(status, "pill unknown")
    return f'<span class="{class_name}">{html.escape(format_status(status))}</span>'


def render_html_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    unit_tests = payload.get("unit_tests")
    requirement_summary = payload["requirements"]["summary"]
    evidence_assets = payload["evidence"]["assets"]
    audio_asset = evidence_assets.get("audio")
    asr_heatmap_asset = evidence_assets.get("asr_rknn_heatmap")
    tts_heatmap_asset = evidence_assets.get("tts_profile_heatmap")
    plan_summary = payload.get("plan", {})

    cards = "".join(
        [
            render_card("综合判定", format_status(summary["verdict"]), summary["message"], "linear-gradient(135deg,#d66d4b,#9f2f1f)"),
            render_card("单元测试", f"{unit_tests['passed']}/{unit_tests['total']}" if unit_tests else "未执行", "通过 / 总数", "linear-gradient(135deg,#5b8c5a,#2f5d50)"),
            render_card("TTS 单句时延", format_number(payload["observed"]["tts_elapsed_ms"], digits=0, suffix=" ms"), f"后端 {payload['observed']['tts_backend']}", "linear-gradient(135deg,#d28d49,#9c5a12)"),
            render_card("ASR RKNN 时延", format_number(payload["observed"]["asr_rknn_elapsed_ms"], digits=0, suffix=" ms"), f"模式 {payload['observed']['asr_mode']}", "linear-gradient(135deg,#618fbf,#2f5e8a)"),
            render_card("NPU 峰值负载", format_number(payload["observed"]["npu_peak_percent"], digits=0, suffix=" %"), "来自 rknn_profile.log / TTS profile", "linear-gradient(135deg,#b65f6f,#7e3240)"),
            render_card("模型总量", format_number(payload["observed"]["models_total_size_mib"], digits=1, suffix=" MiB"), "当前运行包 models 目录", "linear-gradient(135deg,#7d7c98,#514f73)"),
        ]
    )

    requirement_rows = "".join(
        f"<tr><td>{html.escape(item['section'])}</td><td>{html.escape(item['subsection'])}</td><td>{html.escape(item['requirement'])}</td><td>{render_status_pill(item['status'])}</td><td>{html.escape(item['observed'])}</td><td>{html.escape(item['rationale'])}</td></tr>"
        for item in payload["requirements"]["items"]
    )

    unittest_rows = ""
    if unit_tests:
        unittest_rows = "".join(
            f"<tr><td>{html.escape(case['test_id'])}</td><td>{render_status_pill('pass' if case['status']=='passed' else 'fail' if case['status'] in {'failed','error','unexpected-success'} else 'partial' if case['status']=='skipped' else 'unknown')}</td><td>{html.escape(format_status(case['status']))}</td><td>{case['duration_s']:.3f}s</td><td><pre>{html.escape(case['details'])}</pre></td></tr>"
            for case in unit_tests["cases"]
        )

    plan_rows = ""
    for category, count in sorted(plan_summary.get("category_counts", {}).items()):
        plan_rows += f"<tr><td>{html.escape(category)}</td><td>{count}</td></tr>"

    asset_links = "".join(
        f'<li><a href="{html.escape(path)}">{html.escape(name)}</a></li>'
        for name, path in evidence_assets.items()
        if path
    )

    audio_block = '<div class="empty-state">未发现可嵌入的音频证据。</div>'
    if audio_asset:
        audio_block = (
            '<div class="media-card">'
            '<div class="panel-title">音频预览</div>'
            f'<audio controls preload="metadata" src="{html.escape(audio_asset)}"></audio>'
            f'<div class="media-meta">时长 {html.escape(format_number(payload["evidence"]["audio"].get("duration_s"), digits=3, suffix=" s"))}，体积 {html.escape(format_number(payload["evidence"]["audio"].get("size_mib"), digits=3, suffix=" MiB"))}</div>'
            '</div>'
        )

    heatmap_blocks: list[str] = []
    if asr_heatmap_asset:
        heatmap_blocks.append(
            '<div class="media-card">'
            '<div class="panel-title">ASR RKNN Flame-Style Heatmap</div>'
            f'<img alt="ASR RKNN heatmap" src="{html.escape(asr_heatmap_asset)}" />'
            '</div>'
        )
    if tts_heatmap_asset:
        heatmap_blocks.append(
            '<div class="media-card">'
            '<div class="panel-title">TTS Profile Flame-Style Heatmap</div>'
            f'<img alt="TTS profile heatmap" src="{html.escape(tts_heatmap_asset)}" />'
            '</div>'
        )
    if not heatmap_blocks:
        heatmap_blocks.append('<div class="empty-state">未发现可生成 flame-style heatmap 的 profile 采样文件。</div>')

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RKVoice 综合测试报告</title>
  <style>
    :root {{
      --bg-top: #f4efe8;
      --bg-bottom: #f7f6f1;
      --ink: #2e211b;
      --muted: #6f6158;
      --panel: rgba(255,255,255,0.86);
      --line: rgba(112, 84, 69, 0.16);
      --pass: #3d7a4f;
      --fail: #a93a2e;
      --partial: #b87722;
      --unknown: #7c7f84;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI Variable", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(214,109,75,0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(97,143,191,0.14), transparent 24%),
        linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
    }}
    .shell {{ max-width: 1380px; margin: 0 auto; padding: 28px; }}
    .hero {{
      padding: 28px 32px;
      border-radius: 28px;
      background: linear-gradient(135deg, rgba(255,248,239,0.92), rgba(255,255,255,0.84));
      border: 1px solid var(--line);
      box-shadow: 0 24px 60px rgba(79, 57, 46, 0.09);
    }}
    .eyebrow {{ letter-spacing: 0.18em; text-transform: uppercase; font-size: 12px; color: #8a6b5d; }}
    h1 {{ margin: 10px 0 6px; font-size: clamp(32px, 5vw, 52px); line-height: 1.02; }}
    .hero p {{ margin: 0; max-width: 820px; color: var(--muted); font-size: 16px; line-height: 1.7; }}
    .hero-meta {{ margin-top: 18px; display: flex; flex-wrap: wrap; gap: 12px; color: var(--muted); font-size: 13px; }}
    .hero-chip {{ padding: 8px 12px; border-radius: 999px; background: rgba(255,255,255,0.7); border: 1px solid var(--line); }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 24px 0 12px; }}
    .metric-card {{ position: relative; overflow: hidden; padding: 18px; border-radius: 22px; background: var(--panel); border: 1px solid var(--line); min-height: 152px; }}
    .metric-accent {{ position: absolute; inset: 0 auto auto 0; width: 100%; height: 6px; }}
    .metric-title {{ color: #8f6f5f; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 8px; }}
    .metric-value {{ margin-top: 12px; font-size: 34px; font-weight: 700; }}
    .metric-subtitle {{ margin-top: 8px; color: var(--muted); font-size: 13px; line-height: 1.6; }}
    .grid {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 18px; margin-top: 22px; }}
    .panel {{ padding: 22px; border-radius: 24px; background: var(--panel); border: 1px solid var(--line); box-shadow: 0 16px 40px rgba(79,57,46,0.06); }}
    .panel-title {{ font-size: 18px; font-weight: 700; margin-bottom: 12px; }}
    .panel-subtitle {{ color: var(--muted); font-size: 13px; line-height: 1.6; margin-bottom: 18px; }}
    .stack {{ display: grid; gap: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-top: 1px solid var(--line); text-align: left; vertical-align: top; padding: 12px 10px; }}
    th {{ color: #765f53; font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; }}
    tr:first-child th, tr:first-child td {{ border-top: none; }}
    .pill {{ display: inline-flex; padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
    .pill.pass {{ background: rgba(61,122,79,0.12); color: var(--pass); }}
    .pill.fail {{ background: rgba(169,58,46,0.12); color: var(--fail); }}
    .pill.partial {{ background: rgba(184,119,34,0.12); color: var(--partial); }}
    .pill.unknown {{ background: rgba(124,127,132,0.12); color: var(--unknown); }}
    .media-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    .media-card {{ padding: 18px; border-radius: 20px; background: rgba(255,255,255,0.72); border: 1px solid var(--line); }}
    .media-card img {{ width: 100%; display: block; border-radius: 16px; border: 1px solid rgba(112, 84, 69, 0.14); background: #fff8f2; }}
    audio {{ width: 100%; margin-top: 8px; }}
    .media-meta {{ margin-top: 10px; color: var(--muted); font-size: 13px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .summary-box {{ padding: 14px 16px; border-radius: 18px; background: rgba(255,255,255,0.72); border: 1px solid var(--line); }}
    .summary-box strong {{ display: block; font-size: 24px; margin-top: 6px; }}
    .empty-state {{ padding: 22px; border-radius: 20px; background: rgba(255,255,255,0.62); color: var(--muted); border: 1px dashed rgba(112,84,69,0.24); }}
    ul.asset-list {{ margin: 0; padding-left: 18px; color: var(--muted); }}
    a {{ color: #8b3d2c; }}
    pre {{ margin: 0; white-space: pre-wrap; font-family: Consolas, "SFMono-Regular", monospace; font-size: 11px; color: #5a4840; }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 640px) {{
      .shell {{ padding: 18px; }}
      .hero {{ padding: 22px; border-radius: 22px; }}
      .summary-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">RKVoice Integrated Report</div>
      <h1>RK3588 离线 ASR + TTS 综合测试报告</h1>
      <p>这份报告将当前仓库的 unittest 结果、板端 smoke / profile 证据、测试计划模板和项目指标文档汇总到一个可交付的 HTML Dashboard 中。报告中的 flame-style heatmap 是基于采样日志绘制的工程可视化，不等同于 perf 调用栈火焰图。</p>
      <div class="hero-meta">
        <div class="hero-chip">生成时间 {html.escape(payload['generated_at'])}</div>
        <div class="hero-chip">工作区 {html.escape(payload['workspace_root'])}</div>
        <div class="hero-chip">运行包 {html.escape(payload['runtime']['runtime_dir'])}</div>
        <div class="hero-chip">测试计划 {html.escape(plan_summary.get('path', '未配置'))}</div>
      </div>
    </section>

    <section class="metrics">{cards}</section>

    <section class="grid">
      <div class="stack">
        <article class="panel">
          <div class="panel-title">指标总览</div>
          <div class="panel-subtitle">报告按 docs/requirements/项目指标.md 逐条生成状态。未通过说明当前证据已证明不达标，证据不足说明需要追加专门测试。</div>
          <div class="summary-grid">
            <div class="summary-box">通过<strong>{requirement_summary['pass']}</strong></div>
            <div class="summary-box">未通过<strong>{requirement_summary['fail']}</strong></div>
            <div class="summary-box">部分满足<strong>{requirement_summary['partial']}</strong></div>
            <div class="summary-box">证据不足<strong>{requirement_summary['unknown']}</strong></div>
          </div>
        </article>

        <article class="panel">
          <div class="panel-title">指标矩阵</div>
          <table>
            <tr><th>章节</th><th>子项</th><th>要求</th><th>状态</th><th>观测</th><th>判定依据</th></tr>
            {requirement_rows}
          </table>
        </article>

        <article class="panel">
          <div class="panel-title">单元测试明细</div>
          <div class="panel-subtitle">统一复用 tests 目录 discovery 入口，结果同时写入 JSON 和 HTML。</div>
          <table>
            <tr><th>测试</th><th>状态</th><th>类型</th><th>耗时</th><th>详情</th></tr>
            {unittest_rows or '<tr><td colspan="5">本次未执行 unittest。</td></tr>'}
          </table>
        </article>
      </div>

      <div class="stack">
        <article class="panel">
          <div class="panel-title">证据媒体</div>
          <div class="media-grid">
            {audio_block}
            {''.join(heatmap_blocks)}
          </div>
        </article>

        <article class="panel">
          <div class="panel-title">计划与基线</div>
          <div class="panel-subtitle">这里展示测试计划模板中的覆盖面，帮助区分“有计划但未执行”和“根本没有验证入口”。</div>
          <table>
            <tr><th>类别</th><th>用例数</th></tr>
            {plan_rows or '<tr><td colspan="2">未找到测试计划文件。</td></tr>'}
          </table>
        </article>

        <article class="panel">
          <div class="panel-title">原始产物</div>
          <ul class="asset-list">{asset_links or '<li>未复制任何证据文件。</li>'}</ul>
        </article>
      </div>
    </section>
  </div>
</body>
</html>
"""


def build_report(
    *,
    workspace_root: Path = WORKSPACE_ROOT,
    output_root: Path | None = None,
    runtime_dir: Path | None = None,
    evidence_dir: Path | None = None,
    requirements_path: Path | None = None,
    plan_path: Path | None = None,
    run_unittests: bool = True,
    unittest_pattern: str = "test_*.py",
) -> ReportBuildResult:
    workspace_root = workspace_root.resolve()
    resolved_output_root = (output_root or DEFAULT_OUTPUT_ROOT).resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    resolved_runtime_dir = (runtime_dir or (workspace_root / DEFAULT_RUNTIME_DIR.relative_to(WORKSPACE_ROOT))).resolve()
    resolved_evidence_dir = (evidence_dir or (resolved_runtime_dir / "output")).resolve()
    resolved_requirements_path = (requirements_path or (workspace_root / DEFAULT_REQUIREMENTS_PATH.relative_to(WORKSPACE_ROOT))).resolve()
    resolved_plan_path = resolve_plan_path(workspace_root, plan_path.resolve() if plan_path is not None else None)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    generated_at = now.strftime("%Y-%m-%d %H:%M:%S")
    report_dir = resolved_output_root / f"rkvoice-report-{timestamp}"
    assets_dir = report_dir / "assets"
    report_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    unittest_summary: UnittestSummary | None = None
    if run_unittests:
        unittest_summary = run_unittest_suite(workspace_root, unittest_pattern)
        unittest_output_path = assets_dir / "unittest-output.txt"
        write_text(unittest_output_path, unittest_summary.output_text)
        unittest_output_asset = relative_posix(unittest_output_path, report_dir)
    else:
        unittest_output_asset = None

    runtime_info = inspect_runtime(resolved_runtime_dir)
    evidence = collect_evidence(resolved_evidence_dir, assets_dir)
    if unittest_output_asset is not None:
        evidence["assets"]["unittest_output"] = unittest_output_asset

    plan_summary = load_plan_summary(resolved_plan_path)
    observed = build_observed_metrics(runtime_info, evidence, plan_summary)
    requirements = parse_requirements(resolved_requirements_path)
    requirement_items = build_requirement_assessments(requirements, observed)
    requirement_summary = summarize_requirement_status(requirement_items)
    verdict, verdict_message = determine_overall_verdict(requirement_summary, unittest_summary)

    payload: dict[str, Any] = {
        "generated_at": generated_at,
        "workspace_root": str(workspace_root),
        "summary": {
            "verdict": verdict,
            "message": verdict_message,
        },
        "runtime": runtime_info,
        "observed": observed,
        "plan": plan_summary,
        "evidence": evidence,
        "requirements": {
            "path": str(resolved_requirements_path),
            "summary": requirement_summary,
            "items": requirement_items,
        },
        "unit_tests": None,
    }

    if unittest_summary is not None:
        payload["unit_tests"] = {
            "total": unittest_summary.total,
            "passed": unittest_summary.passed,
            "failed": unittest_summary.failed,
            "errors": unittest_summary.errors,
            "skipped": unittest_summary.skipped,
            "expected_failures": unittest_summary.expected_failures,
            "unexpected_successes": unittest_summary.unexpected_successes,
            "duration_s": unittest_summary.duration_s,
            "success": unittest_summary.success,
            "output_asset": unittest_output_asset,
            "cases": [
                {
                    "test_id": case.test_id,
                    "status": case.status,
                    "duration_s": round(case.duration_s, 3),
                    "details": case.details,
                }
                for case in unittest_summary.cases
            ],
        }

    json_path = report_dir / "report.json"
    html_path = report_dir / "index.html"
    write_json(json_path, payload)
    write_text(html_path, render_html_report(payload))

    return ReportBuildResult(
        report_dir=report_dir,
        html_path=html_path,
        json_path=json_path,
        unittest_success=unittest_summary.success if unittest_summary is not None else True,
        requirement_failures=requirement_summary.get("fail", 0),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an integrated RKVoice HTML/JSON test report")
    parser.add_argument("--output-root", default="", help="Report output root directory")
    parser.add_argument("--runtime-dir", default="", help="Runtime bundle directory used for evidence discovery")
    parser.add_argument("--evidence-dir", default="", help="Explicit evidence directory, defaults to <runtime-dir>/output")
    parser.add_argument("--requirements", default="", help="Requirements markdown path")
    parser.add_argument("--plan", default="", help="TTS test plan JSON path")
    parser.add_argument("--skip-unittests", action="store_true", help="Generate the report without rerunning unittest discovery")
    parser.add_argument("--unittest-pattern", default="test_*.py", help="Unittest discovery pattern")
    parser.add_argument("--fail-on-requirement-failures", action="store_true", help="Return exit code 1 when any requirement assessment is marked as fail")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_report(
            output_root=Path(args.output_root).resolve() if args.output_root else None,
            runtime_dir=Path(args.runtime_dir).resolve() if args.runtime_dir else None,
            evidence_dir=Path(args.evidence_dir).resolve() if args.evidence_dir else None,
            requirements_path=Path(args.requirements).resolve() if args.requirements else None,
            plan_path=Path(args.plan).resolve() if args.plan else None,
            run_unittests=not args.skip_unittests,
            unittest_pattern=args.unittest_pattern,
        )
    except ReportBuildError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Report directory: {result.report_dir}")
    print(f"HTML report:      {result.html_path}")
    print(f"JSON report:      {result.json_path}")

    if not result.unittest_success:
        return 1
    if args.fail_on_requirement_failures and result.requirement_failures > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())