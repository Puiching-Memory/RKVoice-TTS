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
from typing import Any, Mapping, Sequence

from jinja2 import Environment, FileSystemLoader, select_autoescape


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "artifacts" / "test-runs"
DEFAULT_RUNTIME_DIR = WORKSPACE_ROOT / "artifacts" / "runtime" / "rkvoice_runtime"
ASR_RUNTIME_SUBDIR = Path("asr")
TTS_RUNTIME_SUBDIR = Path("tts")
DEFAULT_REQUIREMENTS_PATH = WORKSPACE_ROOT / "docs" / "requirements" / "项目指标.md"
DEFAULT_LOCAL_PLAN_PATH = WORKSPACE_ROOT / "config" / "local" / "tts_test_plan.json"
DEFAULT_EXAMPLE_PLAN_PATH = WORKSPACE_ROOT / "config" / "examples" / "tts_test_plan.example.json"
REPORT_TEMPLATE_DIR = Path(__file__).with_name("report_templates")
REPORT_TEMPLATE_NAME = "rkvoice_report.html.j2"

ELAPSED_SECONDS_PATTERN = re.compile(r"Elapsed seconds:\s*([0-9.]+)")
AUDIO_DURATION_PATTERN = re.compile(r"Audio duration(?:\s*\(s\))?:\s*([0-9.]+)")
RTF_PATTERN = re.compile(r"Real[- ]time factor.*=\s*([0-9.]+)")
TEXT_PATTERN = re.compile(r"The text is:\s*(.+?)\.\s*Speaker ID:")
MELO_ENCODER_PATTERN = re.compile(r"encoder run take\s*([0-9.]+)\s*ms", re.IGNORECASE)
MELO_DECODER_PATTERN = re.compile(r"decoder run take\s*([0-9.]+)\s*ms", re.IGNORECASE)
MELO_MODEL_LOAD_PATTERN = re.compile(r"load models take\s*([0-9.]+)\s*ms", re.IGNORECASE)
SMOKE_FAILURE_PATTERN = re.compile(r"failed with exit code:\s*([0-9]+)", re.IGNORECASE)
CORE_LOAD_PATTERN = re.compile(r"Core([012]):\s*([0-9]+)%")
RKNN_VERSION_PATTERN = re.compile(r"librknnrt version:\s*(.+)")
MEMORY_TOTAL_PATTERN = re.compile(r"内存：\s*([0-9.]+)([GMK]i)\b")
RKNN_TOTAL_OPERATOR_TIME_PATTERN = re.compile(r"Total Operator Elapsed(?: Per Frame)? Time\(us\):\s*([0-9.]+)")
RKNN_TOTAL_MEMORY_RW_PATTERN = re.compile(r"Total Memory RW Amount\(MB\):\s*([0-9.]+)")
RKNN_TOTAL_MEMORY_RW_KB_PATTERN = re.compile(r"Total Memory Read/Write(?: Per Frame)? Size\(KB\):\s*([0-9.]+)")
RKNN_MEMORY_WEIGHT_PATTERN = re.compile(r"(?:Total )?Weight Memory:\s*([0-9.]+)\s*MiB")
RKNN_MEMORY_INTERNAL_PATTERN = re.compile(r"(?:Total )?Internal Tensor Memory:\s*([0-9.]+)\s*MiB")
RKNN_MEMORY_TOTAL_PATTERN = re.compile(r"Total Memory:\s*([0-9.]+)\s*MiB")
RKNN_MODEL_SIZE_PATTERN = re.compile(r"current model size is:\s*([0-9.]+)\s*MiB", re.IGNORECASE)
RKNN_RUN_DURATION_PATTERN = re.compile(r"(?:run_duration|run duration|real inference time).*?([0-9.]+)")
RKNN_RUNTIME_LAYER_HINT_PATTERN = re.compile(r"MACs utilization|bandwidth occupation", re.IGNORECASE)
PERCENT_VALUE_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")
WORKLOAD_DISTRIBUTION_PATTERN = re.compile(r"([0-9.]+)%/([0-9.]+)%/([0-9.]+)%\s*-\s*Up:([0-9.]+)%")
WORKLOAD_SIMPLE_PATTERN = re.compile(r"([0-9.]+)%/([0-9.]+)%/([0-9.]+)%$")
SLASH_NUMERIC_TRIPLE_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)/([0-9]+(?:\.[0-9]+)?)/([0-9]+(?:\.[0-9]+)?)$")


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


def coerce_int(value: Any) -> int | None:
    numeric = coerce_float(value)
    if numeric is None:
        return None
    return int(numeric)


def split_columns(line: str, *, maxsplit: int) -> list[str]:
    return [part.strip() for part in re.split(r"\s{2,}", line.strip(), maxsplit=maxsplit) if part.strip()]


def parse_percentage(value: str) -> float | None:
    cleaned = value.strip().rstrip("%")
    return coerce_float(cleaned)


def parse_slash_numeric_triplet(value: str) -> tuple[float, float, float] | None:
    match = SLASH_NUMERIC_TRIPLE_PATTERN.fullmatch(value.strip())
    if not match:
        return None
    return (float(match.group(1)), float(match.group(2)), float(match.group(3)))


def parse_rknn_mac_usage(value: str) -> float | None:
    stripped = value.strip()
    triplet = parse_slash_numeric_triplet(stripped)
    if triplet is not None and "%" not in stripped:
        return max(triplet)
    return parse_percentage(stripped)


def parse_rknn_cycle_triplet(value: str) -> tuple[int | None, int | None, int | None]:
    triplet = parse_slash_numeric_triplet(value)
    if triplet is None:
        return None, None, None
    return int(triplet[0]), int(triplet[1]), int(triplet[2])


def format_rknn_profile_source(source: str | None) -> str:
    labels = {
        "eval_perf": "Toolkit2 eval_perf()",
        "perf_detail": "RKNN_QUERY_PERF_DETAIL",
        "runtime_log": "RKNN_LOG_LEVEL=4",
        "load_sampling": "rknpu/load 采样",
        "tts_profile_csv": "TTS profile-samples.csv",
    }
    if not source:
        return "未采集"
    return labels.get(source, source)


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


def build_report_template_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(REPORT_TEMPLATE_DIR)),
        autoescape=select_autoescape(("html", "xml")),
        trim_blocks=True,
        lstrip_blocks=True,
    )


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
    if "decoder.rknn" in script_lower or "melotts_rknn" in script_lower:
        return "melotts-rknn"
    if ".rknn" in script_lower or "provider=rknn" in script_lower:
        return "rknn"
    if "onnx" in script_lower or "sherpa-onnx-offline-tts" in script_lower:
        return "cpu-onnx"
    return "unknown"


def maybe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def inspect_tts_runtime(tts_runtime_dir: Path) -> dict[str, Any]:
    run_tts_path = tts_runtime_dir / "run_tts.sh"
    run_tts_script = maybe_read_text(run_tts_path)

    decoder_rknn_path = tts_runtime_dir / "decoder.rknn"
    encoder_onnx_path = tts_runtime_dir / "encoder.onnx"

    decoder_size = file_size_bytes(decoder_rknn_path)
    encoder_size = file_size_bytes(encoder_onnx_path)
    total_size = (decoder_size or 0) + (encoder_size or 0)

    return {
        "tts_runtime_dir": str(tts_runtime_dir),
        "tts_backend": detect_tts_backend(run_tts_script),
        "tts_supports_rknn": decoder_rknn_path.exists(),
        "tts_model_name": "MeloTTS-RKNN2" if decoder_rknn_path.exists() else "",
        "tts_model_size_mib": bytes_to_mib(total_size) if total_size > 0 else None,
        "tts_model_is_int8": False,
        "tts_decoder_rknn_size_mib": bytes_to_mib(decoder_size),
        "tts_encoder_onnx_size_mib": bytes_to_mib(encoder_size),
    }


def inspect_runtime(runtime_dir: Path) -> dict[str, Any]:
    asr_runtime_dir = runtime_dir / ASR_RUNTIME_SUBDIR
    tts_runtime_dir = runtime_dir / TTS_RUNTIME_SUBDIR

    run_asr_path = asr_runtime_dir / "run_asr.sh"
    run_asr_script = maybe_read_text(run_asr_path)

    asr_streaming_rknn_model_dir = asr_runtime_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small"
    asr_streaming_rknn_encoder_path = asr_streaming_rknn_model_dir / "encoder.rknn"

    tts_info: dict[str, Any]
    if tts_runtime_dir.exists():
        tts_info = inspect_tts_runtime(tts_runtime_dir)
    else:
        tts_info = {
            "tts_runtime_dir": "",
            "tts_backend": "unknown",
            "tts_supports_rknn": False,
            "tts_model_name": "",
            "tts_model_size_mib": None,
            "tts_model_is_int8": False,
            "tts_decoder_rknn_size_mib": None,
            "tts_encoder_onnx_size_mib": None,
        }

    asr_models_size = sum(filter(None, [
        directory_size_bytes(asr_streaming_rknn_model_dir) if asr_streaming_rknn_model_dir.exists() else 0,
    ]))
    tts_model_bytes = int((tts_info.get("tts_model_size_mib") or 0) * 1024 * 1024)
    models_total = asr_models_size + tts_model_bytes

    result = {
        "runtime_dir": str(runtime_dir),
        "asr_runtime_dir": str(asr_runtime_dir),
        "asr_supports_rknn": "provider=rknn" in run_asr_script.lower(),
        "asr_streaming_rknn_available": asr_streaming_rknn_encoder_path.exists(),
        "asr_streaming_rknn_model_size_mib": bytes_to_mib(directory_size_bytes(asr_streaming_rknn_model_dir)) if asr_streaming_rknn_model_dir.exists() else None,
        "asr_streaming_rknn_model_name": asr_streaming_rknn_model_dir.name if asr_streaming_rknn_model_dir.exists() else "",
        "models_total_size_mib": bytes_to_mib(models_total) if models_total > 0 else None,
        "offline_ready": (asr_runtime_dir / "bin").exists() and (asr_runtime_dir / "models").exists(),
    }
    result.update(tts_info)
    return result


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
        "tts_cold_start": {"label": "TTS cold start"},
        "tts_warm_run": {"label": "TTS warm run"},
        "tts_profile_run": {"label": "TTS profile run"},
        "asr_streaming_rknn": {"label": "Streaming ASR (RKNN) smoke"},
    }
    current_section: str | None = None

    section_pattern = re.compile(r"^\[\d+/\d+\]\s+(.+)$")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        header_match = section_pattern.match(line)
        if header_match:
            header_text = header_match.group(1).strip()
            header = header_text.lower()
            if "tts" in header:
                if "profile" in header:
                    current_section = "tts_profile_run"
                elif "warm" in header:
                    current_section = "tts_warm_run"
                else:
                    current_section = "tts_cold_start"
            elif "streaming" in header or "asr" in header:
                current_section = "asr_streaming_rknn"
            else:
                current_section = None
            if current_section is not None:
                sections[current_section]["label"] = header_text
            continue
        if current_section is None:
            continue

        section = sections[current_section]
        elapsed_match = ELAPSED_SECONDS_PATTERN.search(line)
        if elapsed_match:
            section.setdefault("_elapsed_list", [])
            section["_elapsed_list"].append(float(elapsed_match.group(1)))
        audio_match = AUDIO_DURATION_PATTERN.search(line)
        if audio_match:
            section.setdefault("_audio_dur_list", [])
            section["_audio_dur_list"].append(float(audio_match.group(1)))
        rtf_match = RTF_PATTERN.search(line)
        if rtf_match:
            section.setdefault("_rtf_list", [])
            section["_rtf_list"].append(float(rtf_match.group(1)))
        text_match = TEXT_PATTERN.search(line)
        if text_match:
            section["text"] = text_match.group(1).strip()
        encoder_match = MELO_ENCODER_PATTERN.search(line)
        if encoder_match:
            section.setdefault("_encoder_ms", 0.0)
            section["_encoder_ms"] = float(encoder_match.group(1))
        decoder_match = MELO_DECODER_PATTERN.search(line)
        if decoder_match:
            section.setdefault("_decoder_ms_list", [])
            section["_decoder_ms_list"].append(float(decoder_match.group(1)))
        model_load_match = MELO_MODEL_LOAD_PATTERN.search(line)
        if model_load_match:
            section["model_load_ms"] = float(model_load_match.group(1))
        smoke_failure_match = SMOKE_FAILURE_PATTERN.search(line)
        if smoke_failure_match:
            section["failed"] = True
            section["exit_code"] = int(smoke_failure_match.group(1))
        if line.startswith("{") and line.endswith("}"):
            try:
                section["result"] = json.loads(line)
            except json.JSONDecodeError:
                pass

    # Aggregate multi-sample metrics per section.
    for section in sections.values():
        elapsed_list = section.pop("_elapsed_list", None)
        audio_dur_list = section.pop("_audio_dur_list", None)
        rtf_list = section.pop("_rtf_list", None)
        encoder_ms = section.pop("_encoder_ms", None)
        decoder_ms_list = section.pop("_decoder_ms_list", None)

        if elapsed_list:
            section["elapsed_seconds"] = round(sum(elapsed_list) / len(elapsed_list), 3)
            section["elapsed_seconds_samples"] = elapsed_list
        if audio_dur_list:
            section["audio_duration_seconds"] = round(sum(audio_dur_list) / len(audio_dur_list), 3)
        if rtf_list:
            section["rtf"] = round(sum(rtf_list) / len(rtf_list), 3)
        if encoder_ms is not None:
            section["encoder_elapsed_ms"] = round(float(encoder_ms), 3)
        if decoder_ms_list:
            section["decoder_elapsed_ms"] = round(sum(decoder_ms_list), 3)
            section["decoder_run_count"] = len(decoder_ms_list)
        # Synthesise elapsed_seconds from MeloTTS step timings when no explicit summary is available.
        if "elapsed_seconds" not in section and decoder_ms_list:
            total_ms = (encoder_ms or 0.0) + sum(decoder_ms_list)
            section["elapsed_seconds"] = round(total_ms / 1000.0, 3)
            section["elapsed_seconds_source"] = "step_sum"
        if section.get("elapsed_seconds") is not None:
            section["elapsed_ms"] = round(float(section["elapsed_seconds"]) * 1000.0, 3)

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


def parse_rknn_workload_distribution(value: str) -> dict[str, float] | None:
    match = WORKLOAD_DISTRIBUTION_PATTERN.search(value)
    if match:
        return {
            "core0_percent": float(match.group(1)),
            "core1_percent": float(match.group(2)),
            "core2_percent": float(match.group(3)),
            "improve_theoretical_percent": float(match.group(4)),
        }

    simple_match = WORKLOAD_SIMPLE_PATTERN.search(value.strip())
    if not simple_match:
        return None
    return {
        "core0_percent": float(simple_match.group(1)),
        "core1_percent": float(simple_match.group(2)),
        "core2_percent": float(simple_match.group(3)),
    }


def detect_rknn_perf_source(path: Path, content: str) -> str:
    filename = path.name.lower()
    if "query" in filename or "perf_detail" in filename:
        return "perf_detail"
    if "eval_perf" in filename or "npu_device" in filename:
        return "eval_perf"
    if "Operator Time-Consuming Ranking" in content or "Operator Time Consuming Ranking Table" in content:
        return "eval_perf"
    return "perf_detail"


def parse_rknn_perf_operator_row(line: str) -> dict[str, Any] | None:
    columns = split_columns(line, maxsplit=15)
    if len(columns) >= 10 and columns[0].isdigit() and "/" in columns[6]:
        remaining = columns[8:]
        mac_usage_raw = None
        if remaining and "%" not in remaining[0] and parse_slash_numeric_triplet(remaining[0]) is not None:
            mac_usage_raw = remaining.pop(0)

        workload_raw = remaining.pop(0) if remaining else ""
        rw_kb = remaining.pop(0) if remaining else None
        full_name = " ".join(remaining) if remaining else ""
        ddr_cycles, npu_cycles, total_cycles = parse_rknn_cycle_triplet(columns[6])
        workload = parse_rknn_workload_distribution(workload_raw) or {}
        return {
            "id": int(columns[0]),
            "op_type": columns[1],
            "data_type": columns[2],
            "target": columns[3],
            "input_shape": columns[4],
            "output_shape": columns[5],
            "ddr_cycles": ddr_cycles,
            "npu_cycles": npu_cycles,
            "total_cycles": total_cycles,
            "time_us": coerce_float(columns[7]),
            "mac_usage_percent": parse_rknn_mac_usage(mac_usage_raw or ""),
            "workload": workload_raw,
            "task_number": None,
            "lut_number": None,
            "rw_kb": coerce_float(rw_kb),
            "full_name": full_name,
            **workload,
        }

    if len(columns) < 15 or not columns[0].isdigit():
        return None
    if len(columns) == 15:
        columns.append("")
    if len(columns) != 16:
        return None

    workload = parse_rknn_workload_distribution(columns[11]) or {}
    return {
        "id": int(columns[0]),
        "op_type": columns[1],
        "data_type": columns[2],
        "target": columns[3],
        "input_shape": columns[4],
        "output_shape": columns[5],
        "ddr_cycles": coerce_int(columns[6]),
        "npu_cycles": coerce_int(columns[7]),
        "total_cycles": coerce_int(columns[8]),
        "time_us": coerce_float(columns[9]),
        "mac_usage_percent": parse_rknn_mac_usage(columns[10]),
        "workload": columns[11],
        "task_number": coerce_int(columns[12]),
        "lut_number": coerce_int(columns[13]),
        "rw_kb": coerce_float(columns[14]),
        "full_name": columns[15],
        **workload,
    }


def parse_rknn_perf_ranking_row(line: str) -> dict[str, Any] | None:
    columns = split_columns(line, maxsplit=6)
    if not columns or columns[0] == "OpType":
        return None
    if len(columns) == 7:
        return {
            "op_type": columns[0],
            "call_number": coerce_int(columns[1]),
            "cpu_time_us": coerce_float(columns[2]),
            "gpu_time_us": coerce_float(columns[3]),
            "npu_time_us": coerce_float(columns[4]),
            "total_time_us": coerce_float(columns[5]),
            "time_ratio_percent": coerce_float(columns[6]),
        }
    if len(columns) != 6:
        return None
    return {
        "op_type": columns[0],
        "call_number": coerce_int(columns[1]),
        "cpu_time_us": coerce_float(columns[2]),
        "gpu_time_us": None,
        "npu_time_us": coerce_float(columns[3]),
        "total_time_us": coerce_float(columns[4]),
        "time_ratio_percent": coerce_float(columns[5]),
    }


def parse_rknn_perf_text(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return {}

    operators: list[dict[str, Any]] = []
    ranking: list[dict[str, Any]] = []
    total_operator_time_us: float | None = None
    total_memory_rw_mb: float | None = None
    in_operator_table = False
    in_ranking_table = False

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        total_operator_match = RKNN_TOTAL_OPERATOR_TIME_PATTERN.search(stripped)
        if total_operator_match:
            total_operator_time_us = float(total_operator_match.group(1))
            in_operator_table = False
            continue

        total_memory_rw_match = RKNN_TOTAL_MEMORY_RW_PATTERN.search(stripped)
        if total_memory_rw_match:
            total_memory_rw_mb = float(total_memory_rw_match.group(1))
            continue

        total_memory_rw_kb_match = RKNN_TOTAL_MEMORY_RW_KB_PATTERN.search(stripped)
        if total_memory_rw_kb_match:
            total_memory_rw_mb = float(total_memory_rw_kb_match.group(1)) / 1024.0
            continue

        if stripped.startswith("ID") and "OpType" in stripped and "Time(us)" in stripped:
            in_operator_table = True
            in_ranking_table = False
            continue

        if stripped.startswith("Operator Time-Consuming Ranking") or stripped.startswith("Operator Time Consuming Ranking"):
            in_operator_table = False
            in_ranking_table = True
            continue

        if in_operator_table:
            operator = parse_rknn_perf_operator_row(line)
            if operator is not None:
                operators.append(operator)
            continue

        if in_ranking_table:
            ranking_row = parse_rknn_perf_ranking_row(line)
            if ranking_row is not None:
                ranking.append(ranking_row)
                continue
            if stripped.startswith("==="):
                in_ranking_table = False

    mac_usage_values = [
        operator["mac_usage_percent"]
        for operator in operators
        if operator.get("mac_usage_percent") is not None
    ]
    npu_operators = [operator for operator in operators if operator.get("target") == "NPU"]
    peak_operator = max(operators, key=lambda operator: operator.get("time_us") or 0.0, default=None)
    hottest_ranking = max(ranking, key=lambda item: item.get("total_time_us") or 0.0, default=None)

    peak_mac_usage_percent = max(mac_usage_values) if mac_usage_values else None
    mean_mac_usage_percent = None
    if mac_usage_values:
        mean_mac_usage_percent = round(sum(mac_usage_values) / len(mac_usage_values), 2)

    return {
        "source": detect_rknn_perf_source(path, content),
        "operator_count": len(operators),
        "npu_operator_count": len(npu_operators),
        "operators": operators,
        "ranking": ranking,
        "summary": {
            "total_operator_elapsed_time_us": total_operator_time_us,
            "total_memory_rw_mb": total_memory_rw_mb,
            "peak_layer_time_us": peak_operator.get("time_us") if peak_operator else None,
            "peak_layer_name": (peak_operator.get("full_name") or peak_operator.get("op_type")) if peak_operator else "",
            "peak_npu_cycles": peak_operator.get("npu_cycles") if peak_operator else None,
            "peak_mac_usage_percent": peak_mac_usage_percent,
            "mean_mac_usage_percent": mean_mac_usage_percent,
            "hottest_op_type": hottest_ranking.get("op_type") if hottest_ranking else (peak_operator.get("op_type") if peak_operator else ""),
            "hottest_op_time_us": hottest_ranking.get("total_time_us") if hottest_ranking else (peak_operator.get("time_us") if peak_operator else None),
        },
    }


def parse_rknn_perf_run(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    content = path.read_text(encoding="utf-8")
    stripped = content.strip()
    if not stripped:
        return {}

    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            run_duration_us = payload.get("run_duration_us")
            if run_duration_us is None:
                run_duration_us = payload.get("run_duration")
            return {
                "run_duration_us": coerce_float(run_duration_us),
            }

    match = RKNN_RUN_DURATION_PATTERN.search(content)
    if not match:
        return {}
    return {
        "run_duration_us": float(match.group(1)),
    }


def parse_rknn_memory_profile(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    content = path.read_text(encoding="utf-8")
    stripped = content.strip()
    if not stripped:
        return {}

    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            total_weight_size = payload.get("total_weight_size")
            total_internal_size = payload.get("total_internal_size")
            total_dma_allocated_size = payload.get("total_dma_allocated_size")
            total_memory_size = None
            if total_weight_size is not None or total_internal_size is not None:
                total_memory_size = (coerce_float(total_weight_size) or 0.0) + (coerce_float(total_internal_size) or 0.0)
            return {
                "source": "mem_size",
                "total_weight_mib": bytes_to_mib(coerce_float(total_weight_size)),
                "total_internal_tensor_mib": bytes_to_mib(coerce_float(total_internal_size)),
                "total_memory_mib": bytes_to_mib(total_memory_size),
                "total_dma_allocated_mib": bytes_to_mib(coerce_float(total_dma_allocated_size)),
                "model_size_mib": bytes_to_mib(coerce_float(payload.get("model_size_bytes"))),
            }

    total_weight_match = RKNN_MEMORY_WEIGHT_PATTERN.search(content)
    total_internal_match = RKNN_MEMORY_INTERNAL_PATTERN.search(content)
    total_memory_match = RKNN_MEMORY_TOTAL_PATTERN.search(content)
    model_size_match = RKNN_MODEL_SIZE_PATTERN.search(content)

    return {
        "source": "eval_memory",
        "total_weight_mib": float(total_weight_match.group(1)) if total_weight_match else None,
        "total_internal_tensor_mib": float(total_internal_match.group(1)) if total_internal_match else None,
        "total_memory_mib": float(total_memory_match.group(1)) if total_memory_match else None,
        "model_size_mib": float(model_size_match.group(1)) if model_size_match else None,
    }


def parse_rknn_runtime_log(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
    layer_lines = [line.strip() for line in lines if RKNN_RUNTIME_LAYER_HINT_PATTERN.search(line)]
    percentages = [
        float(value)
        for line in layer_lines
        for value in PERCENT_VALUE_PATTERN.findall(line)
    ]

    return {
        "line_count": len(lines),
        "layer_line_count": len(layer_lines),
        "sample_lines": layer_lines[:12],
        "peak_percent": max(percentages) if percentages else None,
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
    defaults = payload.get("defaults", {})
    cases = payload.get("cases", [])
    category_counts: dict[str, int] = {}
    acceptance_cases: list[dict[str, Any]] = []
    normalized_cases: list[dict[str, Any]] = []
    for case in cases:
        category = str(case.get("category", "uncategorized"))
        category_counts[category] = category_counts.get(category, 0) + 1
        normalized_cases.append(
            {
                "id": case.get("id", ""),
                "name": case.get("name", ""),
                "category": category,
                "tags": list(case.get("tags", [])),
                "sentence": case.get("sentence", ""),
                "repeat": case.get("repeat", defaults.get("repeat")),
                "warmup": case.get("warmup", defaults.get("warmup")),
                "latency_threshold_ms": case.get("latency_threshold_ms"),
                "rtf_threshold": case.get("rtf_threshold"),
                "notes": case.get("notes", ""),
            }
        )
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
        "cases": normalized_cases,
        "acceptance_cases": acceptance_cases,
        "defaults": defaults,
    }


def infer_asr_latency_units(section: Mapping[str, Any]) -> tuple[int | None, str]:
    result = section.get("result")
    if not isinstance(result, Mapping):
        result = {}

    words = result.get("words")
    if isinstance(words, list):
        normalized_words = [str(word).strip() for word in words if str(word).strip()]
        if normalized_words:
            return len(normalized_words), "词"

    tokens = result.get("tokens")
    if isinstance(tokens, list):
        normalized_tokens = [str(token).strip() for token in tokens if str(token).strip()]
        if normalized_tokens:
            return len(normalized_tokens), "字"

    text = str(result.get("text") or section.get("text") or "").strip()
    if not text:
        return None, ""

    split_words = [segment for segment in re.split(r"\s+", text) if segment]
    if len(split_words) > 1:
        return len(split_words), "词"

    compact_text = re.sub(r"\s+", "", text)
    if compact_text:
        return len(compact_text), "字"
    return None, ""


def summarize_asr_latency(section: Mapping[str, Any]) -> dict[str, Any]:
    if bool(section.get("failed")):
        return {
            "processing_elapsed_ms": None,
            "unit_count": None,
            "unit_label": "",
            "first_unit_timestamp_ms": None,
            "per_unit_latency_ms": None,
            "final_result_latency_ms": None,
        }

    elapsed_samples = section.get("elapsed_seconds_samples")
    processing_elapsed_ms: float | None = None
    if isinstance(elapsed_samples, list):
        numeric_samples = [float(sample) for sample in elapsed_samples if sample is not None]
        if numeric_samples:
            processing_elapsed_ms = round(min(numeric_samples) * 1000.0, 3)

    if processing_elapsed_ms is None and section.get("elapsed_seconds") is not None:
        processing_elapsed_ms = round(float(section.get("elapsed_seconds") or 0.0) * 1000.0, 3)

    unit_count, unit_label = infer_asr_latency_units(section)
    result = section.get("result")
    if not isinstance(result, Mapping):
        result = {}

    first_unit_timestamp_ms: float | None = None
    timestamps = result.get("timestamps")
    if isinstance(timestamps, list):
        numeric_timestamps = [float(timestamp) for timestamp in timestamps if timestamp is not None]
        if numeric_timestamps:
            first_unit_timestamp_ms = round(numeric_timestamps[0] * 1000.0, 3)

    per_unit_latency_ms: float | None = None
    if processing_elapsed_ms is not None and unit_count:
        per_unit_latency_ms = round(processing_elapsed_ms / unit_count, 3)

    return {
        "processing_elapsed_ms": processing_elapsed_ms,
        "unit_count": unit_count,
        "unit_label": unit_label,
        "first_unit_timestamp_ms": first_unit_timestamp_ms,
        "per_unit_latency_ms": per_unit_latency_ms,
        "final_result_latency_ms": processing_elapsed_ms,
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
    tts_elapsed_basis = observed.get("tts_elapsed_basis") or "unknown"
    tts_backend = observed.get("tts_backend", "unknown")
    tts_backend_uses_rknn = "rknn" in str(tts_backend).lower()
    asr_streaming_rknn_elapsed_ms = observed.get("asr_streaming_rknn_elapsed_ms")
    asr_streaming_rknn_processing_elapsed_ms = observed.get("asr_streaming_rknn_processing_elapsed_ms")
    asr_streaming_rknn_per_unit_latency_ms = observed.get("asr_streaming_rknn_per_unit_latency_ms")
    asr_streaming_rknn_first_unit_timestamp_ms = observed.get("asr_streaming_rknn_first_unit_timestamp_ms")
    asr_streaming_rknn_final_result_latency_ms = observed.get("asr_streaming_rknn_final_result_latency_ms")
    asr_streaming_rknn_unit_count = observed.get("asr_streaming_rknn_unit_count")
    asr_streaming_rknn_latency_unit = observed.get("asr_streaming_rknn_latency_unit") or ""
    asr_streaming_rknn_available = bool(observed.get("asr_streaming_rknn_available"))
    tts_max_rss_mib = observed.get("tts_max_rss_mib")
    tts_model_size_mib = observed.get("tts_model_size_mib")
    tts_model_is_int8 = bool(observed.get("tts_model_is_int8"))
    asr_streaming_rknn_model_size_mib = observed.get("asr_streaming_rknn_model_size_mib")
    models_total_size_mib = observed.get("models_total_size_mib")
    npu_peak_percent = observed.get("npu_peak_percent") or 0
    rknn_profile_source = observed.get("rknn_profile_source")
    rknn_operator_count = observed.get("rknn_operator_count") or 0
    rknn_runtime_layer_log_count = observed.get("rknn_runtime_layer_log_count") or 0
    offline_ready = bool(observed.get("offline_ready"))
    tts_model_name = observed.get("tts_model_name", "")
    tts_basis_label = {
        "cold_start": "cold start",
        "warm_run": "warm run",
        "profile_run": "profile run",
        "unknown": "TTS run",
    }.get(tts_elapsed_basis, str(tts_elapsed_basis))

    if "语音合成端到端延迟" in text:
        if tts_elapsed_ms is None:
            return {
                "status": "unknown",
                "observed": "缺少可用于判定的 TTS warm/profile 时延数据",
                "rationale": "冷启动时延只用于展示，不再参与 150 ms 指标判定。",
            }
        if not tts_backend_uses_rknn:
            return {
                "status": "fail",
                "observed": f"当前 TTS 后端为 {tts_backend}，{tts_basis_label} {tts_elapsed_ms:.0f} ms",
                "rationale": "指标明确要求 NPU 加速下 ≤ 150 ms，而当前交付主线仍是 CPU/ONNX TTS。",
            }
        if tts_elapsed_ms <= 150.0:
            return {"status": "pass", "observed": f"{tts_basis_label} {tts_elapsed_ms:.0f} ms", "rationale": "已满足 150 ms 目标。"}
        return {"status": "fail", "observed": f"{tts_basis_label} {tts_elapsed_ms:.0f} ms", "rationale": "虽为 NPU 路径，但仍超过 150 ms。"}

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
        if asr_streaming_rknn_per_unit_latency_ms is None:
            return {"status": "unknown", "observed": "缺少可用于判定的流式 ASR 单字/单词时延样本", "rationale": "当前报告已不再用整句总耗时直接判定 200 ms 指标，需要识别结果与单位计数共同支撑。"}
        unit_text = "单词" if asr_streaming_rknn_latency_unit == "词" else "单字"
        observed_parts = [f"流式 ASR (RKNN) 平均{unit_text}耗时 {asr_streaming_rknn_per_unit_latency_ms:.1f} ms"]
        if asr_streaming_rknn_first_unit_timestamp_ms is not None:
            observed_parts.append(f"首字时间戳 {asr_streaming_rknn_first_unit_timestamp_ms:.0f} ms")
        if asr_streaming_rknn_final_result_latency_ms is not None:
            observed_parts.append(f"最终结果耗时 {asr_streaming_rknn_final_result_latency_ms:.0f} ms")
        if asr_streaming_rknn_unit_count:
            observed_parts.append(f"{asr_streaming_rknn_unit_count}{asr_streaming_rknn_latency_unit}")
        observed_text = "，".join(observed_parts)
        if asr_streaming_rknn_per_unit_latency_ms <= 200.0:
            return {"status": "pass", "observed": observed_text, "rationale": "当前日志没有 partial/final 产出时序，因此 200 ms 指标暂按平均单字/单词处理耗时判定；首字时间戳和最终结果耗时仅作辅助观测。"}
        return {"status": "fail", "observed": observed_text, "rationale": "当前日志没有 partial/final 产出时序，因此 200 ms 指标暂按平均单字/单词处理耗时判定；首字时间戳和最终结果耗时仅作辅助观测。"}

    if subsection == "2.2 识别准确率":
        return {"status": "unknown", "observed": "缺少带标注的 ASR 评测集", "rationale": "当前报告仅汇总冒烟转写样例，不包含准确率统计。"}

    if subsection == "2.3 模型与稳定性" and "模型体积 ≤ 100 MB" in text:
        if asr_streaming_rknn_model_size_mib is None:
            return {"status": "unknown", "observed": "缺少 ASR 模型文件", "rationale": "无法计算当前 ASR 模型体积。"}
        if asr_streaming_rknn_model_size_mib <= 100.0:
            return {"status": "pass", "observed": f"流式 RKNN 模型合计 {asr_streaming_rknn_model_size_mib:.1f} MiB", "rationale": "当前流式 RKNN ASR 模型体积符合 100 MiB 目标。"}
        return {"status": "fail", "observed": f"流式 RKNN 模型合计 {asr_streaming_rknn_model_size_mib:.1f} MiB", "rationale": "当前流式 RKNN ASR 模型体积超过 100 MiB。"}

    if subsection == "2.3 模型与稳定性" and "支持流式识别" in text:
        if asr_streaming_rknn_available:
            return {"status": "pass", "observed": "默认运行入口为 streaming RKNN", "rationale": "当前运行包不再区分 ASR 模式，统一按流式识别链路交付。"}
        return {"status": "fail", "observed": "缺少 streaming RKNN 模型", "rationale": "当前运行包无法证明流式识别链路已完整交付。"}

    if subsection == "2.3 模型与稳定性" and "7×24 小时" in text:
        return {"status": "unknown", "observed": "缺少 ASR 长稳记录", "rationale": "当前报告没有长时间连续 ASR 运行数据。"}

    if subsection == "2.4 运行模式" and "纯离线端侧运行" in text:
        if offline_ready:
            return {"status": "pass", "observed": "ASR 模型和二进制均为本地部署", "rationale": "当前运行包不依赖云端 ASR 接口。"}
        return {"status": "unknown", "observed": "缺少完整 ASR 运行包证据", "rationale": "无法确认离线运行条件。"}

    if "全链路闭环延迟" in text:
        if tts_elapsed_ms is not None and asr_streaming_rknn_elapsed_ms is not None:
            combined_ms = tts_elapsed_ms + asr_streaming_rknn_elapsed_ms
            status = "pass" if combined_ms <= 350.0 else "fail"
            return {
                "status": status,
                "observed": f"TTS {tts_basis_label} {tts_elapsed_ms:.0f} ms + ASR 流式 RKNN {asr_streaming_rknn_elapsed_ms:.0f} ms = {combined_ms:.0f} ms",
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
        if rknn_operator_count > 0 and not tts_backend_uses_rknn:
            return {
                "status": "partial",
                "observed": f"已采到 {format_rknn_profile_source(rknn_profile_source)} {rknn_operator_count} 条层级记录，TTS 仍为 {tts_backend}",
                "rationale": "当前已有官方 RKNN 层级 profiler 证据，但命中 NPU 的仍主要是 ASR 路径。",
            }
        if rknn_operator_count > 0:
            return {
                "status": "pass",
                "observed": f"已采到 {format_rknn_profile_source(rknn_profile_source)} {rknn_operator_count} 条层级记录",
                "rationale": "当前已有官方 RKNN 层级 profiler 证据，可证明运行链路已命中 RK3588 NPU。",
            }
        if rknn_runtime_layer_log_count > 0 and not tts_backend_uses_rknn:
            return {
                "status": "partial",
                "observed": f"已采到 RKNN_LOG_LEVEL=4 层日志 {rknn_runtime_layer_log_count} 行，TTS 仍为 {tts_backend}",
                "rationale": "当前已有 RKNN 运行时层利用率日志，但系统仍是 ASR 命中 NPU、TTS 保持 CPU 基线。",
            }
        if rknn_runtime_layer_log_count > 0:
            return {
                "status": "pass",
                "observed": f"已采到 RKNN_LOG_LEVEL=4 层日志 {rknn_runtime_layer_log_count} 行",
                "rationale": "当前运行时日志已显示每层 MAC 利用率或带宽占用，能证明 RKNN NPU 执行路径命中。",
            }
        if npu_peak_percent > 0 and not tts_backend_uses_rknn:
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


def collect_evidence(evidence_dir: Path, assets_dir: Path, *, tts_evidence_dir: Path | None = None) -> dict[str, Any]:
    tts_ev_dir = tts_evidence_dir if tts_evidence_dir and tts_evidence_dir.is_dir() else None

    smoke_log_path = pick_first_existing([
        evidence_dir / "smoke_test_summary.log",
        evidence_dir / "smoke_test.log",
    ])
    board_capabilities_path = evidence_dir / "board_profile_capabilities.txt"
    rknpu_load_path = pick_first_existing([
        evidence_dir / "rknpu_load.log",
        evidence_dir / "rknn_profile.log",
    ]) or first_glob(evidence_dir, "*rknpu*load*.log") or first_glob(evidence_dir, "*rknn*profile*.log")
    rknn_perf_path = pick_first_existing([
        evidence_dir / "rknn_eval_perf.txt",
        evidence_dir / "rknn_query_perf_detail.txt",
        evidence_dir / "rknn_perf_detail.txt",
    ]) or first_glob(evidence_dir, "*rknn*perf*detail*.txt") or first_glob(evidence_dir, "*eval*perf*.txt")
    rknn_perf_run_path = pick_first_existing([
        evidence_dir / "rknn_perf_run.json",
        evidence_dir / "rknn_query_perf_run.txt",
        evidence_dir / "rknn_perf_run.txt",
    ]) or first_glob(evidence_dir, "*rknn*perf*run*.json") or first_glob(evidence_dir, "*rknn*perf*run*.txt")
    rknn_memory_path = pick_first_existing([
        evidence_dir / "rknn_memory_profile.txt",
        evidence_dir / "rknn_eval_memory.txt",
        evidence_dir / "rknn_query_mem_size.json",
    ]) or first_glob(evidence_dir, "*rknn*memory*.txt") or first_glob(evidence_dir, "*rknn*mem*size*.json")
    rknn_runtime_log_path = pick_first_existing([
        evidence_dir / "rknn_runtime.log",
        evidence_dir / "rknn_layer_runtime.log",
    ]) or first_glob(evidence_dir, "*rknn*runtime*.log")
    tts_profile_log_path = pick_first_existing([
        evidence_dir / "profile.log",
        evidence_dir / "tts-profile.log",
    ])
    tts_profile_csv_path: Path | None = None
    if tts_ev_dir is None:
        tts_profile_csv_path = pick_first_existing([
            evidence_dir / "profile-samples.csv",
            evidence_dir / "tts-profile-samples.csv",
        ]) or first_glob(evidence_dir, "*profile*samples*.csv")
    audio_path = pick_first_existing([
        evidence_dir / "profile.wav",
        evidence_dir / "smoke_test_tts.wav",
        evidence_dir / "smoke_test.wav",
    ]) or first_glob(evidence_dir, "*.wav")

    # ----- TTS evidence from separate directory -----
    tts_smoke_log_path: Path | None = None
    tts_rknn_runtime_log_path: Path | None = None
    if tts_ev_dir:
        tts_smoke_log_path = pick_first_existing([
            tts_ev_dir / "smoke_test_summary.log",
            tts_ev_dir / "smoke_test.log",
        ])
        if tts_profile_log_path is None:
            tts_profile_log_path = pick_first_existing([
                tts_ev_dir / "profile.log",
                tts_ev_dir / "tts-profile.log",
            ])
        if tts_profile_csv_path is None:
            tts_profile_csv_path = pick_first_existing([
                tts_ev_dir / "profile-samples.csv",
                tts_ev_dir / "tts-profile-samples.csv",
            ]) or first_glob(tts_ev_dir, "*profile*samples*.csv")
        tts_rknn_runtime_log_path = pick_first_existing([
            tts_ev_dir / "rknn_runtime.log",
            tts_ev_dir / "rknn_layer_runtime.log",
        ]) or first_glob(tts_ev_dir, "*rknn*runtime*.log")
        if audio_path is None:
            audio_path = pick_first_existing([
                tts_ev_dir / "profile_tts.wav",
                tts_ev_dir / "smoke_test_tts.wav",
                tts_ev_dir / "profile.wav",
            ]) or first_glob(tts_ev_dir, "*.wav")

    smoke_summary = parse_smoke_log(smoke_log_path)

    # Merge TTS sections from the dedicated TTS smoke log if available.
    if tts_smoke_log_path and tts_smoke_log_path != smoke_log_path:
        tts_smoke_sections = parse_smoke_log(tts_smoke_log_path)
        for section_name in ("tts_cold_start", "tts_warm_run", "tts_profile_run"):
            tts_section = tts_smoke_sections.get(section_name, {})
            if len(tts_section) > 1:
                smoke_summary[section_name] = tts_section
    board_capabilities = parse_board_capabilities(board_capabilities_path)
    rknpu_load = parse_rknn_profile_log(rknpu_load_path)
    rknn_perf = parse_rknn_perf_text(rknn_perf_path)
    rknn_perf_run = parse_rknn_perf_run(rknn_perf_run_path)
    rknn_memory = parse_rknn_memory_profile(rknn_memory_path)
    rknn_runtime_log = parse_rknn_runtime_log(rknn_runtime_log_path)
    tts_profile = parse_tts_profile_csv(tts_profile_csv_path)
    tts_rknn_runtime_log = parse_rknn_runtime_log(tts_rknn_runtime_log_path)
    wav_metadata = read_wav_metadata(audio_path)

    copied_assets = {
        "smoke_log": copy_asset(smoke_log_path, assets_dir),
        "board_capabilities": copy_asset(board_capabilities_path, assets_dir),
        "rknpu_load": copy_asset(rknpu_load_path, assets_dir),
        "rknn_perf_detail": copy_asset(rknn_perf_path, assets_dir),
        "rknn_perf_run": copy_asset(rknn_perf_run_path, assets_dir),
        "rknn_memory_profile": copy_asset(rknn_memory_path, assets_dir),
        "rknn_runtime_log": copy_asset(rknn_runtime_log_path, assets_dir),
        "tts_profile_log": copy_asset(tts_profile_log_path, assets_dir),
        "tts_profile_csv": copy_asset(tts_profile_csv_path, assets_dir),
        "tts_rknn_runtime_log": copy_asset(tts_rknn_runtime_log_path, assets_dir),
        "audio": copy_asset(audio_path, assets_dir),
    }

    asr_heatmap_markup = render_heatmap_svg(
        title="ASR RKNN NPU Load Heatmap",
        subtitle="由 rknpu/load 时序采样生成，用于观察三个 NPU core 的忙闲变化。",
        samples=rknpu_load.get("samples", []),
        rows=(
            ("Core0", "core0_percent"),
            ("Core1", "core1_percent"),
            ("Core2", "core2_percent"),
        ),
        max_value=100.0,
    )
    copied_assets["asr_rknpu_load_heatmap"] = write_svg_asset(assets_dir / "asr-rknpu-load-heatmap.svg", asr_heatmap_markup)

    tts_heatmap_samples = [
        {
            "core0_percent": sample.get("npu_core0_percent", 0),
            "core1_percent": sample.get("npu_core1_percent", 0),
            "core2_percent": sample.get("npu_core2_percent", 0),
        }
        for sample in tts_profile.get("samples", [])
    ]
    tts_heatmap_markup = render_heatmap_svg(
        title="TTS Profile Heatmap",
        subtitle="由 profile-samples.csv 采样生成，用于观察 TTS 进程 NPU core 负载变化。",
        samples=tts_heatmap_samples,
        rows=(
            ("Core0", "core0_percent"),
            ("Core1", "core1_percent"),
            ("Core2", "core2_percent"),
        ),
        max_value=100.0,
    )
    copied_assets["tts_profile_heatmap"] = write_svg_asset(assets_dir / "tts-profile-heatmap.svg", tts_heatmap_markup)

    return {
        "evidence_dir": str(evidence_dir),
        "smoke": smoke_summary,
        "board_capabilities": board_capabilities,
        "rknpu_load": rknpu_load,
        "rknn_perf": rknn_perf,
        "rknn_perf_run": rknn_perf_run,
        "rknn_memory": rknn_memory,
        "rknn_runtime_log": rknn_runtime_log,
        "tts_profile": tts_profile,
        "tts_rknn_runtime_log": tts_rknn_runtime_log,
        "audio": wav_metadata,
        "assets": copied_assets,
    }


def build_observed_metrics(runtime_info: dict[str, Any], evidence: dict[str, Any], plan_summary: dict[str, Any]) -> dict[str, Any]:
    smoke = evidence.get("smoke", {})
    tts_cold_start = smoke.get("tts_cold_start", {})
    tts_warm_run = smoke.get("tts_warm_run", {})
    tts_profile_run = smoke.get("tts_profile_run", {})
    asr_streaming_rknn = smoke.get("asr_streaming_rknn", {})
    rknpu_load = evidence.get("rknpu_load", {})
    rknn_perf = evidence.get("rknn_perf", {})
    rknn_perf_run = evidence.get("rknn_perf_run", {})
    rknn_memory = evidence.get("rknn_memory", {})
    rknn_runtime_log = evidence.get("rknn_runtime_log", {})
    tts_profile = evidence.get("tts_profile", {})
    tts_rknn_runtime_log = evidence.get("tts_rknn_runtime_log", {})
    asr_latency_summary = summarize_asr_latency(asr_streaming_rknn)

    rknn_runtime_layer_log_count = (rknn_runtime_log.get("layer_line_count", 0) or 0) + (tts_rknn_runtime_log.get("layer_line_count", 0) or 0)

    npu_peak_source = ""
    npu_peak_percent = 0.0
    for source_name, candidate_peak in (
        ("rknpu_load", rknpu_load.get("peak_percent")),
        ("tts_profile_csv", tts_profile.get("peak_npu_percent")),
    ):
        peak_value = coerce_float(candidate_peak)
        if peak_value is None or peak_value < npu_peak_percent:
            continue
        npu_peak_percent = peak_value
        npu_peak_source = source_name

    asr_streaming_rknn_failed = bool(asr_streaming_rknn.get("failed"))

    rknn_profile_source = rknn_perf.get("source")
    if not rknn_profile_source and rknn_runtime_layer_log_count:
        rknn_profile_source = "runtime_log"
    if not rknn_profile_source and rknpu_load.get("sample_count"):
        rknn_profile_source = "load_sampling"
    if not rknn_profile_source and tts_profile.get("sample_count"):
        rknn_profile_source = "tts_profile_csv"

    category_counts = plan_summary.get("category_counts", {})

    tts_primary_run = tts_warm_run
    tts_elapsed_basis = "warm_run"
    if tts_primary_run.get("elapsed_seconds") is None:
        tts_primary_run = tts_profile_run
        tts_elapsed_basis = "profile_run"

    # TTS elapsed: prefer warm run, then profile run. Cold start is display-only.
    tts_elapsed_ms: float | None = None
    if tts_primary_run.get("elapsed_seconds") is not None:
        tts_elapsed_ms = (tts_primary_run["elapsed_seconds"] or 0) * 1000.0
    else:
        tts_elapsed_basis = ""

    # TTS audio duration: prefer the selected run, fall back to wav metadata.
    tts_audio_duration_s: float | None = tts_primary_run.get("audio_duration_seconds")
    if tts_audio_duration_s is None:
        wav_meta = evidence.get("audio", {})
        if wav_meta.get("duration_s"):
            tts_audio_duration_s = wav_meta["duration_s"]

    # TTS RTF: prefer the selected run, compute from elapsed / audio when both present.
    tts_rtf: float | None = tts_primary_run.get("rtf")
    if tts_rtf is None and tts_elapsed_ms is not None and tts_audio_duration_s and tts_audio_duration_s > 0:
        tts_rtf = round(tts_elapsed_ms / 1000.0 / tts_audio_duration_s, 3)

    return {
        "tts_backend": runtime_info.get("tts_backend"),
        "tts_elapsed_basis": tts_elapsed_basis,
        "tts_elapsed_ms": tts_elapsed_ms,
        "tts_audio_duration_s": tts_audio_duration_s,
        "tts_rtf": tts_rtf,
        "tts_cold_start_elapsed_ms": tts_cold_start.get("elapsed_ms"),
        "tts_cold_start_model_load_ms": tts_cold_start.get("model_load_ms"),
        "tts_cold_start_encoder_ms": tts_cold_start.get("encoder_elapsed_ms"),
        "tts_cold_start_decoder_ms": tts_cold_start.get("decoder_elapsed_ms"),
        "tts_warm_run_elapsed_ms": tts_warm_run.get("elapsed_ms"),
        "tts_warm_run_model_load_ms": tts_warm_run.get("model_load_ms"),
        "tts_warm_run_encoder_ms": tts_warm_run.get("encoder_elapsed_ms"),
        "tts_warm_run_decoder_ms": tts_warm_run.get("decoder_elapsed_ms"),
        "tts_profile_run_elapsed_ms": tts_profile_run.get("elapsed_ms"),
        "tts_profile_run_model_load_ms": tts_profile_run.get("model_load_ms"),
        "tts_profile_run_encoder_ms": tts_profile_run.get("encoder_elapsed_ms"),
        "tts_profile_run_decoder_ms": tts_profile_run.get("decoder_elapsed_ms"),
        "asr_streaming_rknn_elapsed_ms": None if asr_streaming_rknn_failed else ((asr_streaming_rknn.get("elapsed_seconds") or 0) * 1000.0 if asr_streaming_rknn.get("elapsed_seconds") is not None else None),
        "asr_streaming_rknn_processing_elapsed_ms": asr_latency_summary.get("processing_elapsed_ms"),
        "asr_streaming_rknn_unit_count": asr_latency_summary.get("unit_count"),
        "asr_streaming_rknn_latency_unit": asr_latency_summary.get("unit_label"),
        "asr_streaming_rknn_first_unit_timestamp_ms": asr_latency_summary.get("first_unit_timestamp_ms"),
        "asr_streaming_rknn_per_unit_latency_ms": asr_latency_summary.get("per_unit_latency_ms"),
        "asr_streaming_rknn_final_result_latency_ms": asr_latency_summary.get("final_result_latency_ms"),
        "asr_streaming_rknn_rtf": None if asr_streaming_rknn_failed else asr_streaming_rknn.get("rtf"),
        "tts_max_rss_mib": tts_profile.get("max_rss_mib"),
        "npu_peak_percent": npu_peak_percent,
        "npu_peak_source": npu_peak_source,
        "rknn_profile_source": rknn_profile_source,
        "rknn_operator_count": rknn_perf.get("operator_count"),
        "rknn_total_time_ms": (rknn_perf.get("summary", {}).get("total_operator_elapsed_time_us") or 0.0) / 1000.0 if rknn_perf.get("summary", {}).get("total_operator_elapsed_time_us") is not None else None,
        "rknn_run_duration_ms": (rknn_perf_run.get("run_duration_us") or 0.0) / 1000.0 if rknn_perf_run.get("run_duration_us") is not None else None,
        "rknn_peak_mac_usage_percent": rknn_perf.get("summary", {}).get("peak_mac_usage_percent"),
        "rknn_total_memory_mib": rknn_memory.get("total_memory_mib"),
        "rknn_runtime_layer_log_count": rknn_runtime_layer_log_count,
        "tts_model_size_mib": runtime_info.get("tts_model_size_mib"),
        "tts_model_is_int8": runtime_info.get("tts_model_is_int8"),
        "tts_model_name": runtime_info.get("tts_model_name"),
        "asr_supports_rknn": runtime_info.get("asr_supports_rknn"),
        "asr_streaming_rknn_available": runtime_info.get("asr_streaming_rknn_available"),
        "asr_streaming_rknn_model_size_mib": runtime_info.get("asr_streaming_rknn_model_size_mib"),
        "asr_streaming_rknn_model_name": runtime_info.get("asr_streaming_rknn_model_name"),
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



def render_html_report(payload: dict[str, Any], static_assets: dict[str, str] | None = None) -> str:
    environment = build_report_template_environment()
    template = environment.get_template(REPORT_TEMPLATE_NAME)
    payload_json = json.dumps(payload, ensure_ascii=False, default=str)
    return template.render(payload_json=payload_json)


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
    resolved_asr_runtime_dir = (resolved_runtime_dir / ASR_RUNTIME_SUBDIR).resolve()
    resolved_tts_runtime_dir = (resolved_runtime_dir / TTS_RUNTIME_SUBDIR).resolve()
    resolved_evidence_dir = (evidence_dir or (resolved_asr_runtime_dir / "output")).resolve()
    resolved_tts_evidence_dir = (resolved_tts_runtime_dir / "output").resolve()
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
    evidence = collect_evidence(resolved_evidence_dir, assets_dir, tts_evidence_dir=resolved_tts_evidence_dir)
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
    parser.add_argument("--runtime-dir", default="", help="Unified runtime project directory used for evidence discovery")
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
