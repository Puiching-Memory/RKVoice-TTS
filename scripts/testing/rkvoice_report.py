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

from jinja2 import Environment, FileSystemLoader, select_autoescape


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "artifacts" / "test-runs"
DEFAULT_RUNTIME_DIR = WORKSPACE_ROOT / "artifacts" / "runtime" / "sherpa_onnx_rk3588_runtime"
DEFAULT_REQUIREMENTS_PATH = WORKSPACE_ROOT / "docs" / "requirements" / "项目指标.md"
DEFAULT_LOCAL_PLAN_PATH = WORKSPACE_ROOT / "config" / "local" / "tts_test_plan.json"
DEFAULT_EXAMPLE_PLAN_PATH = WORKSPACE_ROOT / "config" / "examples" / "tts_test_plan.example.json"
REPORT_TEMPLATE_DIR = Path(__file__).with_name("report_templates")
REPORT_STATIC_DIR = Path(__file__).with_name("report_static")
REPORT_TEMPLATE_NAME = "rkvoice_report.html.j2"
REPORT_STATIC_ROOT_NAME = "report-static"
REPORT_STATIC_FILE_MAP = {
    "tabler_css": Path("vendor") / "tabler.min.css",
    "tabler_js": Path("vendor") / "tabler.min.js",
    "report_css": Path("rkvoice-report.css"),
}

ELAPSED_SECONDS_PATTERN = re.compile(r"Elapsed seconds:\s*([0-9.]+)\s*s")
AUDIO_DURATION_PATTERN = re.compile(r"Audio duration:\s*([0-9.]+)\s*s")
RTF_PATTERN = re.compile(r"Real[- ]time factor(?: \(RTF\))?:.*?=\s*([0-9.]+)")
TEXT_PATTERN = re.compile(r"The text is:\s*(.+?)\.\s*Speaker ID:")
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


def default_report_static_assets() -> dict[str, str]:
    return {
        name: (Path("assets") / REPORT_STATIC_ROOT_NAME / relative_path).as_posix()
        for name, relative_path in REPORT_STATIC_FILE_MAP.items()
    }


def materialize_report_static_assets(assets_dir: Path) -> dict[str, str]:
    target_root = assets_dir / REPORT_STATIC_ROOT_NAME
    target_root.mkdir(parents=True, exist_ok=True)

    for source in REPORT_STATIC_DIR.rglob("*"):
        if not source.is_file():
            continue
        destination = target_root / source.relative_to(REPORT_STATIC_DIR)
        ensure_parent(destination)
        shutil.copy2(source, destination)

    return {
        name: relative_posix(target_root / relative_path, assets_dir.parent)
        for name, relative_path in REPORT_STATIC_FILE_MAP.items()
    }


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
    if ".rknn" in script_lower or "provider=rknn" in script_lower:
        return "rknn"
    if "onnx" in script_lower or "sherpa-onnx-offline-tts" in script_lower:
        return "cpu-onnx"
    return "unknown"


def detect_asr_mode(run_asr_script: str) -> str:
    import re

    m = re.search(r'RKVOICE_ASR_MODE:-(\w+)', run_asr_script)
    if m:
        default_mode = m.group(1).lower()
        if "stream" in default_mode:
            return "streaming"
        if default_mode == "offline":
            return "offline"
        return default_mode
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
    asr_streaming_model_dir = runtime_dir / "models" / "asr" / "streaming" / "streaming-zipformer-multi-zh-hans"
    asr_streaming_encoder_path = asr_streaming_model_dir / "encoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx"

    return {
        "runtime_dir": str(runtime_dir),
        "tts_backend": detect_tts_backend(run_tts_script),
        "tts_supports_rknn": ".rknn" in run_tts_script.lower() or "provider=rknn" in run_tts_script.lower(),
        "asr_mode": detect_asr_mode(run_asr_script),
        "asr_supports_rknn": "provider=rknn" in run_asr_script.lower(),
        "asr_streaming_available": asr_streaming_encoder_path.exists(),
        "tts_model_is_int8": "int8" in tts_model_path.name.lower(),
        "asr_cpu_model_is_int8": "int8" in asr_cpu_model_path.name.lower(),
        "asr_streaming_model_is_int8": "int8" in asr_streaming_encoder_path.name.lower(),
        "tts_model_size_mib": bytes_to_mib(file_size_bytes(tts_model_path)),
        "asr_cpu_model_size_mib": bytes_to_mib(file_size_bytes(asr_cpu_model_path)),
        "asr_rknn_model_size_mib": bytes_to_mib(file_size_bytes(asr_rknn_model_path)),
        "asr_streaming_model_size_mib": bytes_to_mib(directory_size_bytes(asr_streaming_model_dir)) if asr_streaming_model_dir.exists() else None,
        "models_total_size_mib": bytes_to_mib(directory_size_bytes(runtime_dir / "models")),
        "offline_ready": (runtime_dir / "bin").exists() and (runtime_dir / "models").exists(),
        "tts_model_name": tts_model_path.parent.name if tts_model_path.parent.exists() else "",
        "asr_cpu_model_name": asr_cpu_model_path.parent.name if asr_cpu_model_path.parent.exists() else "",
        "asr_rknn_model_name": asr_rknn_model_path.parent.name if asr_rknn_model_path.parent.exists() else "",
        "asr_streaming_model_name": asr_streaming_model_dir.name if asr_streaming_model_dir.exists() else "",
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
        "asr_streaming": {"label": "Streaming ASR smoke"},
        "asr_cpu": {"label": "CPU ASR smoke"},
        "asr_rknn": {"label": "RKNN ASR smoke"},
    }
    current_section: str | None = None

    section_pattern = re.compile(r"^\[\d+/\d+\]\s+(.+)$")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        header_match = section_pattern.match(line)
        if header_match:
            header = header_match.group(1).lower()
            if "tts" in header:
                current_section = "tts"
            elif "streaming" in header:
                current_section = "asr_streaming"
            elif "rknn" in header:
                current_section = "asr_rknn"
            elif "asr" in header:
                current_section = "asr_cpu"
            else:
                current_section = None
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
    asr_streaming_elapsed_ms = observed.get("asr_streaming_elapsed_ms")
    asr_mode = observed.get("asr_mode", "unknown")
    tts_max_rss_mib = observed.get("tts_max_rss_mib")
    tts_model_size_mib = observed.get("tts_model_size_mib")
    tts_model_is_int8 = bool(observed.get("tts_model_is_int8"))
    asr_cpu_model_size_mib = observed.get("asr_cpu_model_size_mib")
    asr_streaming_model_size_mib = observed.get("asr_streaming_model_size_mib")
    models_total_size_mib = observed.get("models_total_size_mib")
    npu_peak_percent = observed.get("npu_peak_percent") or 0
    rknn_profile_source = observed.get("rknn_profile_source")
    rknn_operator_count = observed.get("rknn_operator_count") or 0
    rknn_runtime_layer_log_count = observed.get("rknn_runtime_layer_log_count") or 0
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
        if asr_mode == "streaming":
            asr_latency_ms = asr_streaming_elapsed_ms
            if asr_latency_ms is None:
                return {"status": "unknown", "observed": "缺少流式 ASR 时延样本", "rationale": "未找到可用于 200 ms 判定的流式 ASR 冒烟证据。"}
            if asr_latency_ms <= 200.0:
                return {"status": "pass", "observed": f"流式 ASR 冒烟 {asr_latency_ms:.0f} ms", "rationale": "流式 ASR 延迟满足目标。"}
            return {"status": "fail", "observed": f"流式 ASR 冒烟 {asr_latency_ms:.0f} ms", "rationale": "流式 ASR 延迟超过目标。"}
        observed_text = "当前 ASR 入口为 offline"
        if asr_rknn_elapsed_ms is not None:
            observed_text += f"，RKNN 样本 {asr_rknn_elapsed_ms:.0f} ms"
        return {"status": "fail", "observed": observed_text, "rationale": "指标要求流式识别，而当前 run_asr.sh 和实测证据均为离线识别链路。"}

    if subsection == "2.2 识别准确率":
        return {"status": "unknown", "observed": "缺少带标注的 ASR 评测集", "rationale": "当前报告仅汇总冒烟转写样例，不包含准确率统计。"}

    if subsection == "2.3 模型与稳定性" and "模型体积 ≤ 100 MB" in text:
        if asr_mode == "streaming" and asr_streaming_model_size_mib is not None:
            if asr_streaming_model_size_mib <= 100.0:
                return {"status": "pass", "observed": f"流式模型 INT8 合计 {asr_streaming_model_size_mib:.1f} MiB", "rationale": "当前默认流式 ASR 模型体积符合 100 MiB 目标。"}
            return {"status": "fail", "observed": f"流式模型 INT8 合计 {asr_streaming_model_size_mib:.1f} MiB", "rationale": "当前流式 ASR 模型体积超过 100 MiB。"}
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
        if rknn_operator_count > 0 and tts_backend != "rknn":
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
        if rknn_runtime_layer_log_count > 0 and tts_backend != "rknn":
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
    rknpu_load = parse_rknn_profile_log(rknpu_load_path)
    rknn_perf = parse_rknn_perf_text(rknn_perf_path)
    rknn_perf_run = parse_rknn_perf_run(rknn_perf_run_path)
    rknn_memory = parse_rknn_memory_profile(rknn_memory_path)
    rknn_runtime_log = parse_rknn_runtime_log(rknn_runtime_log_path)
    tts_profile = parse_tts_profile_csv(tts_profile_csv_path)
    wav_metadata = read_wav_metadata(audio_path)

    copied_assets = {
        "smoke_log": copy_asset(smoke_log_path, assets_dir),
        "board_capabilities": copy_asset(board_capabilities_path, assets_dir),
        "rknpu_load": copy_asset(rknpu_load_path, assets_dir),
        "rknn_perf_detail": copy_asset(rknn_perf_path, assets_dir),
        "rknn_perf_run": copy_asset(rknn_perf_run_path, assets_dir),
        "rknn_memory_profile": copy_asset(rknn_memory_path, assets_dir),
        "rknn_runtime_log": copy_asset(rknn_runtime_log_path, assets_dir),
        "tts_profile_csv": copy_asset(tts_profile_csv_path, assets_dir),
        "tts_profile_log": copy_asset(tts_profile_log_path, assets_dir),
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
    tts_heatmap_markup = render_heatmap_svg(
        title="TTS Process Sampling Heatmap",
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

    copied_assets["asr_rknpu_load_heatmap"] = write_svg_asset(assets_dir / "asr-rknpu-load-heatmap.svg", asr_heatmap_markup)
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
        "audio": wav_metadata,
        "assets": copied_assets,
    }


def build_observed_metrics(runtime_info: dict[str, Any], evidence: dict[str, Any], plan_summary: dict[str, Any]) -> dict[str, Any]:
    smoke = evidence.get("smoke", {})
    tts_smoke = smoke.get("tts", {})
    asr_streaming = smoke.get("asr_streaming", {})
    asr_rknn = smoke.get("asr_rknn", {})
    tts_profile = evidence.get("tts_profile", {})
    rknpu_load = evidence.get("rknpu_load", {})
    rknn_perf = evidence.get("rknn_perf", {})
    rknn_perf_run = evidence.get("rknn_perf_run", {})
    rknn_memory = evidence.get("rknn_memory", {})
    rknn_runtime_log = evidence.get("rknn_runtime_log", {})

    rknn_profile_source = rknn_perf.get("source")
    if not rknn_profile_source and rknn_runtime_log.get("layer_line_count"):
        rknn_profile_source = "runtime_log"
    if not rknn_profile_source and rknpu_load.get("sample_count"):
        rknn_profile_source = "load_sampling"

    category_counts = plan_summary.get("category_counts", {})
    return {
        "tts_backend": runtime_info.get("tts_backend"),
        "asr_mode": runtime_info.get("asr_mode"),
        "tts_elapsed_ms": (tts_smoke.get("elapsed_seconds") or 0) * 1000.0 if tts_smoke.get("elapsed_seconds") is not None else None,
        "tts_audio_duration_s": tts_smoke.get("audio_duration_seconds"),
        "tts_rtf": tts_smoke.get("rtf"),
        "asr_streaming_elapsed_ms": (asr_streaming.get("elapsed_seconds") or 0) * 1000.0 if asr_streaming.get("elapsed_seconds") is not None else None,
        "asr_streaming_rtf": asr_streaming.get("rtf"),
        "asr_rknn_elapsed_ms": (asr_rknn.get("elapsed_seconds") or 0) * 1000.0 if asr_rknn.get("elapsed_seconds") is not None else None,
        "asr_rknn_rtf": asr_rknn.get("rtf"),
        "tts_max_rss_mib": tts_profile.get("max_rss_mib"),
        "npu_peak_percent": max(rknpu_load.get("peak_percent", 0), tts_profile.get("peak_npu_percent", 0)),
        "rknn_profile_source": rknn_profile_source,
        "rknn_operator_count": rknn_perf.get("operator_count"),
        "rknn_total_time_ms": (rknn_perf.get("summary", {}).get("total_operator_elapsed_time_us") or 0.0) / 1000.0 if rknn_perf.get("summary", {}).get("total_operator_elapsed_time_us") is not None else None,
        "rknn_run_duration_ms": (rknn_perf_run.get("run_duration_us") or 0.0) / 1000.0 if rknn_perf_run.get("run_duration_us") is not None else None,
        "rknn_peak_mac_usage_percent": rknn_perf.get("summary", {}).get("peak_mac_usage_percent"),
        "rknn_total_memory_mib": rknn_memory.get("total_memory_mib"),
        "rknn_runtime_layer_log_count": rknn_runtime_log.get("layer_line_count", 0),
        "tts_model_size_mib": runtime_info.get("tts_model_size_mib"),
        "tts_model_is_int8": runtime_info.get("tts_model_is_int8"),
        "tts_model_name": runtime_info.get("tts_model_name"),
        "asr_cpu_model_size_mib": runtime_info.get("asr_cpu_model_size_mib"),
        "asr_streaming_model_size_mib": runtime_info.get("asr_streaming_model_size_mib"),
        "asr_streaming_available": runtime_info.get("asr_streaming_available"),
        "asr_streaming_model_name": runtime_info.get("asr_streaming_model_name"),
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


def render_fact_item(label: str, value: Any, detail: str = "") -> str:
    value_text = "n/a" if value in {None, ""} else str(value)
    detail_markup = f'<div class="fact-detail">{html.escape(detail)}</div>' if detail else ""
    return (
        '<div class="fact-row">'
        f'<div class="fact-label">{html.escape(label)}</div>'
        f'<div class="fact-value">{html.escape(value_text)}</div>'
        f'{detail_markup}'
        '</div>'
    )


def _render_legacy_html_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    runtime = payload["runtime"]
    observed = payload["observed"]
    evidence = payload["evidence"]
    unit_tests = payload.get("unit_tests")
    requirement_summary = payload["requirements"]["summary"]
    evidence_assets = evidence["assets"]
    smoke = evidence.get("smoke", {})
    tts_smoke = smoke.get("tts", {})
    asr_rknn_smoke = smoke.get("asr_rknn", {})
    board_capabilities = evidence.get("board_capabilities", {})
    rknpu_load = evidence.get("rknpu_load", {})
    rknn_perf = evidence.get("rknn_perf", {})
    rknn_perf_summary = rknn_perf.get("summary", {})
    rknn_memory = evidence.get("rknn_memory", {})
    rknn_runtime_log = evidence.get("rknn_runtime_log", {})
    audio_asset = evidence_assets.get("audio")
    asr_heatmap_asset = evidence_assets.get("asr_rknpu_load_heatmap")
    tts_heatmap_asset = evidence_assets.get("tts_profile_heatmap")
    plan_summary = payload.get("plan", {})
    rknn_profile_label = format_rknn_profile_source(observed.get("rknn_profile_source"))
    rknn_operator_count = observed.get("rknn_operator_count") or 0
    rknn_run_ms = observed.get("rknn_run_duration_ms")
    if rknn_run_ms is None:
        rknn_run_ms = observed.get("rknn_total_time_ms")

    requirement_total = sum(requirement_summary.values())
    asset_count = sum(1 for path in evidence_assets.values() if path)
    unittest_issue_count = 0
    if unit_tests:
        unittest_issue_count = unit_tests["failed"] + unit_tests["errors"] + unit_tests["unexpected_successes"]
    verdict_class = summary.get("verdict", "unknown")

    cards = "".join(
        [
            render_card("综合判定", format_status(summary["verdict"]), summary["message"], "linear-gradient(135deg,#d66d4b,#9f2f1f)"),
            render_card("单元测试", f"{unit_tests['passed']}/{unit_tests['total']}" if unit_tests else "未执行", "通过 / 总数", "linear-gradient(135deg,#5b8c5a,#2f5d50)"),
            render_card("TTS 单句时延", format_number(observed["tts_elapsed_ms"], digits=0, suffix=" ms"), f"后端 {observed['tts_backend']}", "linear-gradient(135deg,#d28d49,#9c5a12)"),
            render_card("ASR RKNN 时延", format_number(observed["asr_rknn_elapsed_ms"], digits=0, suffix=" ms"), f"模式 {observed['asr_mode']}", "linear-gradient(135deg,#618fbf,#2f5e8a)"),
            render_card("RKNN Profiler", rknn_profile_label, f"{rknn_operator_count} 条层级记录", "linear-gradient(135deg,#8b7ab8,#51407c)"),
            render_card("RKNN 单次运行", format_number(rknn_run_ms, digits=3, suffix=" ms"), "PERF_RUN 优先，否则回退总算子耗时", "linear-gradient(135deg,#547e8e,#274754)"),
            render_card("层级峰值 MacUsage", format_number(observed["rknn_peak_mac_usage_percent"], digits=2, suffix="%"), "仅统计官方 profiler 输出", "linear-gradient(135deg,#c68163,#7f3f2d)"),
            render_card("NPU 峰值负载", format_number(observed["npu_peak_percent"], digits=0, suffix=" %"), "来自 rknpu/load / TTS sampling", "linear-gradient(135deg,#b65f6f,#7e3240)"),
            render_card("RKNN 内存总量", format_number(observed["rknn_total_memory_mib"], digits=2, suffix=" MiB"), "weight + internal tensor", "linear-gradient(135deg,#6d8f79,#325944)"),
            render_card("模型总量", format_number(observed["models_total_size_mib"], digits=1, suffix=" MiB"), "当前运行包 models 目录", "linear-gradient(135deg,#7d7c98,#514f73)"),
        ]
    )

    requirement_rows = "".join(
        f"<tr class=\"status-row status-{html.escape(item['status'])}\"><td>{html.escape(item['section'])}</td><td>{html.escape(item['subsection'])}</td><td>{html.escape(item['requirement'])}</td><td>{render_status_pill(item['status'])}</td><td>{html.escape(item['observed'])}</td><td>{html.escape(item['rationale'])}</td></tr>"
        for item in payload["requirements"]["items"]
    )

    unittest_rows = ""
    if unit_tests:
        rows: list[str] = []
        for case in unit_tests["cases"]:
            case_status = "pass" if case["status"] == "passed" else "fail" if case["status"] in {"failed", "error", "unexpected-success"} else "partial" if case["status"] == "skipped" else "unknown"
            details_text = case["details"].strip()
            details_markup = (
                '<span class="muted-inline">无附加输出</span>'
                if not details_text
                else f'<details class="case-details"><summary>展开日志</summary><pre>{html.escape(details_text)}</pre></details>'
            )
            rows.append(
                f'<tr class="status-row status-{case_status}"><td>{html.escape(case["test_id"])}</td><td>{render_status_pill(case_status)}</td><td>{html.escape(format_status(case["status"]))}</td><td>{case["duration_s"]:.3f}s</td><td>{details_markup}</td></tr>'
            )
        unittest_rows = "".join(rows)

    plan_rows = ""
    for category, count in sorted(plan_summary.get("category_counts", {}).items()):
        plan_rows += f"<tr><td>{html.escape(category)}</td><td>{count}</td></tr>"

    asset_links = "".join(
        f'<li><a href="{html.escape(path)}">{html.escape(name)}</a></li>'
        for name, path in evidence_assets.items()
        if path
    )

    navigation_links = "".join(
        f'<a class="nav-chip" href="#{section_id}">{html.escape(label)}</a>'
        for section_id, label in [
            ("overview", "执行概览"),
            ("profiling", "RKNN Profiling"),
            ("requirements", "指标矩阵"),
            ("evidence", "证据媒体"),
            ("tests", "单元测试"),
            ("appendix", "计划与产物"),
        ]
    )

    status_tiles = "".join(
        [
            f'<div class="status-tile pass"><span>通过</span><strong>{requirement_summary["pass"]}</strong></div>',
            f'<div class="status-tile fail"><span>未通过</span><strong>{requirement_summary["fail"]}</strong></div>',
            f'<div class="status-tile partial"><span>部分满足</span><strong>{requirement_summary["partial"]}</strong></div>',
            f'<div class="status-tile unknown"><span>证据不足</span><strong>{requirement_summary["unknown"]}</strong></div>',
        ]
    )

    runtime_facts = "".join(
        [
            render_fact_item("TTS 后端", runtime.get("tts_backend") or "n/a", "根据 run_tts.sh 推断"),
            render_fact_item("ASR 模式", runtime.get("asr_mode") or "n/a", "根据 run_asr.sh 推断"),
            render_fact_item("RKNN Runtime", board_capabilities.get("rknn_runtime_version") or "n/a", "来自板端能力快照"),
            render_fact_item("板端内存", board_capabilities.get("memory_total") or "n/a", "来自 board_profile_capabilities.txt"),
            render_fact_item("离线运行包", "就绪" if observed.get("offline_ready") else "待确认", "要求 bin 与 models 同时存在"),
            render_fact_item("证据文件", asset_count, "已复制到本次报告 assets"),
        ]
    )

    model_facts = "".join(
        [
            render_fact_item("TTS 模型", runtime.get("tts_model_name") or "n/a", format_number(runtime.get("tts_model_size_mib"), digits=2, suffix=" MiB")),
            render_fact_item("ASR CPU 模型", runtime.get("asr_cpu_model_name") or "n/a", format_number(runtime.get("asr_cpu_model_size_mib"), digits=2, suffix=" MiB")),
            render_fact_item("ASR RKNN 模型", runtime.get("asr_rknn_model_name") or "n/a", format_number(runtime.get("asr_rknn_model_size_mib"), digits=2, suffix=" MiB")),
            render_fact_item("模型总量", format_number(runtime.get("models_total_size_mib"), digits=2, suffix=" MiB"), "当前 runtime/models 目录"),
        ]
    )

    smoke_briefs = "".join(
        panel
        for panel in [
            (
                '<article class="story-card">'
                '<div class="story-label">TTS 冒烟</div>'
                f'<div class="story-value">{html.escape(format_number(tts_smoke.get("elapsed_seconds"), digits=3, suffix=" s"))}</div>'
                f'<div class="story-meta">音频 {html.escape(format_number(tts_smoke.get("audio_duration_seconds"), digits=3, suffix=" s"))} · RTF {html.escape(format_number(tts_smoke.get("rtf"), digits=3))}</div>'
                f'<div class="story-body">{html.escape(tts_smoke.get("text") or "未记录 TTS 文本")}</div>'
                '</article>'
            ),
            (
                '<article class="story-card">'
                '<div class="story-label">ASR RKNN 冒烟</div>'
                f'<div class="story-value">{html.escape(format_number(asr_rknn_smoke.get("elapsed_seconds"), digits=3, suffix=" s"))}</div>'
                f'<div class="story-meta">RTF {html.escape(format_number(asr_rknn_smoke.get("rtf"), digits=3))} · NPU 峰值 {html.escape(format_number(rknpu_load.get("peak_percent"), digits=0, suffix=" %"))}</div>'
                f'<div class="story-body">{html.escape(asr_rknn_smoke.get("result", {}).get("text") or "未记录 ASR 转写")}</div>'
                '</article>'
            ),
        ]
    )

    rknn_summary_boxes = "".join(
        box
        for box in [
            f'<div class="summary-box">Profile Source<strong>{html.escape(rknn_profile_label)}</strong></div>',
            f'<div class="summary-box">层级记录<strong>{rknn_operator_count}</strong></div>',
            f'<div class="summary-box">总算子耗时<strong>{html.escape(format_number(rknn_perf_summary.get("total_operator_elapsed_time_us"), digits=0, suffix=" us"))}</strong></div>',
            f'<div class="summary-box">峰值 MacUsage<strong>{html.escape(format_number(rknn_perf_summary.get("peak_mac_usage_percent"), digits=2, suffix="%"))}</strong></div>',
            f'<div class="summary-box">热点算子<strong>{html.escape(rknn_perf_summary.get("hottest_op_type") or "n/a")}</strong></div>',
            f'<div class="summary-box">层级总内存<strong>{html.escape(format_number(rknn_memory.get("total_memory_mib"), digits=2, suffix=" MiB"))}</strong></div>',
            f'<div class="summary-box">PERF_RUN<strong>{html.escape(format_number(observed.get("rknn_run_duration_ms"), digits=3, suffix=" ms"))}</strong></div>',
            f'<div class="summary-box">层日志行数<strong>{rknn_runtime_log.get("layer_line_count", 0)}</strong></div>',
        ]
    )

    top_layers = sorted(
        rknn_perf.get("operators", []),
        key=lambda item: item.get("time_us") or 0.0,
        reverse=True,
    )[:12]
    rknn_layer_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(layer.get('id', '')))}</td>"
        f"<td>{html.escape(layer.get('op_type', ''))}</td>"
        f"<td>{html.escape(layer.get('target', ''))}</td>"
        f"<td>{html.escape(format_number(layer.get('time_us'), digits=0, suffix=' us'))}</td>"
        f"<td>{html.escape(format_number(layer.get('npu_cycles'), digits=0))}</td>"
        f"<td>{html.escape(format_number(layer.get('mac_usage_percent'), digits=2, suffix='%'))}</td>"
        f"<td>{html.escape(layer.get('workload', ''))}</td>"
        f"<td>{html.escape(layer.get('full_name') or layer.get('output_shape', ''))}</td>"
        "</tr>"
        for layer in top_layers
    )
    rknn_ranking_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item.get('op_type', ''))}</td>"
        f"<td>{html.escape(str(item.get('call_number', '')))}</td>"
        f"<td>{html.escape(format_number(item.get('total_time_us'), digits=0, suffix=' us'))}</td>"
        f"<td>{html.escape(format_number(item.get('time_ratio_percent'), digits=2, suffix='%'))}</td>"
        "</tr>"
        for item in rknn_perf.get("ranking", [])[:8]
    )
    runtime_log_preview = "\n".join(rknn_runtime_log.get("sample_lines", []))
    rknn_profile_block = (
        '<article class="panel panel-highlight">'
        '<div class="section-head compact">'
        '<div>'
        '<div class="section-kicker">Profiler</div>'
        '<h2>RKNN 官方 Profiling</h2>'
        '<div class="section-note">优先展示 Toolkit2 eval_perf() / RKNN_QUERY_PERF_DETAIL / PERF_RUN / MEM_SIZE 的结构化证据；RKNN_LOG_LEVEL=4 作为运行时层日志补充，rknpu/load 热力图仅用于时序负载观察。</div>'
        '</div>'
        f'<div class="section-badge">{html.escape(rknn_profile_label)}</div>'
        '</div>'
        f'<div class="summary-grid">{rknn_summary_boxes}</div>'
        + (
            '<div class="subsection-title">热点层明细</div>'
            '<div class="table-shell">'
            '<table>'
            '<tr><th>ID</th><th>OpType</th><th>Target</th><th>Time</th><th>NPU Cycles</th><th>MacUsage</th><th>WorkLoad</th><th>Layer</th></tr>'
            f'{rknn_layer_rows}'
            '</table>'
            '</div>'
            if rknn_layer_rows
            else '<div class="empty-state" style="margin-top:18px;">未发现可解析的 eval_perf / PERF_DETAIL 层级明细；报告会继续使用运行时层日志和 rknpu/load 采样。</div>'
        )
        + (
            '<div class="subsection-title">算子类型排行</div>'
            '<div class="table-shell">'
            '<table>'
            '<tr><th>OpType</th><th>Call Number</th><th>Total Time</th><th>Ratio</th></tr>'
            f'{rknn_ranking_rows}'
            '</table>'
            '</div>'
            if rknn_ranking_rows
            else ''
        )
        + (
            '<div class="subsection-title">RKNN_LOG_LEVEL=4 预览</div>'
            f'<div class="log-preview"><pre>{html.escape(runtime_log_preview)}</pre></div>'
            if runtime_log_preview
            else ''
        )
        + '</article>'
    )

    audio_block = '<div class="empty-state">未发现可嵌入的音频证据。</div>'
    if audio_asset:
        audio_block = (
            '<div class="media-card">'
                        '<div class="media-title">音频预览</div>'
            f'<audio controls preload="metadata" src="{html.escape(audio_asset)}"></audio>'
                        f'<div class="media-meta">时长 {html.escape(format_number(evidence["audio"].get("duration_s"), digits=3, suffix=" s"))}，体积 {html.escape(format_number(evidence["audio"].get("size_mib"), digits=3, suffix=" MiB"))}</div>'
            '</div>'
        )

    heatmap_blocks: list[str] = []
    if asr_heatmap_asset:
        heatmap_blocks.append(
            '<div class="media-card">'
                        '<div class="media-title">ASR RKNN NPU Load Heatmap</div>'
            f'<img alt="ASR RKNN NPU load heatmap" src="{html.escape(asr_heatmap_asset)}" />'
                        '<div class="media-meta">用于观察三个 NPU core 的时序忙闲变化，不替代官方层级 profiler。</div>'
            '</div>'
        )
    if tts_heatmap_asset:
        heatmap_blocks.append(
            '<div class="media-card">'
                        '<div class="media-title">TTS Process Sampling Heatmap</div>'
            f'<img alt="TTS profile heatmap" src="{html.escape(tts_heatmap_asset)}" />'
                        '<div class="media-meta">由 RSS / CPU / NPU 采样生成，用于补充当前 TTS 基线观察。</div>'
            '</div>'
        )
    if not heatmap_blocks:
        heatmap_blocks.append('<div class="empty-state">未发现可生成时序采样热力图的 profile 文件。</div>')

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RKVoice 综合测试报告</title>
  <style>
    :root {{
            --bg-top: #efe5da;
            --bg-bottom: #f7f4ef;
            --ink: #251d19;
            --muted: #6d645d;
            --panel: rgba(255,255,255,0.84);
            --panel-strong: rgba(255,250,244,0.96);
            --line: rgba(91, 75, 64, 0.16);
            --shadow: 0 24px 60px rgba(73, 51, 37, 0.08);
      --pass: #3d7a4f;
      --fail: #a93a2e;
      --partial: #b87722;
      --unknown: #7c7f84;
            --teal: #2c6a73;
            --ember: #a34f2d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
            font-family: "Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
                radial-gradient(circle at top left, rgba(198,109,56,0.18), transparent 24%),
                radial-gradient(circle at top right, rgba(51,122,133,0.16), transparent 26%),
                repeating-linear-gradient(90deg, rgba(255,255,255,0.18) 0, rgba(255,255,255,0.18) 1px, transparent 1px, transparent 72px),
        linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
    }}
        h1, h2, .metric-value, .verdict-value {{ font-family: "Bahnschrift", "Segoe UI Variable Display", "Microsoft YaHei", sans-serif; letter-spacing: 0.01em; }}
        a {{ color: var(--ember); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .page-shell {{ max-width: 1480px; margin: 0 auto; padding: 26px; }}
    .hero {{
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.85fr);
            gap: 24px;
            padding: 30px 32px;
            border-radius: 30px;
            background: linear-gradient(135deg, rgba(255,248,239,0.96), rgba(255,255,255,0.86));
      border: 1px solid var(--line);
            box-shadow: var(--shadow);
    }}
        .eyebrow {{ letter-spacing: 0.18em; text-transform: uppercase; font-size: 12px; color: #8a6b5d; }}
    h1 {{ margin: 10px 0 6px; font-size: clamp(32px, 5vw, 52px); line-height: 1.02; }}
        .hero-copy p {{ margin: 0; max-width: 820px; color: var(--muted); font-size: 16px; line-height: 1.75; }}
    .hero-meta {{ margin-top: 18px; display: flex; flex-wrap: wrap; gap: 12px; color: var(--muted); font-size: 13px; }}
        .hero-chip {{ padding: 9px 13px; border-radius: 999px; background: rgba(255,255,255,0.72); border: 1px solid var(--line); }}
        .hero-stack {{ display: grid; gap: 16px; }}
        .verdict-panel {{ padding: 20px 22px; border-radius: 24px; color: #fffdf8; background: linear-gradient(135deg, #73452f, #2f5f69); box-shadow: inset 0 1px 0 rgba(255,255,255,0.14); }}
        .verdict-panel.fail {{ background: linear-gradient(135deg, #aa5430, #6b2d25); }}
        .verdict-panel.partial {{ background: linear-gradient(135deg, #b67a26, #6b5330); }}
        .verdict-panel.pass {{ background: linear-gradient(135deg, #2f7354, #23514b); }}
        .verdict-panel.unknown {{ background: linear-gradient(135deg, #5e6972, #46515f); }}
        .verdict-label {{ font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; opacity: 0.84; }}
        .verdict-value {{ margin-top: 10px; font-size: 38px; font-weight: 700; line-height: 1; }}
        .verdict-panel p {{ margin: 10px 0 0; color: rgba(255,253,248,0.86); line-height: 1.7; }}
        .hero-mini-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
        .hero-mini {{ padding: 14px 16px; border-radius: 18px; background: rgba(255,255,255,0.66); border: 1px solid var(--line); }}
        .hero-mini span {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
        .hero-mini strong {{ display: block; margin-top: 6px; font-size: 24px; font-weight: 700; }}
        .quick-nav {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 18px 0 0; padding: 0; }}
        .nav-chip {{ display: inline-flex; align-items: center; padding: 10px 14px; border-radius: 999px; border: 1px solid var(--line); background: rgba(255,255,255,0.62); color: #5a4034; font-size: 13px; font-weight: 600; }}
        .dashboard {{ display: grid; grid-template-columns: minmax(0, 1.48fr) minmax(280px, 0.72fr); gap: 24px; margin-top: 24px; }}
        .main-column {{ display: grid; gap: 22px; }}
        .side-column {{ display: grid; gap: 18px; }}
        .sticky-stack {{ position: sticky; top: 18px; display: grid; gap: 18px; }}
        .report-section {{ display: grid; gap: 16px; }}
        .section-head {{ display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 2px; }}
        .section-head.compact {{ margin-bottom: 14px; }}
        .section-kicker {{ letter-spacing: 0.14em; text-transform: uppercase; font-size: 12px; color: #8b6756; }}
        .section-head h2 {{ margin: 8px 0 0; font-size: 28px; line-height: 1.08; }}
        .section-note {{ margin-top: 8px; color: var(--muted); line-height: 1.7; font-size: 14px; max-width: 900px; }}
        .section-badge {{ padding: 10px 14px; border-radius: 999px; background: rgba(44,106,115,0.1); color: var(--teal); border: 1px solid rgba(44,106,115,0.18); font-size: 13px; font-weight: 700; white-space: nowrap; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
        .metric-card {{ position: relative; overflow: hidden; padding: 18px; border-radius: 22px; background: var(--panel); border: 1px solid var(--line); min-height: 158px; box-shadow: 0 14px 34px rgba(79,57,46,0.05); }}
        .metric-accent {{ position: absolute; inset: 0 auto auto 0; width: 100%; height: 7px; }}
        .metric-title {{ color: #8f6f5f; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 10px; }}
    .metric-value {{ margin-top: 12px; font-size: 34px; font-weight: 700; }}
    .metric-subtitle {{ margin-top: 8px; color: var(--muted); font-size: 13px; line-height: 1.6; }}
        .panel {{ padding: 22px; border-radius: 24px; background: var(--panel); border: 1px solid var(--line); box-shadow: 0 16px 40px rgba(79,57,46,0.06); }}
        .panel-highlight {{ background: linear-gradient(180deg, rgba(255,251,246,0.98), rgba(255,255,255,0.86)); }}
        .rail-panel {{ padding: 18px; border-radius: 22px; background: rgba(255,255,255,0.76); border: 1px solid var(--line); box-shadow: 0 14px 34px rgba(79,57,46,0.05); }}
        .rail-title {{ font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; color: #8b6756; margin-bottom: 12px; }}
        .panel-title {{ font-size: 18px; font-weight: 700; margin-bottom: 12px; }}
    .panel-subtitle {{ color: var(--muted); font-size: 13px; line-height: 1.6; margin-bottom: 18px; }}
        .status-strip {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
        .status-tile {{ padding: 14px 16px; border-radius: 18px; border: 1px solid var(--line); background: rgba(255,255,255,0.7); }}
        .status-tile span {{ display: block; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
        .status-tile strong {{ display: block; margin-top: 6px; font-size: 28px; }}
        .status-tile.pass {{ background: rgba(61,122,79,0.08); }}
        .status-tile.fail {{ background: rgba(169,58,46,0.08); }}
        .status-tile.partial {{ background: rgba(184,119,34,0.08); }}
        .status-tile.unknown {{ background: rgba(124,127,132,0.08); }}
        .fact-list {{ display: grid; gap: 12px; }}
        .fact-row {{ padding: 12px 14px; border-radius: 16px; background: rgba(255,255,255,0.62); border: 1px solid rgba(91,75,64,0.1); }}
        .fact-label {{ font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
        .fact-value {{ margin-top: 5px; font-size: 18px; font-weight: 700; line-height: 1.4; word-break: break-word; }}
        .fact-detail {{ margin-top: 4px; color: var(--muted); font-size: 12px; line-height: 1.6; }}
        .story-grid {{ display: grid; gap: 12px; }}
        .story-card {{ padding: 16px; border-radius: 18px; background: linear-gradient(135deg, rgba(255,248,241,0.9), rgba(255,255,255,0.76)); border: 1px solid var(--line); }}
        .story-label {{ font-size: 12px; letter-spacing: 0.1em; text-transform: uppercase; color: #8b6756; }}
        .story-value {{ margin-top: 8px; font-size: 28px; font-weight: 700; }}
        .story-meta {{ margin-top: 6px; font-size: 13px; color: var(--muted); }}
        .story-body {{ margin-top: 10px; line-height: 1.7; color: #4f433b; }}
        .subsection-title {{ margin: 18px 0 10px; font-size: 16px; font-weight: 700; }}
        .table-shell {{ overflow: auto; border-radius: 18px; border: 1px solid rgba(91,75,64,0.12); background: rgba(255,255,255,0.72); }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; min-width: 720px; }}
        th, td {{ border-top: 1px solid var(--line); text-align: left; vertical-align: top; padding: 12px 10px; }}
        th {{ position: sticky; top: 0; z-index: 1; color: #765f53; font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; background: rgba(255,250,244,0.98); backdrop-filter: blur(8px); }}
    tr:first-child th, tr:first-child td {{ border-top: none; }}
        .status-row.status-fail td {{ background: rgba(169,58,46,0.03); }}
        .status-row.status-partial td {{ background: rgba(184,119,34,0.03); }}
        .status-row.status-pass td {{ background: rgba(61,122,79,0.025); }}
    .pill {{ display: inline-flex; padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
    .pill.pass {{ background: rgba(61,122,79,0.12); color: var(--pass); }}
    .pill.fail {{ background: rgba(169,58,46,0.12); color: var(--fail); }}
    .pill.partial {{ background: rgba(184,119,34,0.12); color: var(--partial); }}
    .pill.unknown {{ background: rgba(124,127,132,0.12); color: var(--unknown); }}
        .muted-inline {{ color: var(--muted); font-size: 13px; }}
        .case-details summary {{ cursor: pointer; font-weight: 700; color: var(--teal); }}
        .case-details pre {{ margin-top: 10px; }}
        .media-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
        .media-card {{ padding: 18px; border-radius: 20px; background: rgba(255,255,255,0.72); border: 1px solid var(--line); }}
        .media-title {{ font-size: 18px; font-weight: 700; margin-bottom: 12px; }}
    .media-card img {{ width: 100%; display: block; border-radius: 16px; border: 1px solid rgba(112, 84, 69, 0.14); background: #fff8f2; }}
    audio {{ width: 100%; margin-top: 8px; }}
    .media-meta {{ margin-top: 10px; color: var(--muted); font-size: 13px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
        .summary-box {{ padding: 14px 16px; border-radius: 18px; background: rgba(255,255,255,0.72); border: 1px solid var(--line); }}
    .summary-box strong {{ display: block; font-size: 24px; margin-top: 6px; }}
    .empty-state {{ padding: 22px; border-radius: 20px; background: rgba(255,255,255,0.62); color: var(--muted); border: 1px dashed rgba(112,84,69,0.24); }}
    .log-preview {{ margin-top: 8px; padding: 14px 16px; border-radius: 18px; background: rgba(255,255,255,0.72); border: 1px solid var(--line); }}
        ul.asset-list {{ margin: 0; padding-left: 18px; color: var(--muted); display: grid; gap: 8px; }}
    pre {{ margin: 0; white-space: pre-wrap; font-family: Consolas, "SFMono-Regular", monospace; font-size: 11px; color: #5a4840; }}
    @media (max-width: 980px) {{
            .hero {{ grid-template-columns: 1fr; }}
            .dashboard {{ grid-template-columns: 1fr; }}
            .sticky-stack {{ position: static; }}
            .status-strip {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 640px) {{
            .page-shell {{ padding: 18px; }}
            .hero {{ padding: 22px; border-radius: 22px; }}
            .hero-mini-grid {{ grid-template-columns: 1fr; }}
            .status-strip {{ grid-template-columns: 1fr; }}
      .summary-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
    <div class="page-shell">
    <section class="hero">
            <div class="hero-copy">
                <div class="eyebrow">RKVoice Integrated Report</div>
                <h1>RK3588 离线 ASR + TTS 综合测试报告</h1>
                <p>这份报告将当前仓库的 unittest 结果、板端 smoke / profile 证据、测试计划模板和项目指标文档汇总到一个更适合交付评审的静态 HTML Dashboard 中。官方 RKNN profiler 位于视觉中心，媒体证据与采样热力图退到补充位置，方便先看结论，再顺着证据往下钻取。</p>
                <div class="hero-meta">
                    <div class="hero-chip">生成时间 {html.escape(payload['generated_at'])}</div>
                    <div class="hero-chip">工作区 {html.escape(payload['workspace_root'])}</div>
                    <div class="hero-chip">运行包 {html.escape(runtime['runtime_dir'])}</div>
                    <div class="hero-chip">测试计划 {html.escape(plan_summary.get('path', '未配置'))}</div>
                </div>
                <div class="quick-nav">
                    {navigation_links}
                </div>
            </div>
            <div class="hero-stack">
                <div class="verdict-panel {verdict_class}">
                    <div class="verdict-label">综合判定</div>
                    <div class="verdict-value">{html.escape(format_status(summary['verdict']))}</div>
                    <p>{html.escape(summary['message'])}</p>
                </div>
                <div class="hero-mini-grid">
                    <div class="hero-mini"><span>指标项</span><strong>{requirement_total}</strong></div>
                    <div class="hero-mini"><span>证据文件</span><strong>{asset_count}</strong></div>
                    <div class="hero-mini"><span>Profiler Layers</span><strong>{rknn_operator_count}</strong></div>
                    <div class="hero-mini"><span>单测问题</span><strong>{unittest_issue_count}</strong></div>
                </div>
      </div>
    </section>

        <section class="dashboard">
            <main class="main-column">
                <section class="report-section" id="overview">
                    <div class="section-head">
                        <div>
                            <div class="section-kicker">Overview</div>
                            <h2>执行概览</h2>
                            <div class="section-note">先看关键数字，再看矩阵和 profiler。导出页不再把所有表格堆在同一层，而是把当前可交付结论、缺口和板端状态分开呈现。</div>
                        </div>
                        <div class="section-badge">Report Navigation Ready</div>
                    </div>
                    <div class="metrics">{cards}</div>
                    <article class="panel">
                        <div class="panel-title">指标总览</div>
                        <div class="panel-subtitle">报告按 docs/requirements/项目指标.md 逐条生成状态。未通过说明当前证据已证明不达标，证据不足说明需要追加专门测试。</div>
                        <div class="status-strip">{status_tiles}</div>
                    </article>
                </section>

                <section class="report-section" id="profiling">
                    {rknn_profile_block}
                </section>

                <section class="report-section" id="requirements">
                    <div class="section-head">
                        <div>
                            <div class="section-kicker">Requirements</div>
                            <h2>指标矩阵</h2>
                            <div class="section-note">这里保留完整矩阵，方便审阅具体条目。表头固定，长表格横向可滚动，移动端也不会把内容压坏。</div>
                        </div>
                    </div>
                    <article class="panel">
                        <div class="table-shell">
                            <table>
                                <tr><th>章节</th><th>子项</th><th>要求</th><th>状态</th><th>观测</th><th>判定依据</th></tr>
                                {requirement_rows}
                            </table>
                        </div>
                    </article>
                </section>

                <section class="report-section" id="evidence">
                    <div class="section-head">
                        <div>
                            <div class="section-kicker">Evidence</div>
                            <h2>证据媒体与时序采样</h2>
                            <div class="section-note">音频预览和热力图用于补充说明运行期表现；真正的层级性能诊断请以上方 RKNN 官方 profiling 面板为准。</div>
                        </div>
                    </div>
                    <article class="panel">
                        <div class="media-grid">
                            {audio_block}
                            {''.join(heatmap_blocks)}
                        </div>
                    </article>
                </section>

                <section class="report-section" id="tests">
                    <div class="section-head">
                        <div>
                            <div class="section-kicker">Tests</div>
                            <h2>单元测试明细</h2>
                            <div class="section-note">统一复用 tests 目录 discovery 入口。测试详情默认折叠，避免大段日志把主界面直接淹没。</div>
                        </div>
                    </div>
                    <article class="panel">
                        <div class="table-shell">
                            <table>
                                <tr><th>测试</th><th>状态</th><th>类型</th><th>耗时</th><th>详情</th></tr>
                                {unittest_rows or '<tr><td colspan="5">本次未执行 unittest。</td></tr>'}
                            </table>
                        </div>
                    </article>
                </section>
            </main>

            <aside class="side-column" id="appendix">
                <div class="sticky-stack">
                    <article class="rail-panel">
                        <div class="rail-title">报告导航</div>
                        <div class="quick-nav">{navigation_links}</div>
                    </article>

                    <article class="rail-panel">
                        <div class="rail-title">运行画像</div>
                        <div class="fact-list">{runtime_facts}</div>
                    </article>

                    <article class="rail-panel">
                        <div class="rail-title">模型基线</div>
                        <div class="fact-list">{model_facts}</div>
                    </article>

                    <article class="rail-panel">
                        <div class="rail-title">冒烟摘要</div>
                        <div class="story-grid">{smoke_briefs}</div>
                    </article>

                    <article class="rail-panel">
                        <div class="rail-title">计划与基线</div>
                        <div class="panel-subtitle">这里展示测试计划模板中的覆盖面，帮助区分“有计划但未执行”和“根本没有验证入口”。</div>
                        <div class="table-shell">
                            <table>
                                <tr><th>类别</th><th>用例数</th></tr>
                                {plan_rows or '<tr><td colspan="2">未找到测试计划文件。</td></tr>'}
                            </table>
                        </div>
                    </article>

                    <article class="rail-panel">
                        <div class="rail-title">原始产物</div>
                        <ul class="asset-list">{asset_links or '<li>未复制任何证据文件。</li>'}</ul>
                    </article>
                </div>
            </aside>
        </section>
  </div>
</body>
</html>
"""


def classify_unittest_case_status(status: str) -> str:
    if status == "passed":
        return "pass"
    if status in {"failed", "error", "unexpected-success"}:
        return "fail"
    if status == "skipped":
        return "partial"
    return "unknown"


def build_html_report_context(payload: dict[str, Any], *, static_assets: dict[str, str]) -> dict[str, Any]:
    summary = payload["summary"]
    runtime = payload["runtime"]
    observed = payload["observed"]
    evidence = payload["evidence"]
    unit_tests = payload.get("unit_tests")
    requirement_summary = payload["requirements"]["summary"]
    requirement_items = payload["requirements"]["items"]
    evidence_assets = evidence["assets"]
    smoke = evidence.get("smoke", {})
    tts_smoke = smoke.get("tts", {})
    asr_rknn_smoke = smoke.get("asr_rknn", {})
    board_capabilities = evidence.get("board_capabilities", {})
    rknpu_load = evidence.get("rknpu_load", {})
    rknn_perf = evidence.get("rknn_perf", {})
    rknn_perf_summary = rknn_perf.get("summary", {})
    rknn_memory = evidence.get("rknn_memory", {})
    rknn_runtime_log = evidence.get("rknn_runtime_log", {})
    audio_asset = evidence_assets.get("audio")
    asr_heatmap_asset = evidence_assets.get("asr_rknpu_load_heatmap")
    tts_heatmap_asset = evidence_assets.get("tts_profile_heatmap")
    plan_summary = payload.get("plan", {})
    rknn_profile_label = format_rknn_profile_source(observed.get("rknn_profile_source"))
    rknn_operator_count = observed.get("rknn_operator_count") or 0
    rknn_run_ms = observed.get("rknn_run_duration_ms")
    if rknn_run_ms is None:
        rknn_run_ms = observed.get("rknn_total_time_ms")

    requirement_total = sum(requirement_summary.values())
    asset_count = sum(1 for path in evidence_assets.values() if path)
    unittest_issue_count = 0
    if unit_tests:
        unittest_issue_count = unit_tests["failed"] + unit_tests["errors"] + unit_tests["unexpected_successes"]

    cards = [
        {
            "title": "综合判定",
            "value": format_status(summary["verdict"]),
            "subtitle": summary["message"],
            "accent": "linear-gradient(135deg,#d66d4b,#9f2f1f)",
        },
        {
            "title": "单元测试",
            "value": f"{unit_tests['passed']}/{unit_tests['total']}" if unit_tests else "未执行",
            "subtitle": "通过 / 总数",
            "accent": "linear-gradient(135deg,#5b8c5a,#2f5d50)",
        },
        {
            "title": "TTS 单句时延",
            "value": format_number(observed.get("tts_elapsed_ms"), digits=0, suffix=" ms"),
            "subtitle": f"后端 {observed.get('tts_backend')}",
            "accent": "linear-gradient(135deg,#d28d49,#9c5a12)",
        },
        {
            "title": "ASR RKNN 时延",
            "value": format_number(observed.get("asr_rknn_elapsed_ms"), digits=0, suffix=" ms"),
            "subtitle": f"模式 {observed.get('asr_mode')}",
            "accent": "linear-gradient(135deg,#618fbf,#2f5e8a)",
        },
        {
            "title": "RKNN Profiler",
            "value": rknn_profile_label,
            "subtitle": f"{rknn_operator_count} 条层级记录",
            "accent": "linear-gradient(135deg,#8b7ab8,#51407c)",
        },
        {
            "title": "RKNN 单次运行",
            "value": format_number(rknn_run_ms, digits=3, suffix=" ms"),
            "subtitle": "PERF_RUN 优先，否则回退总算子耗时",
            "accent": "linear-gradient(135deg,#547e8e,#274754)",
        },
        {
            "title": "层级峰值 MacUsage",
            "value": format_number(observed.get("rknn_peak_mac_usage_percent"), digits=2, suffix="%"),
            "subtitle": "仅统计官方 profiler 输出",
            "accent": "linear-gradient(135deg,#c68163,#7f3f2d)",
        },
        {
            "title": "NPU 峰值负载",
            "value": format_number(observed.get("npu_peak_percent"), digits=0, suffix=" %"),
            "subtitle": "来自 rknpu/load / TTS sampling",
            "accent": "linear-gradient(135deg,#b65f6f,#7e3240)",
        },
        {
            "title": "RKNN 内存总量",
            "value": format_number(observed.get("rknn_total_memory_mib"), digits=2, suffix=" MiB"),
            "subtitle": "weight + internal tensor",
            "accent": "linear-gradient(135deg,#6d8f79,#325944)",
        },
        {
            "title": "模型总量",
            "value": format_number(observed.get("models_total_size_mib"), digits=1, suffix=" MiB"),
            "subtitle": "当前运行包 models 目录",
            "accent": "linear-gradient(135deg,#7d7c98,#514f73)",
        },
    ]

    navigation_links = [
        {"section_id": "overview", "label": "执行概览"},
        {"section_id": "profiling", "label": "RKNN Profiling"},
        {"section_id": "requirements", "label": "指标矩阵"},
        {"section_id": "evidence", "label": "证据媒体"},
        {"section_id": "tests", "label": "单元测试"},
        {"section_id": "appendix", "label": "计划与产物"},
    ]

    hero_chips = [
        {"label": "生成时间", "value": payload["generated_at"]},
        {"label": "工作区", "value": payload["workspace_root"]},
        {"label": "运行包", "value": runtime.get("runtime_dir") or "n/a"},
        {"label": "测试计划", "value": plan_summary.get("path", "未配置")},
    ]
    hero_stats = [
        {"label": "指标项", "value": requirement_total},
        {"label": "证据文件", "value": asset_count},
        {"label": "Profiler Layers", "value": rknn_operator_count},
        {"label": "单测问题", "value": unittest_issue_count},
    ]

    status_tiles = [
        {"label": "通过", "count": requirement_summary["pass"], "status": "pass"},
        {"label": "未通过", "count": requirement_summary["fail"], "status": "fail"},
        {"label": "部分满足", "count": requirement_summary["partial"], "status": "partial"},
        {"label": "证据不足", "count": requirement_summary["unknown"], "status": "unknown"},
    ]

    requirement_rows = [
        {
            **item,
            "status_class": item["status"],
            "status_label": format_status(item["status"]),
        }
        for item in requirement_items
    ]

    unit_test_rows: list[dict[str, Any]] = []
    if unit_tests:
        for case in unit_tests["cases"]:
            case_status = classify_unittest_case_status(case["status"])
            unit_test_rows.append(
                {
                    **case,
                    "status_class": case_status,
                    "status_label": format_status(case_status),
                    "case_label": format_status(case["status"]),
                    "has_details": bool(case["details"].strip()),
                }
            )

    runtime_facts = [
        {"label": "TTS 后端", "value": runtime.get("tts_backend") or "n/a", "detail": "根据 run_tts.sh 推断"},
        {"label": "ASR 模式", "value": runtime.get("asr_mode") or "n/a", "detail": "根据 run_asr.sh 推断"},
        {"label": "RKNN Runtime", "value": board_capabilities.get("rknn_runtime_version") or "n/a", "detail": "来自板端能力快照"},
        {"label": "板端内存", "value": board_capabilities.get("memory_total") or "n/a", "detail": "来自 board_profile_capabilities.txt"},
        {"label": "离线运行包", "value": "就绪" if observed.get("offline_ready") else "待确认", "detail": "要求 bin 与 models 同时存在"},
        {"label": "证据文件", "value": asset_count, "detail": "已复制到本次报告 assets"},
    ]
    model_facts = [
        {
            "label": "TTS 模型",
            "value": runtime.get("tts_model_name") or "n/a",
            "detail": format_number(runtime.get("tts_model_size_mib"), digits=2, suffix=" MiB"),
        },
        {
            "label": "ASR CPU 模型",
            "value": runtime.get("asr_cpu_model_name") or "n/a",
            "detail": format_number(runtime.get("asr_cpu_model_size_mib"), digits=2, suffix=" MiB"),
        },
        {
            "label": "ASR RKNN 模型",
            "value": runtime.get("asr_rknn_model_name") or "n/a",
            "detail": format_number(runtime.get("asr_rknn_model_size_mib"), digits=2, suffix=" MiB"),
        },
        {
            "label": "模型总量",
            "value": format_number(runtime.get("models_total_size_mib"), digits=2, suffix=" MiB"),
            "detail": "当前 runtime/models 目录",
        },
    ]

    smoke_briefs = [
        {
            "label": "TTS 冒烟",
            "value": format_number(tts_smoke.get("elapsed_seconds"), digits=3, suffix=" s"),
            "meta": f"音频 {format_number(tts_smoke.get('audio_duration_seconds'), digits=3, suffix=' s')} · RTF {format_number(tts_smoke.get('rtf'), digits=3)}",
            "body": tts_smoke.get("text") or "未记录 TTS 文本",
        },
        {
            "label": "ASR RKNN 冒烟",
            "value": format_number(asr_rknn_smoke.get("elapsed_seconds"), digits=3, suffix=" s"),
            "meta": f"RTF {format_number(asr_rknn_smoke.get('rtf'), digits=3)} · NPU 峰值 {format_number(rknpu_load.get('peak_percent'), digits=0, suffix=' %')}",
            "body": asr_rknn_smoke.get("result", {}).get("text") or "未记录 ASR 转写",
        },
    ]

    rknn_summary_boxes = [
        {"label": "Profile Source", "value": rknn_profile_label},
        {"label": "层级记录", "value": rknn_operator_count},
        {"label": "总算子耗时", "value": format_number(rknn_perf_summary.get("total_operator_elapsed_time_us"), digits=0, suffix=" us")},
        {"label": "峰值 MacUsage", "value": format_number(rknn_perf_summary.get("peak_mac_usage_percent"), digits=2, suffix="%")},
        {"label": "热点算子", "value": rknn_perf_summary.get("hottest_op_type") or "n/a"},
        {"label": "层级总内存", "value": format_number(rknn_memory.get("total_memory_mib"), digits=2, suffix=" MiB")},
        {"label": "PERF_RUN", "value": format_number(observed.get("rknn_run_duration_ms"), digits=3, suffix=" ms")},
        {"label": "层日志行数", "value": rknn_runtime_log.get("layer_line_count", 0)},
    ]

    top_layers = [
        {
            "id": str(layer.get("id", "")),
            "op_type": layer.get("op_type", ""),
            "target": layer.get("target", ""),
            "time": format_number(layer.get("time_us"), digits=0, suffix=" us"),
            "npu_cycles": format_number(layer.get("npu_cycles"), digits=0),
            "mac_usage": format_number(layer.get("mac_usage_percent"), digits=2, suffix="%"),
            "workload": layer.get("workload", ""),
            "layer": layer.get("full_name") or layer.get("output_shape", ""),
        }
        for layer in sorted(
            rknn_perf.get("operators", []),
            key=lambda item: item.get("time_us") or 0.0,
            reverse=True,
        )[:12]
    ]
    rknn_ranking_rows = [
        {
            "op_type": item.get("op_type", ""),
            "call_number": str(item.get("call_number", "")),
            "total_time": format_number(item.get("total_time_us"), digits=0, suffix=" us"),
            "ratio": format_number(item.get("time_ratio_percent"), digits=2, suffix="%"),
        }
        for item in rknn_perf.get("ranking", [])[:8]
    ]

    media_cards: list[dict[str, Any]] = []
    if audio_asset:
        media_cards.append(
            {
                "kind": "audio",
                "title": "音频预览",
                "src": audio_asset,
                "alt": "",
                "meta": f"时长 {format_number(evidence['audio'].get('duration_s'), digits=3, suffix=' s')}，体积 {format_number(evidence['audio'].get('size_mib'), digits=3, suffix=' MiB')}",
            }
        )
    if asr_heatmap_asset:
        media_cards.append(
            {
                "kind": "image",
                "title": "ASR RKNN NPU Load Heatmap",
                "src": asr_heatmap_asset,
                "alt": "ASR RKNN NPU load heatmap",
                "meta": "用于观察三个 NPU core 的时序忙闲变化，不替代官方层级 profiler。",
            }
        )
    if tts_heatmap_asset:
        media_cards.append(
            {
                "kind": "image",
                "title": "TTS Process Sampling Heatmap",
                "src": tts_heatmap_asset,
                "alt": "TTS profile heatmap",
                "meta": "由 RSS / CPU / NPU 采样生成，用于补充当前 TTS 基线观察。",
            }
        )

    asset_links = [
        {"name": name, "path": path}
        for name, path in evidence_assets.items()
        if path
    ]
    plan_rows = [
        {"category": category, "count": count}
        for category, count in sorted(plan_summary.get("category_counts", {}).items())
    ]

    return {
        "summary": summary,
        "summary_label": format_status(summary["verdict"]),
        "runtime": runtime,
        "observed": observed,
        "unit_tests": unit_tests,
        "static_assets": static_assets,
        "verdict_class": summary.get("verdict", "unknown"),
        "hero_intro": "这份报告将当前仓库的 unittest 结果、板端 smoke / profile 证据、测试计划模板和项目指标文档汇总到一个更适合交付评审的静态 HTML Dashboard 中。官方 RKNN profiler 位于视觉中心，媒体证据与采样热力图退到补充位置，方便先看结论，再顺着证据往下钻取。",
        "navigation_links": navigation_links,
        "hero_chips": hero_chips,
        "hero_stats": hero_stats,
        "cards": cards,
        "status_tiles": status_tiles,
        "requirement_rows": requirement_rows,
        "unit_test_rows": unit_test_rows,
        "runtime_facts": runtime_facts,
        "model_facts": model_facts,
        "smoke_briefs": smoke_briefs,
        "rknn_summary_boxes": rknn_summary_boxes,
        "rknn_profile_label": rknn_profile_label,
        "top_layers": top_layers,
        "rknn_ranking_rows": rknn_ranking_rows,
        "runtime_log_preview": rknn_runtime_log.get("sample_lines", []),
        "media_cards": media_cards,
        "plan_rows": plan_rows,
        "asset_links": asset_links,
        "asset_count": asset_count,
        "requirement_total": requirement_total,
        "unittest_issue_count": unittest_issue_count,
        "requirements_path": payload["requirements"]["path"],
        "plan_summary": plan_summary,
    }


def render_html_report(payload: dict[str, Any], static_assets: dict[str, str] | None = None) -> str:
    environment = build_report_template_environment()
    template = environment.get_template(REPORT_TEMPLATE_NAME)
    context = build_html_report_context(payload, static_assets=static_assets or default_report_static_assets())
    return template.render(**context)


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
    static_assets = materialize_report_static_assets(assets_dir)

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
    write_text(html_path, render_html_report(payload, static_assets=static_assets))

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