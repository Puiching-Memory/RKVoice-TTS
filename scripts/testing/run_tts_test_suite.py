from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.delivery.prepare_paddlespeech_tts_armlinux import (  # noqa: E402
    CONFIG_LOCAL_DIR,
    DEFAULT_REMOTE_DIR,
    guess_source_ip,
    load_local_settings,
    open_ssh_client,
    resolve_int_option,
    resolve_required_text_option,
    resolve_text_option,
    sh_quote,
)


DEFAULT_LOCAL_PLAN = CONFIG_LOCAL_DIR / "tts_test_plan.json"
DEFAULT_EXAMPLE_PLAN = WORKSPACE_ROOT / "config" / "examples" / "tts_test_plan.example.json"
DEFAULT_REPORT_ROOT = WORKSPACE_ROOT / "artifacts" / "test-runs"
COMMON_FAILURE_MARKERS = (
    "traceback",
    "segmentation fault",
    "floating point exception",
    "assert",
    "aborted",
)
INFERENCE_PATTERN = re.compile(
    r"Inference time:\s*(?P<latency_ms>\d+(?:\.\d+)?)\s*ms,\s*"
    r"WAV size \(without header\):\s*(?P<wav_size_bytes>\d+)\s*bytes,\s*"
    r"WAV duration:\s*(?P<wav_duration_ms>\d+(?:\.\d+)?)\s*ms,\s*"
    r"RTF:\s*(?P<rtf>\d+(?:\.\d+)?)"
)
WARNING_PATTERN = re.compile(r"^\[W\b|\bwarning:", re.IGNORECASE)
ERROR_HINT_PATTERN = re.compile(r"\berror\b|traceback|segmentation fault|assert", re.IGNORECASE)
FULL_LOG_TIMESTAMP_PATTERN = re.compile(r"^[A-Z](?P<date>\d{8}) (?P<clock>\d{2}:\d{2}:\d{2}\.\d{6})")
SHORT_LOG_TIMESTAMP_PATTERN = re.compile(r"^\[[A-Z]\s+(?P<month>\d{1,2})/(?P<day>\d{1,2})\s+(?P<clock>\d{1,2}:\d{2}:\d{2}\.\d{3})")


@dataclass(frozen=True)
class TestCase:
    id: str
    name: str
    sentence: str
    category: str
    tags: tuple[str, ...]
    repeat: int
    warmup: int
    latency_threshold_ms: float | None
    rtf_threshold: float | None
    max_warning_count: int | None
    must_contain: tuple[str, ...]
    must_not_contain: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class ParsedLog:
    latency_ms: float | None
    rtf: float | None
    wav_size_bytes: int | None
    wav_duration_ms: float | None
    warning_count: int
    warning_lines: tuple[str, ...]
    error_hints: tuple[str, ...]


@dataclass
class IterationResult:
    case_id: str
    case_name: str
    category: str
    tags: tuple[str, ...]
    iteration: int
    is_warmup: bool
    passed: bool
    exit_code: int
    latency_ms: float | None
    rtf: float | None
    wav_size_bytes: int | None
    wav_duration_ms: float | None
    warning_count: int
    warning_lines: tuple[str, ...]
    error_hints: tuple[str, ...]
    failure_reasons: tuple[str, ...]
    remote_log_path: str
    remote_wav_path: str
    local_log_path: Path
    local_wav_path: Path | None
    started_at_utc: str
    finished_at_utc: str
    elapsed_wall_ms: float
    remote_profile_samples_path: str = ""
    local_profile_samples_path: Path | None = None


def log(message: str) -> None:
    print(f"[tts-suite] {message}")


def fail(message: str, exit_code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_path_fragment(value: str) -> str:
    lowered = value.strip().lower()
    sanitized = re.sub(r"[^a-z0-9]+", "-", lowered)
    return sanitized.strip("-")


def unique_strings(values: list[Any] | tuple[Any, ...] | None) -> tuple[str, ...]:
    if not values:
        return ()
    ordered: dict[str, None] = {}
    for raw in values:
        text = str(raw).strip()
        if text:
            ordered[text] = None
    return tuple(ordered.keys())


def resolve_plan_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        resolved = explicit_path.resolve()
        if not resolved.exists():
            fail(f"测试计划不存在: {resolved}")
        return resolved
    for candidate in (DEFAULT_LOCAL_PLAN, DEFAULT_EXAMPLE_PLAN):
        if candidate.exists():
            return candidate.resolve()
    fail("未找到测试计划。请创建 config/local/tts_test_plan.json，或使用 --plan 指定路径。")


def parse_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def load_test_plan(plan_path: Path) -> tuple[dict[str, Any], list[TestCase]]:
    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"测试计划解析失败 {plan_path}: {exc}")
    if not isinstance(data, dict):
        fail(f"测试计划格式错误 {plan_path}: 顶层必须是 JSON 对象")
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        fail(f"测试计划格式错误 {plan_path}: cases 必须是非空数组")

    defaults = data.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        fail(f"测试计划格式错误 {plan_path}: defaults 必须是对象")

    cases: list[TestCase] = []
    seen_case_ids: set[str] = set()
    default_tags = list(unique_strings(defaults.get("tags")))
    default_must_contain = list(unique_strings(defaults.get("must_contain")))
    default_must_not_contain = list(unique_strings(defaults.get("must_not_contain")))

    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            fail(f"测试计划格式错误 {plan_path}: cases[{index}] 必须是对象")
        raw_case_id = str(raw_case.get("id", "")).strip() or f"case-{index:03d}"
        raw_case_name = str(raw_case.get("name", "")).strip() or raw_case_id
        sentence = str(raw_case.get("sentence", "")).strip()
        if not sentence:
            fail(f"测试计划格式错误 {plan_path}: {raw_case_id} 缺少 sentence")
        if raw_case_id in seen_case_ids:
            fail(f"测试计划格式错误 {plan_path}: 用例 id 重复 {raw_case_id}")
        seen_case_ids.add(raw_case_id)

        repeat = int(raw_case.get("repeat", defaults.get("repeat", 1)))
        warmup = int(raw_case.get("warmup", defaults.get("warmup", 0)))
        if repeat <= 0:
            fail(f"测试计划格式错误 {plan_path}: {raw_case_id} 的 repeat 必须大于 0")
        if warmup < 0:
            fail(f"测试计划格式错误 {plan_path}: {raw_case_id} 的 warmup 不能小于 0")

        case_tags = default_tags + list(unique_strings(raw_case.get("tags")))
        case_must_contain = default_must_contain + list(unique_strings(raw_case.get("must_contain")))
        case_must_not_contain = default_must_not_contain + list(unique_strings(raw_case.get("must_not_contain")))

        cases.append(
            TestCase(
                id=raw_case_id,
                name=raw_case_name,
                sentence=sentence,
                category=str(raw_case.get("category", defaults.get("category", "general"))).strip() or "general",
                tags=unique_strings(case_tags),
                repeat=repeat,
                warmup=warmup,
                latency_threshold_ms=parse_optional_float(raw_case.get("latency_threshold_ms", defaults.get("latency_threshold_ms"))),
                rtf_threshold=parse_optional_float(raw_case.get("rtf_threshold", defaults.get("rtf_threshold"))),
                max_warning_count=parse_optional_int(raw_case.get("max_warning_count", defaults.get("max_warning_count"))),
                must_contain=unique_strings(case_must_contain),
                must_not_contain=unique_strings(case_must_not_contain),
                notes=str(raw_case.get("notes", "")).strip(),
            )
        )

    metadata = {
        "name": str(data.get("name", plan_path.stem)).strip() or plan_path.stem,
        "description": str(data.get("description", "")).strip(),
        "plan_path": path_from_workspace(plan_path),
    }
    return metadata, cases


def select_cases(
    cases: list[TestCase],
    *,
    case_ids: set[str],
    categories: set[str],
    tags: set[str],
) -> list[TestCase]:
    selected: list[TestCase] = []
    for case in cases:
        if case_ids and case.id not in case_ids:
            continue
        if categories and case.category.lower() not in categories:
            continue
        if tags and not any(tag.lower() in tags for tag in case.tags):
            continue
        selected.append(case)
    return selected


def print_selected_cases(metadata: dict[str, Any], cases: list[TestCase]) -> None:
    print(f"Suite: {metadata['name']}")
    if metadata.get("description"):
        print(f"Description: {metadata['description']}")
    print(f"Plan: {metadata['plan_path']}")
    print(f"Cases: {len(cases)}")
    for case in cases:
        tags = ", ".join(case.tags) or "-"
        print(
            f"- {case.id} | category={case.category} | repeat={case.repeat} | "
            f"warmup={case.warmup} | tags={tags} | sentence={case.sentence}"
        )


def run_remote_capture(client: Any, command: str, *, timeout: int) -> tuple[int, str]:
    transport = client.get_transport()
    if transport is None:
        return 255, "SSH transport is not available"
    channel = transport.open_session()
    channel.get_pty()
    channel.exec_command(command)
    start_time = time.monotonic()
    output_parts: list[str] = []
    timed_out = False
    while True:
        if channel.recv_ready():
            output_parts.append(channel.recv(4096).decode("utf-8", "ignore"))
        if channel.recv_stderr_ready():
            output_parts.append(channel.recv_stderr(4096).decode("utf-8", "ignore"))
        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
            break
        if timeout and time.monotonic() - start_time > timeout:
            timed_out = True
            channel.close()
            break
        time.sleep(0.1)
    if timed_out:
        return 124, "".join(output_parts) + "\nCommand timed out"
    return channel.recv_exit_status(), "".join(output_parts)


def download_remote_file(client: Any, remote_path: str, local_path: Path) -> None:
    ensure_parent(local_path)
    sftp = client.open_sftp()
    try:
        sftp.get(remote_path, str(local_path))
    finally:
        sftp.close()


def parse_tts_log(log_text: str) -> ParsedLog:
    match = INFERENCE_PATTERN.search(log_text)
    latency_ms = float(match.group("latency_ms")) if match else None
    rtf = float(match.group("rtf")) if match else None
    wav_size_bytes = int(match.group("wav_size_bytes")) if match else None
    wav_duration_ms = float(match.group("wav_duration_ms")) if match else None
    warning_lines: list[str] = []
    error_hints: list[str] = []
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if WARNING_PATTERN.search(line):
            warning_lines.append(line)
        if ERROR_HINT_PATTERN.search(line):
            error_hints.append(line)
    return ParsedLog(
        latency_ms=latency_ms,
        rtf=rtf,
        wav_size_bytes=wav_size_bytes,
        wav_duration_ms=wav_duration_ms,
        warning_count=len(warning_lines),
        warning_lines=tuple(dict.fromkeys(warning_lines)),
        error_hints=tuple(dict.fromkeys(error_hints)),
    )


def evaluate_iteration(case: TestCase, parsed: ParsedLog, log_text: str, *, exit_code: int, is_warmup: bool) -> tuple[str, ...]:
    reasons: list[str] = []
    lowered_log = log_text.lower()
    if exit_code != 0:
        reasons.append(f"远端命令退出码 {exit_code}")
    if parsed.latency_ms is None or parsed.rtf is None:
        reasons.append("日志中缺少 Inference time / RTF 指标")
    if case.max_warning_count is not None and parsed.warning_count > case.max_warning_count:
        reasons.append(f"告警数 {parsed.warning_count} 超过阈值 {case.max_warning_count}")
    for marker in COMMON_FAILURE_MARKERS:
        if marker in lowered_log:
            reasons.append(f"命中通用失败特征: {marker}")
    for marker in case.must_contain:
        if marker.lower() not in lowered_log:
            reasons.append(f"缺少必备日志片段: {marker}")
    for marker in case.must_not_contain:
        if marker.lower() in lowered_log:
            reasons.append(f"命中禁止日志片段: {marker}")
    if not is_warmup and parsed.latency_ms is not None and case.latency_threshold_ms is not None:
        if parsed.latency_ms > case.latency_threshold_ms:
            reasons.append(f"延迟 {parsed.latency_ms:.3f} ms 超过阈值 {case.latency_threshold_ms:.3f} ms")
    if not is_warmup and parsed.rtf is not None and case.rtf_threshold is not None:
        if parsed.rtf > case.rtf_threshold:
            reasons.append(f"RTF {parsed.rtf:.6f} 超过阈值 {case.rtf_threshold:.6f}")
    deduped: dict[str, None] = {}
    for reason in reasons:
        deduped[reason] = None
    return tuple(deduped.keys())


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    if lower_index == upper_index:
        return lower_value
    return lower_value + (upper_value - lower_value) * (position - lower_index)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def round_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def relative_to(base_dir: Path, target: Path | None) -> str | None:
    if target is None:
        return None
    try:
        return target.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return target.resolve().as_posix()


def path_from_workspace(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_report_artifact_path(report_dir: Path, stored_path: str | None) -> Path | None:
    if not stored_path:
        return None
    candidate = Path(stored_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (report_dir / candidate).resolve()


def parse_log_datetime(raw_line: str, *, default_year: int | None) -> tuple[datetime | None, int | None]:
    match = FULL_LOG_TIMESTAMP_PATTERN.match(raw_line)
    if match:
        parsed = datetime.strptime(f"{match.group('date')} {match.group('clock')}", "%Y%m%d %H:%M:%S.%f")
        return parsed, parsed.year
    match = SHORT_LOG_TIMESTAMP_PATTERN.match(raw_line)
    if match and default_year is not None:
        month = int(match.group("month"))
        day = int(match.group("day"))
        parsed = datetime.strptime(f"{default_year}-{month:02d}-{day:02d} {match.group('clock')}", "%Y-%m-%d %H:%M:%S.%f")
        return parsed, default_year
    return None, default_year


def elapsed_between_ms(start_time: datetime | None, end_time: datetime | None) -> float | None:
    if start_time is None or end_time is None:
        return None
    return max((end_time - start_time).total_seconds() * 1000.0, 0.0)


def make_profile_node(name: str, value_ms: float, *, children: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "value_ms": round(value_ms, 3),
        "children": children or [],
    }


def clone_profile_node(node: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(node)


def merge_profile_nodes(target: dict[str, Any], source: dict[str, Any]) -> None:
    if target.get("name") != source.get("name"):
        raise ValueError("Cannot merge profile nodes with different names")
    target["value_ms"] = round(float(target.get("value_ms", 0.0)) + float(source.get("value_ms", 0.0)), 3)
    child_map = {child["name"]: child for child in target.get("children", [])}
    for child in source.get("children", []):
        existing = child_map.get(child["name"])
        if existing is None:
            target.setdefault("children", []).append(clone_profile_node(child))
            continue
        merge_profile_nodes(existing, child)
    target.setdefault("children", []).sort(key=lambda item: item.get("value_ms", 0.0), reverse=True)


def iteration_payload_key(iteration_payload: dict[str, Any]) -> tuple[str, bool, int, str]:
    return (
        str(iteration_payload.get("case_id", "")).strip(),
        bool(iteration_payload.get("is_warmup", False)),
        int(iteration_payload.get("iteration", 0) or 0),
        str(iteration_payload.get("local_log_path", "")).strip(),
    )


def parse_profile_samples_csv(samples_path: Path | None) -> dict[str, Any] | None:
    if samples_path is None or not samples_path.exists():
        return None
    samples: list[dict[str, float | int | str]] = []
    with samples_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            elapsed_ms = parse_optional_float(row.get("elapsed_ms"))
            if elapsed_ms is None:
                continue
            rss_kb = parse_optional_int(row.get("rss_kb")) or 0
            vm_size_kb = parse_optional_int(row.get("vm_size_kb")) or 0
            threads = parse_optional_int(row.get("threads")) or 0
            npu_core0 = parse_optional_int(row.get("npu_core0_percent")) or 0
            npu_core1 = parse_optional_int(row.get("npu_core1_percent")) or 0
            npu_core2 = parse_optional_int(row.get("npu_core2_percent")) or 0
            npu_max = max(npu_core0, npu_core1, npu_core2)
            samples.append(
                {
                    "elapsed_ms": elapsed_ms,
                    "rss_kb": rss_kb,
                    "vm_size_kb": vm_size_kb,
                    "threads": threads,
                    "state": str(row.get("state") or "?"),
                    "npu_core0_percent": npu_core0,
                    "npu_core1_percent": npu_core1,
                    "npu_core2_percent": npu_core2,
                    "npu_max_percent": npu_max,
                }
            )
    if not samples:
        return None

    npu_active_samples = [sample for sample in samples if int(sample["npu_max_percent"]) > 0]
    first_npu_sample_ms = float(npu_active_samples[0]["elapsed_ms"]) if npu_active_samples else None
    last_npu_sample_ms = float(npu_active_samples[-1]["elapsed_ms"]) if npu_active_samples else None
    npu_active_window_ms = None
    if first_npu_sample_ms is not None and last_npu_sample_ms is not None:
        npu_active_window_ms = max(last_npu_sample_ms - first_npu_sample_ms, 0.0)

    return {
        "sample_count": len(samples),
        "peak_rss_kb": max(int(sample["rss_kb"]) for sample in samples),
        "peak_vm_size_kb": max(int(sample["vm_size_kb"]) for sample in samples),
        "max_threads": max(int(sample["threads"]) for sample in samples),
        "peak_npu_load_percent": max(int(sample["npu_max_percent"]) for sample in samples),
        "mean_npu_load_percent": round(statistics.fmean(int(sample["npu_max_percent"]) for sample in samples), 3),
        "first_npu_sample_ms": first_npu_sample_ms,
        "last_npu_sample_ms": last_npu_sample_ms,
        "npu_active_window_ms": npu_active_window_ms,
    }


def build_iteration_profile_from_log(iteration_payload: dict[str, Any], report_dir: Path) -> dict[str, Any] | None:
    local_log_path = resolve_report_artifact_path(report_dir, str(iteration_payload.get("local_log_path") or ""))
    if local_log_path is None or not local_log_path.exists():
        return None

    log_text = local_log_path.read_text(encoding="utf-8", errors="ignore")
    first_timestamp: datetime | None = None
    default_year: int | None = None
    segment_start: datetime | None = None
    segment_end: datetime | None = None
    phoneme_start: datetime | None = None
    phoneme_end: datetime | None = None
    inference_end: datetime | None = None
    normalization_points: list[datetime] = []
    device_probe_points: list[datetime] = []
    model_load_points: list[datetime] = []
    saw_rknn_marker = False

    for raw_line in log_text.splitlines():
        lowered = raw_line.lower()
        if "rknn" in lowered or "rknpu" in lowered or "librknnrt" in lowered:
            saw_rknn_marker = True

        timestamp, default_year = parse_log_datetime(raw_line, default_year=default_year)
        if timestamp is None:
            continue
        if first_timestamp is None:
            first_timestamp = timestamp
        if "Start to segment sentences by punctuation" in raw_line:
            segment_start = timestamp
        elif "Segment sentences through punctuation successfully" in raw_line:
            segment_end = timestamp
        elif "Start to get the phoneme and tone id sequence of each sentence" in raw_line:
            phoneme_start = timestamp
        elif "After normalization sentence is:" in raw_line:
            normalization_points.append(timestamp)
        elif "Get the phoneme id sequence of each sentence successfully" in raw_line:
            phoneme_end = timestamp
        elif "device_info.cc" in raw_line:
            device_probe_points.append(timestamp)
        elif "LoadModelFbsFromFile" in raw_line:
            model_load_points.append(timestamp)
        elif "Inference time:" in raw_line:
            inference_end = timestamp

    parsed_log = parse_tts_log(log_text)
    samples_path = resolve_report_artifact_path(report_dir, str(iteration_payload.get("local_profile_samples_path") or ""))
    sample_telemetry = parse_profile_samples_csv(samples_path)
    reported_inference_ms = parse_optional_float(iteration_payload.get("latency_ms"))
    if reported_inference_ms is None:
        reported_inference_ms = parsed_log.latency_ms

    total_wall_ms = parse_optional_float(iteration_payload.get("elapsed_wall_ms"))
    if total_wall_ms is None and first_timestamp is not None and inference_end is not None:
        total_wall_ms = elapsed_between_ms(first_timestamp, inference_end)
    if total_wall_ms is None or total_wall_ms <= 0.0:
        return None

    root_children: list[dict[str, Any]] = []
    startup_ms = elapsed_between_ms(first_timestamp, segment_start)
    if startup_ms is not None and startup_ms > 0.0:
        root_children.append(make_profile_node("startup.initialization", startup_ms))

    segmentation_ms = elapsed_between_ms(segment_start, segment_end)
    if segmentation_ms is not None and segmentation_ms > 0.0:
        root_children.append(make_profile_node("frontend.segment_sentences", segmentation_ms))

    phoneme_children: list[dict[str, Any]] = []
    phoneme_total_ms = elapsed_between_ms(phoneme_start, phoneme_end)
    if phoneme_total_ms is not None and phoneme_total_ms > 0.0:
        first_normalized = normalization_points[0] if normalization_points else None
        last_normalized = normalization_points[-1] if normalization_points else None
        pre_normalize_ms = elapsed_between_ms(phoneme_start, first_normalized)
        if pre_normalize_ms is not None and pre_normalize_ms > 0.0:
            phoneme_children.append(make_profile_node("frontend.pre_normalization", pre_normalize_ms))
        if len(normalization_points) >= 2 and first_normalized is not None and last_normalized is not None:
            normalization_ms = elapsed_between_ms(first_normalized, last_normalized)
            if normalization_ms is not None and normalization_ms > 0.0:
                phoneme_children.append(make_profile_node("frontend.normalize_sentences", normalization_ms))
        build_phone_ms = elapsed_between_ms(last_normalized or phoneme_start, phoneme_end)
        if build_phone_ms is not None and build_phone_ms > 0.0:
            phoneme_children.append(make_profile_node("frontend.build_phone_ids", build_phone_ms))
        root_children.append(make_profile_node("frontend.phoneme_pipeline", phoneme_total_ms, children=phoneme_children))

    runtime_children: list[dict[str, Any]] = []
    runtime_window_ms = elapsed_between_ms(phoneme_end, inference_end)
    runtime_total_ms = runtime_window_ms if runtime_window_ms is not None else 0.0
    if reported_inference_ms is not None and reported_inference_ms > 0.0:
        runtime_total_ms = max(runtime_total_ms, reported_inference_ms)
        runtime_accounted_ms = 0.0
        inference_start = None
        inference_relative_start_ms = None
        if inference_end is not None:
            inference_start = inference_end - timedelta(milliseconds=reported_inference_ms)
        if first_timestamp is not None and inference_start is not None:
            inference_relative_start_ms = elapsed_between_ms(first_timestamp, inference_start)

        runtime_probe_start = device_probe_points[0] if device_probe_points else (model_load_points[0] if model_load_points else None)
        runtime_bootstrap_ms = elapsed_between_ms(phoneme_end, runtime_probe_start)
        if runtime_bootstrap_ms is not None and runtime_bootstrap_ms > 0.0:
            runtime_children.append(make_profile_node("runtime.bootstrap", runtime_bootstrap_ms))
            runtime_accounted_ms += runtime_bootstrap_ms

        if model_load_points and inference_start is not None:
            model_load_start = model_load_points[0]
            model_load_end = model_load_points[-1] if len(model_load_points) > 1 else inference_start
            runtime_model_load_ms = elapsed_between_ms(model_load_start, model_load_end)
            if runtime_model_load_ms is not None and runtime_model_load_ms > 0.0:
                runtime_children.append(make_profile_node("runtime.model_load", runtime_model_load_ms))
                runtime_accounted_ms += runtime_model_load_ms

            runtime_inference_prepare_ms = elapsed_between_ms(model_load_end, inference_start)
            if runtime_inference_prepare_ms is not None and runtime_inference_prepare_ms > 0.0:
                runtime_children.append(make_profile_node("runtime.inference_prepare", runtime_inference_prepare_ms))
                runtime_accounted_ms += runtime_inference_prepare_ms

        runtime_prepare_ms = None
        if not model_load_points and runtime_window_ms is not None:
            runtime_prepare_ms = max(runtime_window_ms - reported_inference_ms, 0.0)
        if runtime_prepare_ms is not None and runtime_prepare_ms > 0.0:
            runtime_children.append(make_profile_node("runtime.prepare", runtime_prepare_ms))
            runtime_accounted_ms += runtime_prepare_ms

        reported_inference_children: list[dict[str, Any]] = []
        if sample_telemetry is not None and inference_relative_start_ms is not None:
            first_npu_sample_ms = parse_optional_float(sample_telemetry.get("first_npu_sample_ms"))
            last_npu_sample_ms = parse_optional_float(sample_telemetry.get("last_npu_sample_ms"))
            if first_npu_sample_ms is not None and last_npu_sample_ms is not None:
                npu_start_within_inference = max(first_npu_sample_ms - inference_relative_start_ms, 0.0)
                npu_end_within_inference = min(last_npu_sample_ms - inference_relative_start_ms, reported_inference_ms)
                npu_active_duration_ms = max(npu_end_within_inference - npu_start_within_inference, 0.0)
                cpu_before_npu_ms = max(npu_start_within_inference, 0.0)
                cpu_after_npu_ms = max(reported_inference_ms - cpu_before_npu_ms - npu_active_duration_ms, 0.0)
                if cpu_before_npu_ms > 0.0:
                    reported_inference_children.append(make_profile_node("runtime.cpu_before_npu", cpu_before_npu_ms))
                if npu_active_duration_ms > 0.0:
                    reported_inference_children.append(make_profile_node("runtime.npu_active", npu_active_duration_ms))
                if cpu_after_npu_ms > 0.0:
                    reported_inference_children.append(make_profile_node("runtime.cpu_after_npu", cpu_after_npu_ms))

        runtime_children.append(make_profile_node("runtime.reported_inference", reported_inference_ms, children=reported_inference_children))
        runtime_accounted_ms += reported_inference_ms

        runtime_gap_ms = runtime_total_ms - runtime_accounted_ms
        if runtime_gap_ms > 0.0:
            runtime_children.append(make_profile_node("runtime.postprocess_gap", runtime_gap_ms))
    if runtime_total_ms > 0.0:
        root_children.append(make_profile_node("runtime.execution", runtime_total_ms, children=runtime_children))

    accounted_ms = sum(float(child.get("value_ms", 0.0)) for child in root_children)
    residual_ms = total_wall_ms - accounted_ms
    if residual_ms > 0.0:
        root_children.append(make_profile_node("runtime.unaccounted", residual_ms))

    notes = ["火焰图基于板端日志时间戳与 reported inference 反推，不依赖 perf 采样。"]
    if sample_telemetry is not None:
        notes.append("检测到板端 profile 采样文件，已将 RSS/NPU 采样纳入运行时分解。")
    if not saw_rknn_marker:
        notes.append("当前日志未检测到 RKNN/NPU 专属标记，现阶段展示的是通用板端阶段剖析。")

    return {
        "source": "log-derived-stage-profile",
        "notes": notes,
        "has_rknn_markers": saw_rknn_marker,
        "log_path": relative_to(report_dir, local_log_path),
        "telemetry": sample_telemetry,
        "root": make_profile_node("board.tts_request", total_wall_ms, children=root_children),
    }


def choose_case_profile(case_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | tuple[None, None]:
    iterations = case_payload.get("iterations", [])
    if not isinstance(iterations, list):
        return None, None
    for iteration in iterations:
        if iteration.get("profile") and not iteration.get("is_warmup"):
            return iteration["profile"], iteration
    for iteration in iterations:
        if iteration.get("profile"):
            return iteration["profile"], iteration
    return None, None


def augment_payload_with_profiles(payload: dict[str, Any], report_dir: Path) -> dict[str, Any]:
    top_level_iterations = payload.get("iterations", [])
    if not isinstance(top_level_iterations, list):
        payload["profiles"] = {
            "available": False,
            "source": "log-derived-stage-profile",
            "notes": ["结果中不存在 iteration 明细，无法补全板端 profile。"],
        }
        return payload

    profile_map: dict[tuple[str, bool, int, str], dict[str, Any]] = {}
    measured_profiles: list[dict[str, Any]] = []
    measured_profile_count = 0
    cases_with_profiles = 0
    saw_rknn_markers = False
    notes: dict[str, None] = {}

    for iteration in top_level_iterations:
        if not isinstance(iteration, dict):
            continue
        profile = build_iteration_profile_from_log(iteration, report_dir)
        if profile is not None:
            iteration["profile"] = profile
            profile_map[iteration_payload_key(iteration)] = profile
            if not iteration.get("is_warmup", False):
                measured_profiles.append(profile)
                measured_profile_count += 1
            saw_rknn_markers = saw_rknn_markers or bool(profile.get("has_rknn_markers"))
            for note in profile.get("notes", []):
                notes[str(note)] = None
        else:
            iteration.pop("profile", None)

    for case in payload.get("cases", []):
        if not isinstance(case, dict):
            continue
        profiled_iterations = 0
        for iteration in case.get("iterations", []):
            if not isinstance(iteration, dict):
                continue
            profile = profile_map.get(iteration_payload_key(iteration))
            if profile is None:
                iteration.pop("profile", None)
                continue
            iteration["profile"] = profile
            profiled_iterations += 1
        case["profiled_iterations"] = profiled_iterations
        case_profile, case_profile_iteration = choose_case_profile(case)
        if case_profile is None or case_profile_iteration is None:
            case.pop("profile", None)
            case.pop("profile_iteration", None)
            continue
        cases_with_profiles += 1
        case["profile"] = case_profile
        case["profile_iteration"] = {
            "iteration": int(case_profile_iteration.get("iteration", 0) or 0),
            "is_warmup": bool(case_profile_iteration.get("is_warmup", False)),
        }

    if not measured_profiles:
        payload["profiles"] = {
            "available": False,
            "source": "log-derived-stage-profile",
            "notes": ["当前结果缺少可解析的日志时间戳，无法生成火焰图。"],
        }
        return payload

    aggregate_root = clone_profile_node(measured_profiles[0]["root"])
    for profile in measured_profiles[1:]:
        merge_profile_nodes(aggregate_root, profile["root"])

    payload["profiles"] = {
        "available": True,
        "source": "log-derived-stage-profile",
        "measured_iterations_profiled": measured_profile_count,
        "cases_with_profiles": cases_with_profiles,
        "has_rknn_markers": saw_rknn_markers,
        "notes": list(notes.keys()),
        "aggregate_profile": {
            "source": "log-derived-stage-profile",
            "notes": list(notes.keys()),
            "has_rknn_markers": saw_rknn_markers,
            "root": aggregate_root,
        },
    }
    return payload


def execute_case(
    client: Any,
    *,
    case: TestCase,
    case_index: int,
    report_dir: Path,
    remote_dir: str,
    remote_run_root: str,
    remote_timeout: int,
    download_wavs: bool,
    deep_profile: bool,
    profile_sample_interval_ms: int,
) -> list[IterationResult]:
    artifact_case_key = safe_path_fragment(case.id) or f"case-{case_index:03d}"
    remote_case_dir = f"{remote_run_root}/{artifact_case_key}"
    total_iterations = case.warmup + case.repeat
    results: list[IterationResult] = []
    for iteration_index in range(1, total_iterations + 1):
        is_warmup = iteration_index <= case.warmup
        phase_name = "warmup" if is_warmup else "run"
        phase_iteration = iteration_index if is_warmup else iteration_index - case.warmup
        artifact_name = f"{artifact_case_key}-{phase_name}-{phase_iteration:03d}"
        remote_log_path = f"{remote_case_dir}/{artifact_name}.log"
        remote_wav_path = f"{remote_case_dir}/{artifact_name}.wav"
        remote_profile_samples_path = f"{remote_case_dir}/{artifact_name}.profile-samples.csv" if deep_profile else ""
        local_log_path = report_dir / "logs" / artifact_case_key / f"{artifact_name}.log"
        local_wav_path = report_dir / "audio" / artifact_case_key / f"{artifact_name}.wav" if download_wavs else None
        local_profile_samples_path = report_dir / "profiles" / artifact_case_key / f"{artifact_name}.profile-samples.csv" if deep_profile else None
        if deep_profile:
            remote_command = (
                f"mkdir -p {sh_quote(remote_case_dir)}; "
                f"./tools/profile_tts_inference.sh --sentence {sh_quote(case.sentence)} "
                f"--output_wav {sh_quote(remote_wav_path)} "
                f"--log {sh_quote(remote_log_path)} "
                f"--samples-csv {sh_quote(remote_profile_samples_path)} "
                f"--sample-interval-ms {sh_quote(str(profile_sample_interval_ms))}"
            )
        else:
            remote_command = (
                "set -o pipefail; "
                f"mkdir -p {sh_quote(remote_case_dir)}; "
                f"./run_tts.sh --sentence {sh_quote(case.sentence)} --output_wav {sh_quote(remote_wav_path)} 2>&1 | tee {sh_quote(remote_log_path)}"
            )
        started_at_utc = utc_now_iso()
        started_perf = time.perf_counter()
        exit_code, output = run_remote_capture(
            client,
            f"cd {sh_quote(remote_dir)} && bash -lc {sh_quote(remote_command)}",
            timeout=remote_timeout,
        )
        elapsed_wall_ms = (time.perf_counter() - started_perf) * 1000.0
        finished_at_utc = utc_now_iso()
        write_text(local_log_path, output)
        download_error: str | None = None
        if download_wavs and exit_code == 0 and local_wav_path is not None:
            try:
                download_remote_file(client, remote_wav_path, local_wav_path)
            except OSError as exc:
                download_error = str(exc)
        profile_download_error: str | None = None
        if deep_profile and local_profile_samples_path is not None:
            try:
                download_remote_file(client, remote_profile_samples_path, local_profile_samples_path)
            except OSError as exc:
                profile_download_error = str(exc)
        parsed = parse_tts_log(output)
        failure_reasons = list(evaluate_iteration(case, parsed, output, exit_code=exit_code, is_warmup=is_warmup))
        if download_error is not None:
            failure_reasons.append(f"下载音频失败: {download_error}")
        if profile_download_error is not None:
            failure_reasons.append(f"下载 profile 采样失败: {profile_download_error}")
        results.append(
            IterationResult(
                case_id=case.id,
                case_name=case.name,
                category=case.category,
                tags=case.tags,
                iteration=phase_iteration,
                is_warmup=is_warmup,
                passed=not failure_reasons,
                exit_code=exit_code,
                latency_ms=parsed.latency_ms,
                rtf=parsed.rtf,
                wav_size_bytes=parsed.wav_size_bytes,
                wav_duration_ms=parsed.wav_duration_ms,
                warning_count=parsed.warning_count,
                warning_lines=parsed.warning_lines,
                error_hints=parsed.error_hints,
                failure_reasons=tuple(failure_reasons),
                remote_log_path=remote_log_path,
                remote_wav_path=remote_wav_path,
                local_log_path=local_log_path,
                local_wav_path=local_wav_path,
                remote_profile_samples_path=remote_profile_samples_path,
                local_profile_samples_path=local_profile_samples_path,
                started_at_utc=started_at_utc,
                finished_at_utc=finished_at_utc,
                elapsed_wall_ms=elapsed_wall_ms,
            )
        )
    return results


def serialize_iteration(result: IterationResult, report_dir: Path) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "case_name": result.case_name,
        "category": result.category,
        "tags": list(result.tags),
        "iteration": result.iteration,
        "is_warmup": result.is_warmup,
        "passed": result.passed,
        "exit_code": result.exit_code,
        "latency_ms": round_or_none(result.latency_ms),
        "rtf": round_or_none(result.rtf, digits=6),
        "wav_size_bytes": result.wav_size_bytes,
        "wav_duration_ms": result.wav_duration_ms,
        "warning_count": result.warning_count,
        "warning_lines": list(result.warning_lines),
        "error_hints": list(result.error_hints),
        "failure_reasons": list(result.failure_reasons),
        "remote_log_path": result.remote_log_path,
        "remote_wav_path": result.remote_wav_path,
        "remote_profile_samples_path": result.remote_profile_samples_path,
        "local_log_path": relative_to(report_dir, result.local_log_path),
        "local_wav_path": relative_to(report_dir, result.local_wav_path),
        "local_profile_samples_path": relative_to(report_dir, result.local_profile_samples_path),
        "started_at_utc": result.started_at_utc,
        "finished_at_utc": result.finished_at_utc,
        "elapsed_wall_ms": round(result.elapsed_wall_ms, 3),
    }


def summarize_case(case: TestCase, report_dir: Path, iterations: list[IterationResult]) -> dict[str, Any]:
    measured = [item for item in iterations if not item.is_warmup]
    latencies = [item.latency_ms for item in measured if item.latency_ms is not None]
    rtfs = [item.rtf for item in measured if item.rtf is not None]
    failure_reasons: dict[str, None] = {}
    for item in iterations:
        for reason in item.failure_reasons:
            failure_reasons[reason] = None
    return {
        "id": case.id,
        "name": case.name,
        "category": case.category,
        "tags": list(case.tags),
        "sentence": case.sentence,
        "repeat": case.repeat,
        "warmup": case.warmup,
        "notes": case.notes,
        "passed": all(item.passed for item in iterations),
        "measured_iterations": len(measured),
        "warmup_iterations": len(iterations) - len(measured),
        "mean_latency_ms": round_or_none(mean_or_none(latencies)),
        "p50_latency_ms": round_or_none(percentile(latencies, 0.50)),
        "p95_latency_ms": round_or_none(percentile(latencies, 0.95)),
        "max_latency_ms": round_or_none(max(latencies) if latencies else None),
        "mean_rtf": round_or_none(mean_or_none(rtfs), digits=6),
        "warning_count": sum(item.warning_count for item in iterations),
        "failure_reasons": list(failure_reasons.keys()),
        "thresholds": {
            "latency_threshold_ms": case.latency_threshold_ms,
            "rtf_threshold": case.rtf_threshold,
            "max_warning_count": case.max_warning_count,
            "must_contain": list(case.must_contain),
            "must_not_contain": list(case.must_not_contain),
        },
        "iterations": [serialize_iteration(item, report_dir) for item in iterations],
    }


def build_report_payload(
    *,
    metadata: dict[str, Any],
    cases: list[TestCase],
    report_dir: Path,
    host: str,
    remote_dir: str,
    source_ip: str | None,
    results: list[IterationResult],
    filters: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[IterationResult]] = {case.id: [] for case in cases}
    for result in results:
        grouped.setdefault(result.case_id, []).append(result)

    case_payloads = [summarize_case(case, report_dir, grouped.get(case.id, [])) for case in cases]
    measured_results = [item for item in results if not item.is_warmup]
    measured_latencies = [item.latency_ms for item in measured_results if item.latency_ms is not None]
    measured_rtfs = [item.rtf for item in measured_results if item.rtf is not None]
    category_payloads: list[dict[str, Any]] = []
    categories = sorted({case.category for case in cases})
    for category in categories:
        category_cases = [case_payload for case_payload in case_payloads if case_payload["category"] == category]
        category_iterations = [
            item
            for item in measured_results
            if item.category == category
        ]
        category_latencies = [item.latency_ms for item in category_iterations if item.latency_ms is not None]
        category_rtfs = [item.rtf for item in category_iterations if item.rtf is not None]
        category_payloads.append(
            {
                "category": category,
                "cases_total": len(category_cases),
                "cases_passed": sum(1 for item in category_cases if item["passed"]),
                "measured_iterations": len(category_iterations),
                "mean_latency_ms": round_or_none(mean_or_none(category_latencies)),
                "p95_latency_ms": round_or_none(percentile(category_latencies, 0.95)),
                "mean_rtf": round_or_none(mean_or_none(category_rtfs), digits=6),
            }
        )

    top_slow_cases = sorted(
        case_payloads,
        key=lambda item: (item["p95_latency_ms"] is None, item["p95_latency_ms"] or -1.0),
        reverse=True,
    )[:8]
    failing_cases = [item for item in case_payloads if not item["passed"]]
    payload = {
        "meta": {
            "suite_name": metadata["name"],
            "description": metadata.get("description", ""),
            "plan_path": metadata["plan_path"],
            "report_dir": path_from_workspace(report_dir),
            "generated_at_utc": utc_now_iso(),
            "host": host,
            "remote_dir": remote_dir,
            "source_ip": source_ip or "",
            "filters": filters,
        },
        "summary": {
            "cases_total": len(case_payloads),
            "cases_passed": sum(1 for item in case_payloads if item["passed"]),
            "measured_iterations": len(measured_results),
            "warmup_iterations": len(results) - len(measured_results),
            "iterations_passed": sum(1 for item in measured_results if item.passed),
            "mean_latency_ms": round_or_none(mean_or_none(measured_latencies)),
            "p50_latency_ms": round_or_none(percentile(measured_latencies, 0.50)),
            "p95_latency_ms": round_or_none(percentile(measured_latencies, 0.95)),
            "max_latency_ms": round_or_none(max(measured_latencies) if measured_latencies else None),
            "mean_rtf": round_or_none(mean_or_none(measured_rtfs), digits=6),
            "case_pass_rate_percent": round((sum(1 for item in case_payloads if item["passed"]) / len(case_payloads) * 100.0), 2)
            if case_payloads
            else 0.0,
            "iteration_pass_rate_percent": round((sum(1 for item in measured_results if item.passed) / len(measured_results) * 100.0), 2)
            if measured_results
            else 0.0,
        },
        "categories": category_payloads,
        "cases": case_payloads,
        "iterations": [serialize_iteration(item, report_dir) for item in results],
        "top_slow_cases": [
            {
                "id": item["id"],
                "name": item["name"],
                "category": item["category"],
                "p95_latency_ms": item["p95_latency_ms"],
                "mean_latency_ms": item["mean_latency_ms"],
                "mean_rtf": item["mean_rtf"],
                "passed": item["passed"],
            }
            for item in top_slow_cases
        ],
        "failing_cases": [
            {
                "id": item["id"],
                "name": item["name"],
                "category": item["category"],
                "failure_reasons": item["failure_reasons"],
            }
            for item in failing_cases
        ],
    }
    return augment_payload_with_profiles(payload, report_dir)


def format_metric(value: float | None, suffix: str = "", digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}{suffix}"


def get_case_preview_iteration(case: dict[str, Any]) -> dict[str, Any] | None:
    iterations = case.get("iterations", [])
    if not isinstance(iterations, list):
        return None
    for iteration in iterations:
        if iteration.get("local_wav_path") and not iteration.get("is_warmup"):
            return iteration
    for iteration in iterations:
        if iteration.get("local_wav_path"):
            return iteration
    return None


def render_audio_player(local_wav_path: str | None, *, compact: bool) -> str:
    if not local_wav_path:
        return "<span class='muted-inline'>未下载音频</span>"
    css_class = "audio-player compact" if compact else "audio-player"
    return f"<audio class='{css_class}' controls preload='none' src='{html.escape(local_wav_path)}'></audio>"


def render_bar_chart(title: str, items: list[tuple[str, float | None, bool]], unit: str) -> str:
    available_items = [(label, value, passed) for label, value, passed in items if value is not None]
    if not available_items:
        return f"<section class=\"chart-card\"><h3>{html.escape(title)}</h3><p class=\"empty\">无可视化数据</p></section>"
    max_value = max(value for _, value, _ in available_items)
    chart_width = 360
    label_width = 160
    row_height = 34
    width = 580
    height = 32 + len(available_items) * row_height
    scale = chart_width / max(max_value, 1.0)
    svg_rows: list[str] = []
    for index, (label, value, passed) in enumerate(available_items):
        y = 18 + index * row_height
        bar_width = value * scale
        color = "#2e8b57" if passed else "#c14f3f"
        svg_rows.append(
            "".join(
                [
                    f"<text x='0' y='{y + 12}' class='chart-label'>{html.escape(label)}</text>",
                    f"<rect x='{label_width}' y='{y}' width='{bar_width:.2f}' height='18' rx='6' fill='{color}'></rect>",
                    f"<text x='{label_width + bar_width + 8:.2f}' y='{y + 13}' class='chart-value'>{value:.3f}{html.escape(unit)}</text>",
                ]
            )
        )
    svg = (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        + "".join(svg_rows)
        + "</svg>"
    )
    return f"<section class=\"chart-card\"><h3>{html.escape(title)}</h3>{svg}</section>"


def flame_color(name: str) -> str:
    hue = sum((index + 1) * ord(character) for index, character in enumerate(name)) % 360
    return f"hsl({hue} 64% 68%)"


def profile_depth(node: dict[str, Any]) -> int:
    children = node.get("children", [])
    if not children:
        return 1
    return 1 + max(profile_depth(child) for child in children)


def collect_flame_graph_nodes(
    node: dict[str, Any],
    *,
    x: float,
    width: float,
    depth: int,
    max_depth: int,
    row_height: int,
    nodes: list[dict[str, Any]],
    path: tuple[str, ...],
) -> None:
    value_ms = float(node.get("value_ms", 0.0) or 0.0)
    if value_ms <= 0.0 or width <= 0.5:
        return
    y = 8 + (max_depth - depth - 1) * row_height
    current_path = (*path, str(node.get("name", "")))
    nodes.append(
        {
            "name": str(node.get("name", "")),
            "value_ms": value_ms,
            "x": x,
            "y": y,
            "width": width,
            "path": current_path,
        }
    )
    child_x = x
    for child in sorted(node.get("children", []), key=lambda item: item.get("value_ms", 0.0), reverse=True):
        child_value_ms = float(child.get("value_ms", 0.0) or 0.0)
        child_width = width * child_value_ms / value_ms if value_ms else 0.0
        collect_flame_graph_nodes(
            child,
            x=child_x,
            width=child_width,
            depth=depth + 1,
            max_depth=max_depth,
            row_height=row_height,
            nodes=nodes,
            path=current_path,
        )
        child_x += child_width


def render_profile_stage_breakdown(profile_bundle: dict[str, Any] | None) -> str:
    if not profile_bundle:
        return ""
    root = profile_bundle.get("root")
    if not isinstance(root, dict):
        return ""
    total_ms = float(root.get("value_ms", 0.0) or 0.0)
    if total_ms <= 0.0:
        return ""
    cards: list[str] = []
    for child in sorted(root.get("children", []), key=lambda item: item.get("value_ms", 0.0), reverse=True):
        value_ms = float(child.get("value_ms", 0.0) or 0.0)
        ratio_percent = value_ms / total_ms * 100.0 if total_ms else 0.0
        cards.append(
            "".join(
                [
                    "<article class='profile-chip'>",
                    f"<strong>{html.escape(str(child.get('name', '-')))}</strong>",
                    f"<span>{format_metric(value_ms, ' ms')} / {ratio_percent:.1f}%</span>",
                    "</article>",
                ]
            )
        )
    return "<div class='profile-chip-grid'>" + "".join(cards) + "</div>"


def render_profile_telemetry(profile_bundle: dict[str, Any] | None) -> str:
    if not profile_bundle:
        return ""
    telemetry = profile_bundle.get("telemetry")
    if not isinstance(telemetry, dict):
        return ""
    metrics = [
        ("采样点", str(int(telemetry.get("sample_count", 0) or 0))),
        ("RSS 峰值", f"{int(telemetry.get('peak_rss_kb', 0) or 0)} KB"),
        ("VM 峰值", f"{int(telemetry.get('peak_vm_size_kb', 0) or 0)} KB"),
        ("线程峰值", str(int(telemetry.get("max_threads", 0) or 0))),
        ("NPU 峰值", f"{int(telemetry.get('peak_npu_load_percent', 0) or 0)}%"),
        ("NPU 均值", f"{float(telemetry.get('mean_npu_load_percent', 0.0) or 0.0):.1f}%"),
    ]
    active_window_ms = parse_optional_float(telemetry.get("npu_active_window_ms"))
    if active_window_ms is not None:
        metrics.append(("NPU 活跃窗口", format_metric(active_window_ms, " ms")))
    cards = [
        "".join(
            [
                "<article class='profile-chip'>",
                f"<strong>{html.escape(label)}</strong>",
                f"<span>{html.escape(value)}</span>",
                "</article>",
            ]
        )
        for label, value in metrics
    ]
    return "<div class='profile-chip-grid profile-telemetry-grid'>" + "".join(cards) + "</div>"


def flatten_profile_tree(node: dict[str, Any], *, depth: int = 0) -> list[dict[str, Any]]:
    rows = [
        {
            "name": str(node.get("name", "-")),
            "value_ms": float(node.get("value_ms", 0.0) or 0.0),
            "depth": depth,
        }
    ]
    for child in sorted(node.get("children", []), key=lambda item: item.get("value_ms", 0.0), reverse=True):
        rows.extend(flatten_profile_tree(child, depth=depth + 1))
    return rows


def render_profile_tree_table(profile_bundle: dict[str, Any] | None) -> str:
    if not profile_bundle:
        return ""
    root = profile_bundle.get("root")
    if not isinstance(root, dict):
        return ""
    rows = flatten_profile_tree(root)
    total_ms = float(root.get("value_ms", 0.0) or 0.0)
    rendered_rows: list[str] = []
    for row in rows:
        ratio_percent = row["value_ms"] / total_ms * 100.0 if total_ms else 0.0
        rendered_rows.append(
            "".join(
                [
                    "<tr>",
                    f"<td class='profile-stage-name' style='padding-left: {12 + row['depth'] * 18}px'>{html.escape(row['name'])}</td>",
                    f"<td>{format_metric(row['value_ms'], ' ms')}</td>",
                    f"<td>{ratio_percent:.1f}%</td>",
                    "</tr>",
                ]
            )
        )
    return "".join(
        [
            "<div class='profile-stage-table-wrap'>",
            "<table class='profile-stage-table'>",
            "<thead><tr><th>阶段</th><th>耗时</th><th>占比</th></tr></thead>",
            f"<tbody>{''.join(rendered_rows)}</tbody>",
            "</table>",
            "</div>",
        ]
    )


def render_profile_notes(profile_bundle: dict[str, Any] | None) -> str:
    if not profile_bundle:
        return ""
    notes = profile_bundle.get("notes", [])
    if not isinstance(notes, list) or not notes:
        return ""
    rendered = "".join(f"<li>{html.escape(str(note))}</li>" for note in notes)
    return f"<ul class='profile-notes'>{rendered}</ul>"


def render_flame_graph_svg(profile_bundle: dict[str, Any] | None) -> str:
    if not profile_bundle:
        return "<p class='empty'>无可用 profile 数据</p>"
    root = profile_bundle.get("root")
    if not isinstance(root, dict):
        return "<p class='empty'>无可用 profile 数据</p>"
    max_depth = profile_depth(root)
    row_height = 34
    chart_width = 1120
    chart_height = max_depth * row_height + 16
    nodes: list[dict[str, Any]] = []
    collect_flame_graph_nodes(root, x=0.0, width=chart_width, depth=0, max_depth=max_depth, row_height=row_height, nodes=nodes, path=())
    total_ms = float(root.get("value_ms", 0.0) or 0.0)
    svg_parts: list[str] = [
        f"<svg class='flame-graph' viewBox='0 0 {chart_width} {chart_height}' role='img' aria-label='板端推理火焰图'>"
    ]
    for node in nodes:
        label = str(node.get("name", "-"))
        value_ms = float(node.get("value_ms", 0.0) or 0.0)
        ratio_percent = value_ms / total_ms * 100.0 if total_ms else 0.0
        x = float(node.get("x", 0.0))
        y = float(node.get("y", 0.0))
        width = float(node.get("width", 0.0))
        rect_width = max(width - 1.0, 0.0)
        title = " › ".join(str(part) for part in node.get("path", (label,)))
        svg_parts.append(
            "".join(
                [
                    f"<g class='flame-node'>",
                    f"<title>{html.escape(title)}: {value_ms:.3f} ms ({ratio_percent:.1f}%)</title>",
                    f"<rect x='{x:.2f}' y='{y:.2f}' width='{rect_width:.2f}' height='28' rx='6' fill='{flame_color(label)}'></rect>",
                    (f"<text x='{x + 8:.2f}' y='{y + 18:.2f}' class='flame-node-label'>{html.escape(label)}</text>" if width >= 110.0 else ""),
                    "</g>",
                ]
            )
        )
    svg_parts.append("</svg>")
    return "".join(svg_parts)


def render_profile_card(title: str, subtitle: str, profile_bundle: dict[str, Any] | None) -> str:
    if profile_bundle is None:
        return ""
    return "".join(
        [
            "<section class='profile-card'>",
            f"<h3>{html.escape(title)}</h3>",
            f"<p class='profile-caption'>{html.escape(subtitle)}</p>",
            render_profile_telemetry(profile_bundle),
            render_profile_stage_breakdown(profile_bundle),
            "<div class='flame-wrap'>",
            render_flame_graph_svg(profile_bundle),
            "</div>",
            render_profile_tree_table(profile_bundle),
            render_profile_notes(profile_bundle),
            "</section>",
        ]
    )


def render_case_rows(case_payloads: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for case in case_payloads:
        status_class = "pass" if case["passed"] else "fail"
        status_text = "通过" if case["passed"] else "失败"
        tags = "<br>".join(html.escape(tag) for tag in case["tags"]) or "-"
        failure_text = "<br>".join(html.escape(reason) for reason in case["failure_reasons"][:3]) or "-"
        preview_iteration = get_case_preview_iteration(case)
        preview_audio = render_audio_player(preview_iteration.get("local_wav_path") if preview_iteration else None, compact=True)
        rows.append(
            "".join(
                [
                    "<tr>",
                    f"<td><span class='status-pill {status_class}'>{status_text}</span></td>",
                    f"<td>{html.escape(case['category'])}</td>",
                    f"<td><strong>{html.escape(case['id'])}</strong><br>{html.escape(case['name'])}</td>",
                    "<td>",
                    f"<div class='case-sentence'>{html.escape(case['sentence'])}</div>",
                    f"<div class='case-audio'>{preview_audio}</div>",
                    "</td>",
                    f"<td>{tags}</td>",
                    f"<td>{case['measured_iterations']}</td>",
                    f"<td>{format_metric(case['p50_latency_ms'], ' ms')}</td>",
                    f"<td>{format_metric(case['p95_latency_ms'], ' ms')}</td>",
                    f"<td>{format_metric(case['mean_rtf'], digits=6)}</td>",
                    f"<td>{case['warning_count']}</td>",
                    f"<td>{failure_text}</td>",
                    "</tr>",
                ]
            )
        )
    return "".join(rows)


def render_iteration_rows(iterations: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for iteration in iterations:
        row_class = "warmup" if iteration["is_warmup"] else ("pass" if iteration["passed"] else "fail")
        artifacts: list[str] = []
        local_log_path = iteration.get("local_log_path")
        if local_log_path:
            artifacts.append(f"<a href='{html.escape(local_log_path)}'>日志</a>")
        local_wav_path = iteration.get("local_wav_path")
        if local_wav_path:
            artifacts.append(f"<a href='{html.escape(local_wav_path)}'>音频</a>")
        failure_text = "<br>".join(html.escape(reason) for reason in iteration["failure_reasons"]) or "-"
        artifact_text = "<div class='artifact-stack'>" + ("<br>".join(artifacts) or "-") + "</div>"
        rows.append(
            "".join(
                [
                    f"<tr class='{row_class}'>",
                    f"<td>{'warmup' if iteration['is_warmup'] else 'run'}</td>",
                    f"<td>{iteration['iteration']}</td>",
                    f"<td>{'通过' if iteration['passed'] else '失败'}</td>",
                    f"<td>{iteration['exit_code']}</td>",
                    f"<td>{format_metric(iteration['latency_ms'], ' ms')}</td>",
                    f"<td>{format_metric(iteration['rtf'], digits=6)}</td>",
                    f"<td>{iteration['warning_count']}</td>",
                    f"<td>{format_metric(iteration['elapsed_wall_ms'], ' ms')}</td>",
                    f"<td>{artifact_text}</td>",
                    f"<td>{failure_text}</td>",
                    "</tr>",
                ]
            )
        )
    return "".join(rows)


def render_case_details(case_payloads: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for case in case_payloads:
        status_class = "pass" if case["passed"] else "fail"
        status_text = "通过" if case["passed"] else "失败"
        preview_iteration = get_case_preview_iteration(case)
        preview_audio = render_audio_player(preview_iteration.get("local_wav_path") if preview_iteration else None, compact=False)
        thresholds = case["thresholds"]
        threshold_parts = [
            f"时延阈值: {format_metric(thresholds.get('latency_threshold_ms'), ' ms')}" if thresholds.get("latency_threshold_ms") is not None else "时延阈值: -",
            f"RTF 阈值: {format_metric(thresholds.get('rtf_threshold'), digits=6)}" if thresholds.get("rtf_threshold") is not None else "RTF 阈值: -",
            f"告警阈值: {thresholds.get('max_warning_count') if thresholds.get('max_warning_count') is not None else '-'}",
        ]
        if thresholds.get("must_contain"):
            threshold_parts.append("必须包含: " + ", ".join(thresholds["must_contain"]))
        if thresholds.get("must_not_contain"):
            threshold_parts.append("禁止包含: " + ", ".join(thresholds["must_not_contain"]))
        profile_iteration = case.get("profile_iteration", {}) if isinstance(case.get("profile_iteration"), dict) else {}
        profile_subtitle = "未生成阶段 profile"
        if case.get("profile"):
            phase_name = "warmup" if profile_iteration.get("is_warmup") else "run"
            profile_subtitle = f"代表轮次: {phase_name} #{int(profile_iteration.get('iteration', 0) or 0)}"
        blocks.append(
            "".join(
                [
                    "<details class='case-detail'>",
                    "<summary>",
                    f"<span class='status-pill {status_class}'>{status_text}</span>",
                    f"<strong>{html.escape(case['id'])}</strong>",
                    f"<span>{html.escape(case['name'])}</span>",
                    f"<span class='detail-meta'>{html.escape(case['category'])}</span>",
                    "</summary>",
                    "<div class='case-body'>",
                    "<div class='preview-grid'>",
                    "<article class='preview-card'>",
                    "<span>输入文字</span>",
                    f"<p>{html.escape(case['sentence'])}</p>",
                    "</article>",
                    "<article class='preview-card'>",
                    "<span>音频预览</span>",
                    preview_audio,
                    (f"<p class='preview-caption'>样本轮次: {'warmup' if preview_iteration and preview_iteration.get('is_warmup') else 'run'} #{preview_iteration.get('iteration')}</p>" if preview_iteration else "<p class='preview-caption'>未下载音频文件</p>"),
                    "</article>",
                    "</div>",
                    f"<p><strong>说明:</strong> {html.escape(case['notes'] or '-')}</p>",
                    f"<p><strong>约束:</strong> {html.escape(' | '.join(threshold_parts))}</p>",
                    render_profile_card("代表轮次火焰图", profile_subtitle, case.get("profile")),
                    "<table class='iteration-table'>",
                    "<thead><tr><th>阶段</th><th>轮次</th><th>结果</th><th>退出码</th><th>时延</th><th>RTF</th><th>告警</th><th>墙钟耗时</th><th>产物</th><th>失败原因</th></tr></thead>",
                    f"<tbody>{render_iteration_rows(case['iterations'])}</tbody>",
                    "</table>",
                    "</div>",
                    "</details>",
                ]
            )
        )
    return "".join(blocks)


def render_html_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    meta = payload["meta"]
    profiles = payload.get("profiles", {}) if isinstance(payload.get("profiles"), dict) else {}
    category_chart_items = [
        (item["category"], item["mean_latency_ms"], item["cases_passed"] == item["cases_total"])
        for item in payload["categories"]
    ]
    slow_case_chart_items = [
        (item["id"], item["p95_latency_ms"], bool(item["passed"]))
        for item in payload["top_slow_cases"]
    ]
    profile_section = ""
    if profiles.get("available"):
        profile_section = render_profile_card(
            "板端推理火焰图",
            f"来源: {profiles.get('source', '-')} | 已剖析有效轮次: {int(profiles.get('measured_iterations_profiled', 0) or 0)} | 覆盖用例: {int(profiles.get('cases_with_profiles', 0) or 0)}",
            profiles.get("aggregate_profile"),
        )
    elif profiles.get("notes"):
        profile_section = "".join(
            [
                "<section class='profile-card'>",
                "<h3>板端推理火焰图</h3>",
                "<p class='profile-caption'>当前报告没有足够的 profile 数据。</p>",
                render_profile_notes(profiles),
                "</section>",
            ]
        )
    return "".join(
        [
            "<!DOCTYPE html>",
            "<html lang='zh-CN'>",
            "<head>",
            "<meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            f"<title>{html.escape(meta['suite_name'])} 测试报告</title>",
            "<style>",
            ":root { --bg: #f6f1e8; --card: #fffaf2; --ink: #1d2a36; --muted: #6c7680; --line: #d8cdbb; --ok: #2e8b57; --bad: #c14f3f; --warm: #d1862b; }",
            "* { box-sizing: border-box; }",
            "body { margin: 0; font-family: Bahnschrift, 'Segoe UI', sans-serif; color: var(--ink); background: radial-gradient(circle at top left, #fff8eb 0%, var(--bg) 42%, #efe3d1 100%); }",
            ".page { max-width: 1280px; margin: 0 auto; padding: 28px; }",
            ".hero { background: linear-gradient(135deg, rgba(244, 225, 192, 0.88), rgba(255, 250, 242, 0.96)); border: 1px solid rgba(121, 100, 72, 0.18); border-radius: 28px; padding: 28px; box-shadow: 0 18px 38px rgba(68, 48, 24, 0.08); }",
            ".hero h1 { margin: 0 0 8px; font-size: 34px; letter-spacing: 0.02em; }",
            ".hero p { margin: 0; color: var(--muted); line-height: 1.6; }",
            ".meta-grid, .summary-grid, .chart-grid { display: grid; gap: 16px; margin-top: 18px; }",
            ".meta-grid { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }",
            ".summary-grid { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }",
            ".chart-grid { grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); align-items: start; }",
            ".meta-card, .summary-card, .chart-card, .table-card, .detail-card, .case-detail { background: var(--card); border: 1px solid rgba(121, 100, 72, 0.18); border-radius: 22px; box-shadow: 0 10px 26px rgba(68, 48, 24, 0.06); }",
            ".meta-card, .summary-card, .detail-card { padding: 18px 20px; }",
            ".summary-card strong { display: block; font-size: 30px; line-height: 1.1; }",
            ".summary-card span, .meta-card span, .detail-card span { color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }",
            ".meta-card p, .detail-card p { margin: 10px 0 0; font-size: 15px; line-height: 1.5; }",
            ".chart-card { padding: 18px 20px; overflow: auto; }",
            ".chart-card h3 { margin: 0 0 12px; font-size: 18px; }",
            ".chart-label { font-size: 12px; fill: #435162; }",
            ".chart-value { font-size: 12px; fill: #435162; }",
            ".empty { color: var(--muted); margin: 0; }",
            ".table-card { margin-top: 18px; overflow: hidden; }",
            ".table-card h2, .section-title { margin: 0; padding: 20px 24px 0; font-size: 22px; }",
            ".table-card p.section-copy, .section-copy { margin: 6px 0 0; padding: 0 24px; color: var(--muted); }",
            "table { width: 100%; border-collapse: collapse; }",
            "thead th { text-align: left; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); background: rgba(244, 225, 192, 0.45); }",
            "th, td { padding: 14px 16px; border-top: 1px solid rgba(121, 100, 72, 0.12); vertical-align: top; }",
            "tbody tr:hover { background: rgba(244, 225, 192, 0.18); }",
            ".status-pill { display: inline-flex; align-items: center; justify-content: center; min-width: 56px; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }",
            ".status-pill.pass { background: rgba(46, 139, 87, 0.14); color: var(--ok); }",
            ".status-pill.fail { background: rgba(193, 79, 63, 0.14); color: var(--bad); }",
            ".detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-top: 18px; }",
            ".case-section { margin-top: 18px; }",
            ".case-detail { margin-top: 14px; overflow: hidden; }",
            ".case-detail summary { list-style: none; cursor: pointer; padding: 18px 20px; display: grid; grid-template-columns: auto minmax(160px, 220px) 1fr auto; gap: 12px; align-items: center; }",
            ".case-detail summary::-webkit-details-marker { display: none; }",
            ".case-detail .detail-meta { color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }",
            ".case-body { padding: 0 20px 20px; }",
            ".case-body p { line-height: 1.6; }",
            ".case-sentence { color: var(--ink); line-height: 1.5; margin-bottom: 8px; }",
            ".case-audio { min-width: 220px; }",
            ".preview-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 14px; }",
            ".preview-card { border: 1px solid rgba(121, 100, 72, 0.14); border-radius: 18px; background: rgba(244, 225, 192, 0.18); padding: 14px 16px; }",
            ".preview-card span { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }",
            ".preview-card p { margin: 10px 0 0; }",
            ".preview-caption { color: var(--muted); font-size: 13px; }",
            ".profile-card { margin-top: 18px; padding: 18px 20px; background: var(--card); border: 1px solid rgba(121, 100, 72, 0.18); border-radius: 22px; box-shadow: 0 10px 26px rgba(68, 48, 24, 0.06); }",
            ".profile-card h3 { margin: 0; font-size: 18px; }",
            ".profile-caption { margin: 6px 0 0; color: var(--muted); }",
            ".profile-chip-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-top: 14px; }",
            ".profile-chip { padding: 12px 14px; border-radius: 16px; border: 1px solid rgba(121, 100, 72, 0.14); background: rgba(244, 225, 192, 0.18); }",
            ".profile-chip strong { display: block; font-size: 13px; }",
            ".profile-chip span { color: var(--muted); font-size: 13px; }",
            ".flame-wrap { margin-top: 14px; overflow-x: auto; border-radius: 18px; border: 1px solid rgba(121, 100, 72, 0.14); background: linear-gradient(180deg, rgba(248, 241, 230, 0.96), rgba(255, 250, 242, 0.98)); padding: 10px; }",
            ".flame-graph { min-width: 1120px; width: 100%; height: auto; }",
            ".flame-node-label { fill: #1d2a36; font-size: 12px; font-weight: 600; pointer-events: none; }",
            ".profile-stage-table-wrap { margin-top: 14px; overflow-x: auto; }",
            ".profile-stage-table { width: 100%; border-collapse: collapse; font-size: 14px; }",
            ".profile-stage-table th, .profile-stage-table td { padding: 10px 12px; border-top: 1px solid rgba(121, 100, 72, 0.12); }",
            ".profile-stage-table thead th { background: rgba(244, 225, 192, 0.35); color: var(--muted); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; }",
            ".profile-stage-name { font-family: Consolas, 'Courier New', monospace; white-space: nowrap; }",
            ".profile-notes { margin: 12px 0 0; padding-left: 18px; color: var(--muted); }",
            ".profile-notes li { margin-top: 4px; }",
            ".audio-player { width: 100%; margin-top: 10px; }",
            ".audio-player.compact { max-width: 260px; width: 100%; margin-top: 0; }",
            ".muted-inline { color: var(--muted); font-size: 13px; }",
            ".artifact-stack { display: flex; flex-direction: column; gap: 4px; }",
            ".iteration-table { margin-top: 12px; font-size: 14px; }",
            ".iteration-table tr.pass { background: rgba(46, 139, 87, 0.06); }",
            ".iteration-table tr.fail { background: rgba(193, 79, 63, 0.06); }",
            ".iteration-table tr.warmup { background: rgba(209, 134, 43, 0.08); }",
            "a { color: #1d5b86; text-decoration: none; }",
            "a:hover { text-decoration: underline; }",
            "@media (max-width: 880px) { .page { padding: 16px; } .hero { padding: 22px; } .case-detail summary { grid-template-columns: 1fr; } table { display: block; overflow-x: auto; } }",
            "</style>",
            "</head>",
            "<body>",
            "<main class='page'>",
            "<section class='hero'>",
            f"<h1>{html.escape(meta['suite_name'])}</h1>",
            f"<p>{html.escape(meta['description'] or 'RK3588 板端离线 TTS 批量测试报告')}</p>",
            "<div class='meta-grid'>",
            f"<article class='meta-card'><span>计划文件</span><p>{html.escape(meta['plan_path'])}</p></article>",
            f"<article class='meta-card'><span>生成时间</span><p>{html.escape(meta['generated_at_utc'])}</p></article>",
            f"<article class='meta-card'><span>板卡主机</span><p>{html.escape(meta['host'])}</p></article>",
            f"<article class='meta-card'><span>板端目录</span><p>{html.escape(meta['remote_dir'])}</p></article>",
            "</div>",
            "<div class='summary-grid'>",
            f"<article class='summary-card'><span>用例通过率</span><strong>{summary['case_pass_rate_percent']:.2f}%</strong></article>",
            f"<article class='summary-card'><span>有效轮次</span><strong>{summary['measured_iterations']}</strong></article>",
            f"<article class='summary-card'><span>P95 时延</span><strong>{format_metric(summary['p95_latency_ms'], ' ms')}</strong></article>",
            f"<article class='summary-card'><span>平均时延</span><strong>{format_metric(summary['mean_latency_ms'], ' ms')}</strong></article>",
            f"<article class='summary-card'><span>平均 RTF</span><strong>{format_metric(summary['mean_rtf'], digits=6)}</strong></article>",
            f"<article class='summary-card'><span>Warmup 轮次</span><strong>{summary['warmup_iterations']}</strong></article>",
            "</div>",
            "</section>",
            "<section class='chart-grid'>",
            render_bar_chart("分类平均时延", category_chart_items, " ms"),
            render_bar_chart("最慢用例 P95 时延", slow_case_chart_items, " ms"),
            "</section>",
            profile_section,
            "<section class='table-card'>",
            "<h2>用例总览</h2>",
            "<p class='section-copy'>同一用例的 warmup 轮次不计入性能统计，但仍会参与通过性判定。</p>",
            "<table>",
            "<thead><tr><th>结果</th><th>分类</th><th>用例</th><th>输入文字 / 音频</th><th>标签</th><th>有效轮次</th><th>P50</th><th>P95</th><th>平均 RTF</th><th>告警</th><th>失败原因</th></tr></thead>",
            f"<tbody>{render_case_rows(payload['cases'])}</tbody>",
            "</table>",
            "</section>",
            "<section class='detail-grid'>",
            f"<article class='detail-card'><span>失败用例</span><p>{' / '.join(html.escape(item['id']) for item in payload['failing_cases']) if payload['failing_cases'] else '无'}</p></article>",
            f"<article class='detail-card'><span>筛选条件</span><p>{html.escape(json.dumps(meta['filters'], ensure_ascii=False))}</p></article>",
            f"<article class='detail-card'><span>报告目录</span><p>{html.escape(meta['report_dir'])}</p></article>",
            f"<article class='detail-card'><span>源地址</span><p>{html.escape(meta['source_ip'] or '-')}</p></article>",
            "</section>",
            "<section class='case-section'>",
            "<h2 class='section-title'>用例详情</h2>",
            "<p class='section-copy'>点击展开可以查看每轮执行明细、失败原因和本地日志链接。</p>",
            render_case_details(payload["cases"]),
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def render_summary_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    meta = payload["meta"]
    profiles = payload.get("profiles", {}) if isinstance(payload.get("profiles"), dict) else {}
    lines = [
        f"# {meta['suite_name']} 测试报告",
        "",
        f"- 计划文件: {meta['plan_path']}",
        f"- 生成时间(UTC): {meta['generated_at_utc']}",
        f"- 板卡主机: {meta['host']}",
        f"- 板端目录: {meta['remote_dir']}",
        f"- 报告目录: {meta['report_dir']}",
        "",
        "## 总览",
        "",
        f"- 用例总数: {summary['cases_total']}",
        f"- 用例通过数: {summary['cases_passed']}",
        f"- 用例通过率: {summary['case_pass_rate_percent']:.2f}%",
        f"- 有效轮次: {summary['measured_iterations']}",
        f"- Warmup 轮次: {summary['warmup_iterations']}",
        f"- 平均时延: {format_metric(summary['mean_latency_ms'], ' ms')}",
        f"- P95 时延: {format_metric(summary['p95_latency_ms'], ' ms')}",
        f"- 最大时延: {format_metric(summary['max_latency_ms'], ' ms')}",
        f"- 平均 RTF: {format_metric(summary['mean_rtf'], digits=6)}",
        "",
        "## 板端 Profile",
        "",
        f"- 火焰图可用: {'是' if profiles.get('available') else '否'}",
        f"- Profile 来源: {profiles.get('source', '-')}",
        f"- 已剖析有效轮次: {int(profiles.get('measured_iterations_profiled', 0) or 0)}",
        "",
        "## 失败用例",
        "",
    ]
    if payload["failing_cases"]:
        for item in payload["failing_cases"]:
            reasons = "；".join(item["failure_reasons"]) or "未提供"
            lines.append(f"- {item['id']} ({item['category']}): {reasons}")
    else:
        lines.append("- 无")
    lines.extend(["", "## 最慢用例", "", "| 用例 | 分类 | P95 时延(ms) | 平均 RTF | 结果 |", "| --- | --- | ---: | ---: | --- |"])
    for item in payload["top_slow_cases"]:
        lines.append(
            f"| {item['id']} | {item['category']} | {format_metric(item['p95_latency_ms'])} | {format_metric(item['mean_rtf'], digits=6)} | {'通过' if item['passed'] else '失败'} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(report_dir: Path, payload: dict[str, Any]) -> None:
    csv_path = report_dir / "results.csv"
    ensure_parent(csv_path)
    fieldnames = [
        "case_id",
        "case_name",
        "category",
        "iteration",
        "is_warmup",
        "passed",
        "exit_code",
        "latency_ms",
        "rtf",
        "wav_size_bytes",
        "wav_duration_ms",
        "warning_count",
        "started_at_utc",
        "finished_at_utc",
        "elapsed_wall_ms",
        "local_log_path",
        "local_wav_path",
        "failure_reasons",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in payload["iterations"]:
            writer.writerow(
                {
                    "case_id": item["case_id"],
                    "case_name": item["case_name"],
                    "category": item["category"],
                    "iteration": item["iteration"],
                    "is_warmup": item["is_warmup"],
                    "passed": item["passed"],
                    "exit_code": item["exit_code"],
                    "latency_ms": item["latency_ms"],
                    "rtf": item["rtf"],
                    "wav_size_bytes": item["wav_size_bytes"],
                    "wav_duration_ms": item["wav_duration_ms"],
                    "warning_count": item["warning_count"],
                    "started_at_utc": item["started_at_utc"],
                    "finished_at_utc": item["finished_at_utc"],
                    "elapsed_wall_ms": item["elapsed_wall_ms"],
                    "local_log_path": item["local_log_path"],
                    "local_wav_path": item["local_wav_path"],
                    "failure_reasons": " | ".join(item["failure_reasons"]),
                }
            )


def write_report_bundle(report_dir: Path, payload: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "results.json", payload)
    write_csv(report_dir, payload)
    write_text(report_dir / "summary.md", render_summary_markdown(payload))
    write_text(report_dir / "report.html", render_html_report(payload))


def resolve_run_filters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "case_ids": args.case_id or [],
        "categories": args.category or [],
        "tags": args.tag or [],
        "download_wavs": args.download_wavs,
        "deep_profile": args.deep_profile,
        "profile_sample_interval_ms": args.profile_sample_interval_ms,
        "stop_on_failure": args.stop_on_failure,
        "max_cases": args.max_cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run scalable RK3588 TTS test plans and generate HTML reports")
    subparsers = parser.add_subparsers(dest="action", required=True)

    run_parser = subparsers.add_parser("run", help="Run a test plan on the board and generate reports")
    run_parser.add_argument("--plan", type=Path, default=None, help="Test plan JSON path")
    run_parser.add_argument("--report-dir", type=Path, default=None, help="Output directory for this test run")
    run_parser.add_argument("--host", default=None, help="Board SSH host")
    run_parser.add_argument("--username", default=None, help="Board SSH username")
    run_parser.add_argument("--password", default=None, help="Board SSH password")
    run_parser.add_argument("--source-ip", default=None, help="Optional local source IP used to reach the board")
    run_parser.add_argument("--remote-dir", default=None, help="Remote runtime directory on the board")
    run_parser.add_argument("--ssh-timeout", type=int, default=None, help="SSH connect timeout in seconds")
    run_parser.add_argument("--remote-timeout", type=int, default=None, help="Remote command timeout in seconds")
    run_parser.add_argument("--case-id", action="append", default=[], help="Run only the specified case id (repeatable)")
    run_parser.add_argument("--category", action="append", default=[], help="Run only the specified category (repeatable)")
    run_parser.add_argument("--tag", action="append", default=[], help="Run only cases containing any of the specified tags (repeatable)")
    run_parser.add_argument("--max-cases", type=int, default=None, help="Run only the first N selected cases")
    run_parser.add_argument("--download-wavs", action="store_true", help="Download every generated WAV into the report directory")
    run_parser.add_argument("--deep-profile", action="store_true", help="Use the uploaded board profiler wrapper and collect process/NPU samples")
    run_parser.add_argument("--profile-sample-interval-ms", type=int, default=20, help="Sampling interval in milliseconds for board-side deep profile")
    run_parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failed case")
    run_parser.add_argument("--dry-run", action="store_true", help="Print selected cases and exit without connecting to the board")

    report_parser = subparsers.add_parser("report", help="Rebuild HTML, CSV, and Markdown reports from an existing results.json")
    report_parser.add_argument("--results-json", type=Path, required=True, help="Existing results.json path")
    report_parser.add_argument("--report-dir", type=Path, default=None, help="Override output directory; defaults to the JSON parent directory")

    return parser


def run_plan(args: argparse.Namespace) -> None:
    plan_path = resolve_plan_path(args.plan)
    metadata, cases = load_test_plan(plan_path)
    selected_cases = select_cases(
        cases,
        case_ids={item.strip() for item in args.case_id if item.strip()},
        categories={item.strip().lower() for item in args.category if item.strip()},
        tags={item.strip().lower() for item in args.tag if item.strip()},
    )
    if args.max_cases is not None:
        selected_cases = selected_cases[: args.max_cases]
    if not selected_cases:
        fail("筛选后没有可执行的测试用例")

    if args.dry_run:
        print_selected_cases(metadata, selected_cases)
        return

    local_settings = load_local_settings()
    host = resolve_required_text_option(
        args.host,
        env_names=("TTS_BOARD_HOST",),
        local_settings=local_settings,
        option_name="board host",
    )
    username = resolve_required_text_option(
        args.username,
        env_names=("TTS_BOARD_USERNAME",),
        local_settings=local_settings,
        option_name="board username",
    )
    password = resolve_required_text_option(
        args.password,
        env_names=("TTS_BOARD_PASSWORD",),
        local_settings=local_settings,
        option_name="board password",
    )
    remote_dir = resolve_text_option(
        args.remote_dir,
        env_names=("TTS_REMOTE_DIR",),
        local_settings=local_settings,
        default=DEFAULT_REMOTE_DIR,
    )
    ssh_timeout = resolve_int_option(
        args.ssh_timeout,
        env_names=("TTS_SSH_TIMEOUT",),
        local_settings=local_settings,
        default=8,
    )
    remote_timeout = resolve_int_option(
        args.remote_timeout,
        env_names=("TTS_REMOTE_TIMEOUT",),
        local_settings=local_settings,
        default=1800,
    )
    source_ip_value = resolve_text_option(
        args.source_ip,
        env_names=("TTS_SOURCE_IP",),
        local_settings=local_settings,
        default="",
    )
    source_ip = source_ip_value.strip() or None
    if source_ip is None:
        source_ip = guess_source_ip(host)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suite_key = safe_path_fragment(metadata["name"]) or "tts-suite"
    report_dir = args.report_dir.resolve() if args.report_dir else (DEFAULT_REPORT_ROOT / f"{suite_key}-{timestamp}").resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    remote_run_root = f"{remote_dir.rstrip('/')}/output/test-runs/{suite_key}-{timestamp}"
    log(f"Executing {len(selected_cases)} cases on {host}; report will be written to {report_dir}")

    results: list[IterationResult] = []
    client = open_ssh_client(host, username, password, source_ip=source_ip, timeout=ssh_timeout)
    try:
        exit_code, mkdir_output = run_remote_capture(client, f"mkdir -p {sh_quote(remote_run_root)}", timeout=remote_timeout)
        if exit_code != 0:
            fail(f"创建远端测试目录失败: {mkdir_output or exit_code}")
        for index, case in enumerate(selected_cases, start=1):
            log(f"[{index}/{len(selected_cases)}] {case.id} ({case.category})")
            case_results = execute_case(
                client,
                case=case,
                case_index=index,
                report_dir=report_dir,
                remote_dir=remote_dir,
                remote_run_root=remote_run_root,
                remote_timeout=remote_timeout,
                download_wavs=args.download_wavs,
                deep_profile=args.deep_profile,
                profile_sample_interval_ms=args.profile_sample_interval_ms,
            )
            results.extend(case_results)
            case_failed = any(not item.passed for item in case_results)
            if case_failed and args.stop_on_failure:
                log(f"Stopped on failure at case {case.id}")
                break
    finally:
        client.close()

    payload = build_report_payload(
        metadata=metadata,
        cases=selected_cases,
        report_dir=report_dir,
        host=host,
        remote_dir=remote_dir,
        source_ip=source_ip,
        results=results,
        filters=resolve_run_filters(args),
    )
    write_report_bundle(report_dir, payload)
    summary = payload["summary"]
    log(
        f"Suite complete: cases={summary['cases_total']}, passed={summary['cases_passed']}, "
        f"p95={format_metric(summary['p95_latency_ms'], ' ms')}"
    )
    log(f"HTML report: {report_dir / 'report.html'}")


def rebuild_report(args: argparse.Namespace) -> None:
    results_json = args.results_json.resolve()
    if not results_json.exists():
        fail(f"结果文件不存在: {results_json}")
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    report_dir = args.report_dir.resolve() if args.report_dir else results_json.parent.resolve()
    meta = payload.setdefault("meta", {})
    meta["report_dir"] = path_from_workspace(report_dir)
    meta["generated_at_utc"] = utc_now_iso()
    augment_payload_with_profiles(payload, report_dir)
    write_report_bundle(report_dir, payload)
    log(f"Rebuilt report bundle at {report_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.action == "run":
        run_plan(args)
        return
    if args.action == "report":
        rebuild_report(args)
        return
    fail(f"Unsupported action: {args.action}")


if __name__ == "__main__":
    main()