from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


TOTAL_OPERATOR_ELAPSED_PATTERN = re.compile(r"Total Operator Elapsed(?: Per Frame)? Time\(us\):\s*([0-9.]+)")


class Toolkit2ProfileError(Exception):
    pass


@dataclass(frozen=True)
class Toolkit2ProfileSummary:
    generated_at_utc: str
    model_path: str
    output_dir: str
    target: str
    device_id: str
    adb_connect: str
    adb_serial: str
    sdk_version_path: str | None
    eval_perf_path: str | None
    perf_run_path: str | None
    eval_memory_path: str | None
    adb_devices_path: str | None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def extract_total_operator_elapsed_time_us(text: str) -> float | None:
    match = TOTAL_OPERATOR_ELAPSED_PATTERN.search(text)
    if not match:
        return None
    return float(match.group(1))


def normalize_sdk_version(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    normalized: dict[str, Any] = {}
    for attribute in ("api_version", "drv_version"):
        attribute_value = getattr(value, attribute, None)
        if attribute_value not in {None, ""}:
            normalized[attribute] = attribute_value
    if normalized:
        return normalized
    if isinstance(value, (list, tuple)):
        return {"values": [str(item) for item in value]}
    return {"raw": str(value)}


def resolve_rknn() -> Any:
    try:
        from rknn.api import RKNN  # type: ignore
    except ImportError as exc:
        raise Toolkit2ProfileError(
            "rknn.api is unavailable. Run this script inside the Toolkit2 Docker image or install rknn-toolkit2 first."
        ) from exc
    return RKNN


def run_command(command: Sequence[str], *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0 and not allow_failure:
        command_text = " ".join(command)
        message = completed.stdout + completed.stderr
        raise Toolkit2ProfileError(f"Command failed ({completed.returncode}): {command_text}\n{message}")
    return completed


def ensure_model_exists(model_path: Path) -> Path:
    resolved = model_path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise Toolkit2ProfileError(f"RKNN model does not exist: {resolved}")
    return resolved


def build_runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "target": args.target,
        "perf_debug": True,
        "eval_mem": True,
    }
    device_id = args.device_id or args.adb_serial
    if device_id:
        kwargs["device_id"] = device_id
    return kwargs


def create_rknn_context(args: argparse.Namespace) -> Any:
    RKNN = resolve_rknn()
    rknn = RKNN(verbose=args.verbose)
    model_path = ensure_model_exists(args.model)

    load_result = rknn.load_rknn(str(model_path))
    if load_result != 0:
        raise Toolkit2ProfileError(f"rknn.load_rknn failed: {load_result}")

    init_result = rknn.init_runtime(**build_runtime_kwargs(args))
    if init_result != 0:
        raise Toolkit2ProfileError(f"rknn.init_runtime failed: {init_result}")
    return rknn


def action_eval_perf(args: argparse.Namespace) -> int:
    rknn = create_rknn_context(args)
    try:
        rknn.eval_perf()
    finally:
        rknn.release()
    return 0


def action_eval_memory(args: argparse.Namespace) -> int:
    rknn = create_rknn_context(args)
    try:
        rknn.eval_memory()
    finally:
        rknn.release()
    return 0


def capture_stage_output(args: argparse.Namespace, *, action: str, destination: Path) -> str:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--action",
        action,
        "--model",
        str(args.model),
        "--output-dir",
        str(args.output_dir),
        "--target",
        args.target,
    ]
    if args.device_id:
        command.extend(["--device-id", args.device_id])
    if args.adb_serial:
        command.extend(["--adb-serial", args.adb_serial])
    if args.verbose:
        command.append("--verbose")

    completed = run_command(command, allow_failure=True)
    content = completed.stdout + completed.stderr
    write_text(destination, content)
    if completed.returncode != 0:
        raise Toolkit2ProfileError(f"Toolkit2 stage {action} failed; see {destination}")
    return content


def collect_adb_devices(*, output_dir: Path, adb_connect: str, adb_serial: str) -> str | None:
    run_command(["adb", "start-server"], allow_failure=True)
    output_parts: list[str] = []

    if adb_connect:
        connect_result = run_command(["adb", "connect", adb_connect], allow_failure=True)
        output_parts.append(connect_result.stdout + connect_result.stderr)

    devices_command = ["adb", "devices", "-l"]
    if adb_serial:
        devices_command = ["adb", "-s", adb_serial, "get-state"]
    devices_result = run_command(devices_command, allow_failure=True)
    output_parts.append(devices_result.stdout + devices_result.stderr)

    content = "".join(output_parts).strip()
    if not content:
        return None

    adb_devices_path = output_dir / "rknn_adb_devices.txt"
    write_text(adb_devices_path, content + "\n")
    return adb_devices_path.name


def collect_profile(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adb_devices_asset = collect_adb_devices(output_dir=output_dir, adb_connect=args.adb_connect, adb_serial=args.adb_serial)

    rknn = create_rknn_context(args)
    try:
        sdk_version = normalize_sdk_version(rknn.get_sdk_version())
    finally:
        rknn.release()

    sdk_version_path = output_dir / "rknn_sdk_version.json"
    write_json(
        sdk_version_path,
        {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "target": args.target,
            "device_id": args.device_id or args.adb_serial,
            "sdk_version": sdk_version,
        },
    )

    eval_perf_path = output_dir / "rknn_eval_perf.txt"
    eval_perf_output = capture_stage_output(args, action="eval-perf", destination=eval_perf_path)
    total_operator_elapsed_time_us = extract_total_operator_elapsed_time_us(eval_perf_output)

    perf_run_path: Path | None = None
    if total_operator_elapsed_time_us is not None:
        perf_run_path = output_dir / "rknn_perf_run.json"
        write_json(
            perf_run_path,
            {
                "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": "eval_perf_total_operator_elapsed_time_us",
                "run_duration_us": total_operator_elapsed_time_us,
            },
        )

    eval_memory_path = output_dir / "rknn_memory_profile.txt"
    capture_stage_output(args, action="eval-memory", destination=eval_memory_path)

    summary = Toolkit2ProfileSummary(
        generated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        model_path=str(ensure_model_exists(args.model)),
        output_dir=str(output_dir),
        target=args.target,
        device_id=args.device_id,
        adb_connect=args.adb_connect,
        adb_serial=args.adb_serial,
        sdk_version_path=sdk_version_path.name,
        eval_perf_path=eval_perf_path.name,
        perf_run_path=perf_run_path.name if perf_run_path is not None else None,
        eval_memory_path=eval_memory_path.name,
        adb_devices_path=adb_devices_asset,
    )
    write_json(output_dir / "rknn_toolkit2_profile_manifest.json", asdict(summary))
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect RKNN Toolkit2 perf and memory profiling artifacts")
    parser.add_argument("--action", choices=("collect", "eval-perf", "eval-memory"), default="collect")
    parser.add_argument("--model", type=Path, required=True, help="Path to the RKNN model file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory used to write profiling artifacts")
    parser.add_argument("--target", default="rk3588", help="Target platform passed to init_runtime")
    parser.add_argument("--device-id", default="", help="Explicit RKNN device id passed to init_runtime")
    parser.add_argument("--adb-connect", default="", help="Optional adb connect target such as 192.168.1.10:5555")
    parser.add_argument("--adb-serial", default="", help="Optional adb serial used for init_runtime device_id fallback")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose RKNN logging")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.action == "collect":
            return collect_profile(args)
        if args.action == "eval-perf":
            return action_eval_perf(args)
        if args.action == "eval-memory":
            return action_eval_memory(args)
        raise Toolkit2ProfileError(f"Unsupported action: {args.action}")
    except Toolkit2ProfileError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())