from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

try:
    from scripts.delivery.sherpa_onnx_rk3588.config import DEFAULT_RUNTIME_DIR, load_local_settings, resolve_path_option, resolve_text_option
except ImportError:
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))
    from scripts.delivery.sherpa_onnx_rk3588.config import DEFAULT_RUNTIME_DIR, load_local_settings, resolve_path_option, resolve_text_option


DEFAULT_IMAGE_TAG = "rkvoice/rknn-toolkit2-profile:2.3.2-py312"
DEFAULT_CONTAINER_WORKSPACE = Path("/workspace")
DEFAULT_DOCKERFILE = WORKSPACE_ROOT / "docker" / "toolkit2-profile" / "Dockerfile"
DEFAULT_BUILD_CONTEXT = DEFAULT_DOCKERFILE.parent
DEFAULT_BOARD_PREPARE_SCRIPT = WORKSPACE_ROOT / "scripts" / "board" / "prepare_rknn_debug_bridge.py"
DEFAULT_MODEL_RELATIVE_PATH = Path("models") / "asr" / "rknn" / "sense-voice-rk3588-20s" / "model.rknn"
DEFAULT_OUTPUT_RELATIVE_PATH = Path("output")


class DockerToolkit2Error(Exception):
    pass


def format_docker_mount_source(path: Path) -> str:
    resolved = path.resolve()
    if os.name == "nt":
        return resolved.as_posix()
    return str(resolved)


def split_mount_anchor(path: Path, *, treat_as_file: bool) -> tuple[Path, Path]:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        if resolved.is_file() or treat_as_file:
            return resolved.parent, Path(resolved.name)
        return resolved, Path()

    suffix_parts: list[str] = []
    anchor = resolved
    while not anchor.exists():
        if anchor.parent == anchor:
            raise DockerToolkit2Error(f"无法为容器映射路径：{resolved}")
        suffix_parts.append(anchor.name)
        anchor = anchor.parent
    suffix_parts.reverse()
    return anchor, Path(*suffix_parts)


def map_host_path_to_container(
    *,
    host_path: Path,
    workspace_root: Path,
    treat_as_file: bool,
    mount_index: int,
) -> tuple[list[str], str]:
    resolved = host_path.expanduser().resolve()
    try:
        relative = resolved.relative_to(workspace_root)
    except ValueError:
        anchor, suffix = split_mount_anchor(resolved, treat_as_file=treat_as_file)
        container_root = Path("/mnt/external") / str(mount_index)
        container_path = (container_root / suffix).as_posix()
        return ["-v", f"{format_docker_mount_source(anchor)}:{container_root.as_posix()}"], container_path

    return [], (DEFAULT_CONTAINER_WORKSPACE / relative).as_posix()


def build_toolkit2_args(
    *,
    model_path: str,
    output_dir: str,
    target: str,
    device_id: str,
    adb_connect: str,
    adb_serial: str,
    verbose: bool,
) -> list[str]:
    args = ["--model", model_path, "--output-dir", output_dir, "--target", target]
    if device_id:
        args.extend(["--device-id", device_id])
    if adb_connect:
        args.extend(["--adb-connect", adb_connect])
    if adb_serial:
        args.extend(["--adb-serial", adb_serial])
    if verbose:
        args.append("--verbose")
    return args


def build_docker_build_command(*, image_tag: str) -> list[str]:
    return [
        "docker",
        "build",
        "-t",
        image_tag,
        "-f",
        str(DEFAULT_DOCKERFILE),
        str(DEFAULT_BUILD_CONTEXT),
    ]


def build_prepare_board_command() -> list[str]:
    return [sys.executable, str(DEFAULT_BOARD_PREPARE_SCRIPT)]


def build_docker_run_command(
    *,
    workspace_root: Path,
    image_tag: str,
    model_path: Path,
    output_dir: Path,
    target: str,
    device_id: str,
    adb_connect: str,
    adb_serial: str,
    verbose: bool,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{format_docker_mount_source(workspace_root)}:{DEFAULT_CONTAINER_WORKSPACE.as_posix()}",
        "--workdir",
        DEFAULT_CONTAINER_WORKSPACE.as_posix(),
    ]

    next_mount_index = 0
    model_mount_args, mapped_model_path = map_host_path_to_container(
        host_path=model_path,
        workspace_root=workspace_root,
        treat_as_file=True,
        mount_index=next_mount_index,
    )
    command.extend(model_mount_args)
    if model_mount_args:
        next_mount_index += 1

    output_mount_args, mapped_output_dir = map_host_path_to_container(
        host_path=output_dir,
        workspace_root=workspace_root,
        treat_as_file=False,
        mount_index=next_mount_index,
    )
    command.extend(output_mount_args)
    if output_mount_args:
        next_mount_index += 1

    command.append(image_tag)
    command.extend(
        build_toolkit2_args(
            model_path=mapped_model_path,
            output_dir=mapped_output_dir,
            target=target,
            device_id=device_id,
            adb_connect=adb_connect,
            adb_serial=adb_serial,
            verbose=verbose,
        )
    )
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    local_settings = load_local_settings()
    parser = argparse.ArgumentParser(description="Run RKNN Toolkit2 profiling inside Docker and emit report-friendly artifacts")
    parser.add_argument("--workspace-dir", type=Path, default=WORKSPACE_ROOT, help="Workspace root mounted into the container")
    parser.add_argument("--image-tag", default=DEFAULT_IMAGE_TAG, help="Docker image tag used for Toolkit2 profiling")
    parser.add_argument("--skip-image-build", action="store_true", help="Reuse an existing image and skip docker build")
    parser.add_argument("--runtime-dir", type=Path, default=None, help="Runtime bundle directory used to resolve default model/output paths")
    parser.add_argument("--model", type=Path, default=None, help="RKNN model path on the host")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for Toolkit2 profiling artifacts")
    parser.add_argument("--target", default=None, help="Target platform passed to Toolkit2 init_runtime")
    parser.add_argument("--device-id", default=None, help="Explicit RKNN device id passed to init_runtime")
    parser.add_argument("--adb-connect", default=None, help="Optional adb connect target such as 192.168.1.10:5555")
    parser.add_argument("--adb-serial", default=None, help="Optional adb serial used for init_runtime device_id fallback")
    parser.add_argument("--prepare-board-debug-bridge", action="store_true", help="Repair adbd 5037 listener and restart rknn_server over SSH before starting Docker profiling")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose RKNN logging")
    parsed = parser.parse_args(argv)

    runtime_dir = resolve_path_option(
        parsed.runtime_dir,
        env_names=("RKVOICE_RUNTIME_DIR", "TTS_RUNTIME_DIR"),
        local_settings=local_settings,
        default=DEFAULT_RUNTIME_DIR,
    )
    parsed.runtime_dir = runtime_dir
    parsed.model = (parsed.model or (runtime_dir / DEFAULT_MODEL_RELATIVE_PATH)).resolve()
    parsed.output_dir = (parsed.output_dir or (runtime_dir / DEFAULT_OUTPUT_RELATIVE_PATH)).resolve()
    parsed.target = resolve_text_option(parsed.target, env_names=("RKVOICE_TOOLKIT2_TARGET",), local_settings=local_settings, default="rk3588") or "rk3588"
    parsed.device_id = resolve_text_option(parsed.device_id, env_names=("RKVOICE_TOOLKIT2_DEVICE_ID",), local_settings=local_settings, default="") or ""
    parsed.adb_connect = resolve_text_option(parsed.adb_connect, env_names=("RKVOICE_TOOLKIT2_ADB_CONNECT",), local_settings=local_settings, default="") or ""
    parsed.adb_serial = resolve_text_option(parsed.adb_serial, env_names=("RKVOICE_TOOLKIT2_ADB_SERIAL",), local_settings=local_settings, default="") or ""
    return parsed


def run_command(command: Sequence[str]) -> int:
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    workspace_root = args.workspace_dir.resolve()
    if not workspace_root.exists():
        print(f"Workspace directory does not exist: {workspace_root}", file=sys.stderr)
        return 1

    if not args.skip_image_build:
        build_status = run_command(build_docker_build_command(image_tag=args.image_tag))
        if build_status != 0:
            return build_status

    if args.prepare_board_debug_bridge:
        prepare_status = run_command(build_prepare_board_command())
        if prepare_status != 0:
            return prepare_status

    run_status = run_command(
        build_docker_run_command(
            workspace_root=workspace_root,
            image_tag=args.image_tag,
            model_path=args.model,
            output_dir=args.output_dir,
            target=args.target,
            device_id=args.device_id,
            adb_connect=args.adb_connect,
            adb_serial=args.adb_serial,
            verbose=args.verbose,
        )
    )
    return run_status


if __name__ == "__main__":
    raise SystemExit(main())