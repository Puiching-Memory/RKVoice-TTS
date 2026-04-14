from __future__ import annotations

import argparse
from pathlib import Path

from . import asr, tts
from .config import (
    ASR_DEFAULT_REMOTE_DIR,
    ASR_DEFAULT_RUNTIME_DIR,
    ASR_DEFAULT_STAGE_DIR,
    TTS_DEFAULT_REMOTE_DIR,
    TTS_DEFAULT_RUNTIME_DIR,
    TTS_DEFAULT_STAGE_DIR,
    TTS_DEFAULT_TEXT,
    load_local_settings,
    resolve_int_option,
    resolve_path_option,
    resolve_required_text_option,
    resolve_text_option,
)
from .remote import guess_source_ip
from .shared import fail, log


def _add_source_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--stage-dir", type=Path, default=None, help="Local source bundle directory")
    subparser.add_argument("--force", action="store_true", help="Recreate local bundle directories")


def _add_build_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--runtime-dir", type=Path, default=None, help="Local unified runtime project directory")


def _add_remote_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--host", default=None, help="Board SSH host")
    subparser.add_argument("--username", default=None, help="Board SSH username")
    subparser.add_argument("--password", default=None, help="Board SSH password")
    subparser.add_argument("--source-ip", default=None, help="Optional local source IP used to reach the board")
    subparser.add_argument("--remote-dir", default=None, help="Remote unified runtime project directory on the board")
    subparser.add_argument("--ssh-timeout", type=int, default=None, help="SSH connect timeout in seconds")
    subparser.add_argument("--remote-timeout", type=int, default=None, help="Remote command timeout in seconds")
    subparser.add_argument("--skip-smoketest", action="store_true", help="Skip remote smoke test")


def _build_asr_parser(subparsers: argparse._SubParsersAction) -> None:
    asr_parser = subparsers.add_parser("asr", help="ASR pipeline (sherpa-onnx RK3588)")
    asr_sub = asr_parser.add_subparsers(dest="action", required=True)

    download_p = asr_sub.add_parser("download", help="Prepare the local ASR source bundle")
    _add_source_args(download_p)

    build_p = asr_sub.add_parser("build", help="Assemble the local ASR component inside the unified runtime project")
    _add_source_args(build_p)
    _add_build_args(build_p)

    upload_p = asr_sub.add_parser("upload", help="Upload the local ASR runtime component to the board")
    upload_p.add_argument("--runtime-dir", type=Path, default=None, help="Local unified runtime project directory")
    _add_remote_args(upload_p)
    upload_p.add_argument("--skip-rknn-smoketest", action="store_true", help="Run remote smoke test without the RKNN ASR verification step")

    all_p = asr_sub.add_parser("all", help="Prepare, assemble, upload, and smoke test the ASR runtime component")
    _add_source_args(all_p)
    _add_build_args(all_p)
    _add_remote_args(all_p)
    all_p.add_argument("--skip-build", action="store_true", help="Reuse an existing runtime bundle and skip local assembly")
    all_p.add_argument("--skip-rknn-smoketest", action="store_true", help="Run remote smoke test without the RKNN ASR verification step")


def _build_tts_parser(subparsers: argparse._SubParsersAction) -> None:
    tts_parser = subparsers.add_parser("tts", help="TTS pipeline (MeloTTS-RKNN2)")
    tts_sub = tts_parser.add_subparsers(dest="action", required=True)

    download_p = tts_sub.add_parser("download", help="Prepare the local TTS source bundle")
    _add_source_args(download_p)

    build_p = tts_sub.add_parser("build", help="Assemble the local TTS component inside the unified runtime project")
    _add_source_args(build_p)
    _add_build_args(build_p)

    upload_p = tts_sub.add_parser("upload", help="Upload the local TTS runtime component to the board")
    upload_p.add_argument("--runtime-dir", type=Path, default=None, help="Local unified runtime project directory")
    _add_remote_args(upload_p)
    upload_p.add_argument("--text", default=None, help="Smoke test TTS text")
    upload_p.add_argument("--skip-python-deps-install", action="store_true", help="Skip the board-side offline Python dependency installation step")

    all_p = tts_sub.add_parser("all", help="Prepare, assemble, upload, and smoke test the TTS runtime component")
    _add_source_args(all_p)
    _add_build_args(all_p)
    _add_remote_args(all_p)
    all_p.add_argument("--text", default=None, help="Smoke test TTS text")
    all_p.add_argument("--skip-build", action="store_true", help="Reuse an existing runtime bundle and skip local assembly")
    all_p.add_argument("--skip-python-deps-install", action="store_true", help="Skip the board-side offline Python dependency installation step")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RKVoice delivery: prepare, assemble, and deploy ASR/TTS runtime components inside one unified runtime project")
    subparsers = parser.add_subparsers(dest="pipeline", required=True)
    _build_asr_parser(subparsers)
    _build_tts_parser(subparsers)
    return parser.parse_args()


def _resolve_board_connection(args: argparse.Namespace, local_settings: dict[str, str]) -> dict:
    host = resolve_required_text_option(
        args.host,
        env_names=("RKVOICE_BOARD_HOST",),
        local_settings=local_settings,
        option_name="board host",
    )
    username = resolve_required_text_option(
        args.username,
        env_names=("RKVOICE_BOARD_USERNAME",),
        local_settings=local_settings,
        option_name="board username",
    )
    password = resolve_required_text_option(
        args.password,
        env_names=("RKVOICE_BOARD_PASSWORD",),
        local_settings=local_settings,
        option_name="board password",
    )
    ssh_timeout = resolve_int_option(
        args.ssh_timeout,
        env_names=("RKVOICE_SSH_TIMEOUT",),
        local_settings=local_settings,
        default=8,
    )
    remote_timeout = resolve_int_option(
        args.remote_timeout,
        env_names=("RKVOICE_REMOTE_TIMEOUT",),
        local_settings=local_settings,
        default=1800,
    )
    source_ip_value = resolve_text_option(
        args.source_ip,
        env_names=("RKVOICE_SOURCE_IP",),
        local_settings=local_settings,
        default="",
    )
    source_ip = source_ip_value.strip() or None
    if source_ip is None:
        source_ip = guess_source_ip(host)
        if source_ip:
            log(f"Detected local source IP for board access: {source_ip}")
    return dict(
        host=host,
        username=username,
        password=password,
        source_ip=source_ip,
        ssh_timeout=ssh_timeout,
        remote_timeout=remote_timeout,
    )


def _run_asr(args: argparse.Namespace) -> None:
    local_settings = load_local_settings()

    stage_dir = resolve_path_option(
        getattr(args, "stage_dir", None),
        env_names=("RKVOICE_ASR_STAGE_DIR", "RKVOICE_STAGE_DIR"),
        local_settings=local_settings,
        default=ASR_DEFAULT_STAGE_DIR,
    )
    runtime_dir = resolve_path_option(
        getattr(args, "runtime_dir", None),
        env_names=("RKVOICE_ASR_RUNTIME_DIR", "RKVOICE_RUNTIME_DIR"),
        local_settings=local_settings,
        default=ASR_DEFAULT_RUNTIME_DIR,
    )

    if args.action in {"download", "build", "all"}:
        stage_dir = asr.prepare_source_bundle(stage_dir, force=getattr(args, "force", False))
        log(f"Source bundle prepared at {stage_dir}")

    if args.action in {"build", "all"} and not getattr(args, "skip_build", False):
        runtime_dir = asr.build_runtime_bundle(stage_dir, runtime_dir, force=getattr(args, "force", False))

    if args.action == "upload" and not all(p.exists() for p in asr.runtime_bundle_required_paths(runtime_dir)):
        fail(f"Runtime bundle does not exist: {runtime_dir}")

    if args.action == "all" and getattr(args, "skip_build", False) and not all(p.exists() for p in asr.runtime_bundle_required_paths(runtime_dir)):
        fail(f"Requested --skip-build but runtime bundle is missing: {runtime_dir}")

    if args.action in {"upload", "all"}:
        conn = _resolve_board_connection(args, local_settings)
        remote_dir = resolve_text_option(
            args.remote_dir,
            env_names=("RKVOICE_ASR_REMOTE_DIR", "RKVOICE_REMOTE_DIR"),
            local_settings=local_settings,
            default=ASR_DEFAULT_REMOTE_DIR,
        )
        asr.deploy_runtime_bundle(
            runtime_dir,
            remote_dir,
            host=conn["host"],
            username=conn["username"],
            password=conn["password"],
            source_ip=conn["source_ip"],
            ssh_timeout=conn["ssh_timeout"],
            remote_timeout=conn["remote_timeout"],
            skip_smoketest=args.skip_smoketest,
            enable_rknn_smoketest=not getattr(args, "skip_rknn_smoketest", False),
        )
        log(f"Runtime bundle deployed to {remote_dir}")


def _run_tts(args: argparse.Namespace) -> None:
    local_settings = load_local_settings()

    stage_dir = resolve_path_option(
        getattr(args, "stage_dir", None),
        env_names=("RKVOICE_TTS_STAGE_DIR", "RKVOICE_MELO_STAGE_DIR"),
        local_settings=local_settings,
        default=TTS_DEFAULT_STAGE_DIR,
    )
    runtime_dir = resolve_path_option(
        getattr(args, "runtime_dir", None),
        env_names=("RKVOICE_TTS_RUNTIME_DIR", "RKVOICE_MELO_RUNTIME_DIR"),
        local_settings=local_settings,
        default=TTS_DEFAULT_RUNTIME_DIR,
    )

    if args.action in {"download", "build", "all"}:
        stage_dir = tts.prepare_source_bundle(stage_dir, force=getattr(args, "force", False))
        log(f"Source bundle prepared at {stage_dir}")

    if args.action in {"build", "all"} and not getattr(args, "skip_build", False):
        runtime_dir = tts.build_runtime_bundle(stage_dir, runtime_dir, force=getattr(args, "force", False))

    if args.action == "upload" and not all(p.exists() for p in tts.runtime_bundle_required_paths(runtime_dir)):
        fail(f"Runtime bundle does not exist: {runtime_dir}")

    if args.action == "all" and getattr(args, "skip_build", False) and not all(p.exists() for p in tts.runtime_bundle_required_paths(runtime_dir)):
        fail(f"Requested --skip-build but runtime bundle is missing: {runtime_dir}")

    if args.action in {"upload", "all"}:
        conn = _resolve_board_connection(args, local_settings)
        remote_dir = resolve_text_option(
            args.remote_dir,
            env_names=("RKVOICE_TTS_REMOTE_DIR", "RKVOICE_MELO_REMOTE_DIR"),
            local_settings=local_settings,
            default=TTS_DEFAULT_REMOTE_DIR,
        )
        text = resolve_text_option(
            getattr(args, "text", None),
            env_names=("RKVOICE_TTS_TEXT", "RKVOICE_MELO_TTS_TEXT"),
            local_settings=local_settings,
            default=TTS_DEFAULT_TEXT,
        )
        tts.deploy_runtime_bundle(
            runtime_dir,
            remote_dir,
            host=conn["host"],
            username=conn["username"],
            password=conn["password"],
            source_ip=conn["source_ip"],
            ssh_timeout=conn["ssh_timeout"],
            remote_timeout=conn["remote_timeout"],
            text=text,
            install_python_deps=not getattr(args, "skip_python_deps_install", False),
            skip_smoketest=args.skip_smoketest,
        )
        log(f"Runtime bundle deployed to {remote_dir}")


def main() -> None:
    args = parse_args()
    if args.pipeline == "asr":
        _run_asr(args)
    elif args.pipeline == "tts":
        _run_tts(args)
    else:
        fail(f"Unknown pipeline: {args.pipeline}")
