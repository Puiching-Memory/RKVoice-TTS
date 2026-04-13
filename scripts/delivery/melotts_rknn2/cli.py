from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    DEFAULT_REMOTE_DIR,
    DEFAULT_RUNTIME_DIR,
    DEFAULT_STAGE_DIR,
    DEFAULT_TTS_TEXT,
    load_local_settings,
    resolve_int_option,
    resolve_path_option,
    resolve_required_text_option,
    resolve_text_option,
)
from .remote import deploy_runtime_bundle, guess_source_ip
from .runtime_bundle import build_runtime_bundle, runtime_bundle_required_paths
from .shared import fail, log
from .source_bundle import prepare_source_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, assemble, and deploy a MeloTTS-RKNN2 runtime bundle")
    subparsers = parser.add_subparsers(dest="action", required=True)

    def add_source_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--stage-dir", type=Path, default=None, help="Local source bundle directory")
        subparser.add_argument("--force", action="store_true", help="Recreate local bundle directories")

    def add_build_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--runtime-dir", type=Path, default=None, help="Local runtime bundle directory")

    def add_remote_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--host", default=None, help="Board SSH host")
        subparser.add_argument("--username", default=None, help="Board SSH username")
        subparser.add_argument("--password", default=None, help="Board SSH password")
        subparser.add_argument("--source-ip", default=None, help="Optional local source IP used to reach the board")
        subparser.add_argument("--remote-dir", default=None, help="Remote runtime directory on the board")
        subparser.add_argument("--ssh-timeout", type=int, default=None, help="SSH connect timeout in seconds")
        subparser.add_argument("--remote-timeout", type=int, default=None, help="Remote command timeout in seconds")
        subparser.add_argument("--text", default=None, help="Smoke test TTS text")
        subparser.add_argument("--skip-python-deps-install", action="store_true", help="Skip the board-side offline Python dependency installation step")
        subparser.add_argument("--skip-smoketest", action="store_true", help="Skip remote smoke test")

    download_parser = subparsers.add_parser("download", help="Prepare the local MeloTTS-RKNN2 source bundle")
    add_source_args(download_parser)

    build_parser = subparsers.add_parser("build", help="Assemble the local MeloTTS-RKNN2 runtime bundle")
    add_source_args(build_parser)
    add_build_args(build_parser)

    upload_parser = subparsers.add_parser("upload", help="Upload an existing runtime bundle to the board")
    upload_parser.add_argument("--runtime-dir", type=Path, default=None, help="Local runtime bundle directory")
    add_remote_args(upload_parser)

    all_parser = subparsers.add_parser("all", help="Prepare source bundle, assemble runtime bundle, upload, and smoke test")
    add_source_args(all_parser)
    add_build_args(all_parser)
    add_remote_args(all_parser)
    all_parser.add_argument("--skip-build", action="store_true", help="Reuse an existing runtime bundle and skip local assembly")

    return parser.parse_args()


def runtime_bundle_is_complete(runtime_dir: Path) -> bool:
    return all(path.exists() for path in runtime_bundle_required_paths(runtime_dir))


def main() -> None:
    args = parse_args()
    local_settings = load_local_settings()

    stage_dir = resolve_path_option(
        getattr(args, "stage_dir", None),
        env_names=("RKVOICE_MELO_STAGE_DIR",),
        local_settings=local_settings,
        default=DEFAULT_STAGE_DIR,
    )
    runtime_dir = resolve_path_option(
        getattr(args, "runtime_dir", None),
        env_names=("RKVOICE_MELO_RUNTIME_DIR",),
        local_settings=local_settings,
        default=DEFAULT_RUNTIME_DIR,
    )

    if args.action in {"download", "build", "all"}:
        stage_dir = prepare_source_bundle(stage_dir, force=getattr(args, "force", False))
        log(f"Source bundle prepared at {stage_dir}")

    if args.action in {"build", "all"} and not getattr(args, "skip_build", False):
        runtime_dir = build_runtime_bundle(stage_dir, runtime_dir, force=getattr(args, "force", False))

    if args.action == "upload" and not runtime_bundle_is_complete(runtime_dir):
        fail(f"Runtime bundle does not exist: {runtime_dir}")

    if args.action == "all" and getattr(args, "skip_build", False) and not runtime_bundle_is_complete(runtime_dir):
        fail(f"Requested --skip-build but runtime bundle is missing: {runtime_dir}")

    if args.action in {"upload", "all"}:
        host = resolve_required_text_option(
            args.host,
            env_names=("RKVOICE_BOARD_HOST", "TTS_BOARD_HOST"),
            local_settings=local_settings,
            option_name="board host",
        )
        username = resolve_required_text_option(
            args.username,
            env_names=("RKVOICE_BOARD_USERNAME", "TTS_BOARD_USERNAME"),
            local_settings=local_settings,
            option_name="board username",
        )
        password = resolve_required_text_option(
            args.password,
            env_names=("RKVOICE_BOARD_PASSWORD", "TTS_BOARD_PASSWORD"),
            local_settings=local_settings,
            option_name="board password",
        )
        remote_dir = resolve_text_option(
            args.remote_dir,
            env_names=("RKVOICE_MELO_REMOTE_DIR",),
            local_settings=local_settings,
            default=DEFAULT_REMOTE_DIR,
        )
        ssh_timeout = resolve_int_option(
            args.ssh_timeout,
            env_names=("RKVOICE_SSH_TIMEOUT", "TTS_SSH_TIMEOUT"),
            local_settings=local_settings,
            default=8,
        )
        remote_timeout = resolve_int_option(
            args.remote_timeout,
            env_names=("RKVOICE_REMOTE_TIMEOUT", "TTS_REMOTE_TIMEOUT"),
            local_settings=local_settings,
            default=1800,
        )
        text = resolve_text_option(
            args.text,
            env_names=("RKVOICE_MELO_TTS_TEXT",),
            local_settings=local_settings,
            default=DEFAULT_TTS_TEXT,
        )
        source_ip_value = resolve_text_option(
            args.source_ip,
            env_names=("RKVOICE_SOURCE_IP", "TTS_SOURCE_IP"),
            local_settings=local_settings,
            default="",
        )
        source_ip = source_ip_value.strip() or None
        if source_ip is None:
            source_ip = guess_source_ip(host)
            if source_ip:
                log(f"Detected local source IP for board access: {source_ip}")
        deploy_runtime_bundle(
            runtime_dir,
            remote_dir,
            host=host,
            username=username,
            password=password,
            source_ip=source_ip,
            ssh_timeout=ssh_timeout,
            remote_timeout=remote_timeout,
            text=text,
            install_python_deps=not args.skip_python_deps_install,
            skip_smoketest=args.skip_smoketest,
        )
        log(f"Runtime bundle deployed to {remote_dir}")