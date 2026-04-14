from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

try:
    from scripts.delivery.config import load_local_settings, resolve_int_option, resolve_required_text_option, resolve_text_option
    from scripts.delivery.remote import guess_source_ip, open_ssh_client, run_remote_command, sh_quote, upload_file
except ImportError:
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))
    from scripts.delivery.config import load_local_settings, resolve_int_option, resolve_required_text_option, resolve_text_option
    from scripts.delivery.remote import guess_source_ip, open_ssh_client, run_remote_command, sh_quote, upload_file


DEFAULT_ADBD_ZIP_URL = "https://ftzr.zbox.filez.com/v2/delivery/data/7f0ac30dfa474892841fcb2cd29ad924/adbd.zip"
DEFAULT_ADBD_ARCHIVE_MEMBER = "adbd/linux-aarch64/adbd"
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / "artifacts" / "cache" / "adbd"
DEFAULT_REMOTE_STAGE_DIR = "/userdata/rkvoice-adbd"
DEFAULT_REMOTE_ADBD_PATH = "/usr/bin/adbd"
DEFAULT_REMOTE_RKNN_SERVER_PATH = "/usr/bin/rknn_server"
DEFAULT_REMOTE_RKNN_SERVER_LOG_PATH = "/userdata/rknn_server.log"


class BoardPreparationError(Exception):
    pass


@dataclass(frozen=True)
class SocketState:
    has_adbd_5037: bool
    has_adbd_5555: bool
    raw_5037_output: str
    raw_5555_output: str


@dataclass(frozen=True)
class BoardPreparationSummary:
    generated_at_utc: str
    board_host: str
    remote_adbd_path: str
    remote_rknn_server_path: str
    adbd_backup_path: str | None
    adbd_replaced: bool
    had_adbd_5037_before: bool
    has_adbd_5037_after: bool
    has_adbd_5555_after: bool
    rknn_server_restarted: bool
    rknn_server_running: bool
    transfer_proxy_ready: bool
    remote_rknn_server_log_path: str


def path_to_posix(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def socket_state_has_port(output: str, port: int) -> bool:
    return f":{port}" in output


def build_replace_adbd_command(*, staged_binary: str, target_binary: str, backup_path: str | None) -> str:
    commands: list[str] = []
    temp_target_binary = f"{target_binary}.rkvoice.new"
    if backup_path:
        commands.append(f"cp {sh_quote(target_binary)} {sh_quote(backup_path)}")
    commands.append(f"cp {sh_quote(staged_binary)} {sh_quote(temp_target_binary)}")
    commands.append(f"chmod +x {sh_quote(temp_target_binary)}")
    commands.append(f"mv -f {sh_quote(temp_target_binary)} {sh_quote(target_binary)}")
    return " && ".join(commands)


def build_start_rknn_server_command(*, remote_rknn_server_path: str, loglevel: int, log_path: str) -> str:
    log_dir = str(Path(log_path).parent).replace("\\", "/")
    return (
        f"mkdir -p {sh_quote(log_dir)} && "
        "pkill -x rknn_server >/dev/null 2>&1 || true; "
        f"export RKNN_SERVER_LOGLEVEL={int(loglevel)}; "
        f"nohup {sh_quote(remote_rknn_server_path)} >{sh_quote(log_path)} 2>&1 </dev/null &"
    )


def download_url(url: str, destination: Path) -> None:
    ensure_parent(destination)
    with urllib.request.urlopen(url) as response, destination.open("wb") as file_handle:
        shutil.copyfileobj(response, file_handle)


def ensure_local_adbd_binary(*, cache_dir: Path, adbd_zip_url: str, archive_member: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "adbd.zip"
    if not zip_path.exists():
        download_url(adbd_zip_url, zip_path)

    extract_root = cache_dir / "extracted"
    local_binary = extract_root / archive_member
    if not local_binary.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extract(archive_member, path=extract_root)
        os.chmod(local_binary, os.stat(local_binary).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return local_binary.resolve()


def query_socket_state(client: object, *, timeout: int) -> SocketState:
    output_5037 = run_remote_command(client, "ss -lntp | grep 5037 || true", timeout=timeout)
    output_5555 = run_remote_command(client, "ss -lntp | grep 5555 || true", timeout=timeout)
    return SocketState(
        has_adbd_5037=socket_state_has_port(output_5037, 5037),
        has_adbd_5555=socket_state_has_port(output_5555, 5555),
        raw_5037_output=output_5037.strip(),
        raw_5555_output=output_5555.strip(),
    )


def remote_process_running(client: object, process_name: str, *, timeout: int) -> bool:
    output = run_remote_command(client, f"pgrep -x {sh_quote(process_name)} >/dev/null 2>&1 && echo running || true", timeout=timeout)
    return "running" in output


def remote_transfer_proxy_ready(client: object, *, timeout: int) -> bool:
    output = run_remote_command(client, "ss -xap | grep '@transfer_proxy' || true", timeout=timeout)
    return "@transfer_proxy" in output


def backup_remote_adbd_path() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{DEFAULT_REMOTE_ADBD_PATH}.rkvoice.bak.{timestamp}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    local_settings = load_local_settings()
    parser = argparse.ArgumentParser(description="Repair board-side adbd/rknn_server state for RKNN Toolkit2 host-side profiling")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Host cache directory for the adbd repair bundle")
    parser.add_argument("--adbd-zip-url", default=DEFAULT_ADBD_ZIP_URL, help="URL of the adbd repair bundle documented by RKNN Toolkit2")
    parser.add_argument("--archive-member", default=DEFAULT_ADBD_ARCHIVE_MEMBER, help="Archive member path for the Linux aarch64 adbd binary")
    parser.add_argument("--remote-stage-dir", default=DEFAULT_REMOTE_STAGE_DIR, help="Temporary remote directory used to upload the repaired adbd binary")
    parser.add_argument("--remote-adbd-path", default=DEFAULT_REMOTE_ADBD_PATH, help="Target adbd binary path on the board")
    parser.add_argument("--remote-rknn-server-path", default=DEFAULT_REMOTE_RKNN_SERVER_PATH, help="rknn_server path on the board")
    parser.add_argument("--remote-rknn-server-log-path", default=DEFAULT_REMOTE_RKNN_SERVER_LOG_PATH, help="Remote log path used when restarting rknn_server")
    parser.add_argument("--force-adbd-replace", action="store_true", help="Replace adbd even if 5037 is already listening")
    parser.add_argument("--skip-adbd-repair", action="store_true", help="Skip adbd replacement and only restart rknn_server")
    parser.add_argument("--skip-rknn-server-restart", action="store_true", help="Only repair adbd and leave rknn_server untouched")
    parser.add_argument("--rknn-server-loglevel", type=int, default=0, help="RKNN_SERVER_LOGLEVEL used when restarting rknn_server")
    parser.add_argument("--ssh-timeout", type=int, default=None, help="Override SSH timeout in seconds")
    parser.add_argument("--summary-path", type=Path, default=None, help="Optional local JSON summary output path")
    parsed = parser.parse_args(argv)

    parsed.board_host = resolve_required_text_option(None, env_names=("RKVOICE_BOARD_HOST", "TTS_BOARD_HOST"), local_settings=local_settings, option_name="board host")
    parsed.board_username = resolve_required_text_option(None, env_names=("RKVOICE_BOARD_USERNAME", "TTS_BOARD_USERNAME"), local_settings=local_settings, option_name="board username")
    parsed.board_password = resolve_required_text_option(None, env_names=("RKVOICE_BOARD_PASSWORD", "TTS_BOARD_PASSWORD"), local_settings=local_settings, option_name="board password")
    parsed.source_ip = resolve_text_option(None, env_names=("RKVOICE_SOURCE_IP", "TTS_SOURCE_IP"), local_settings=local_settings, default="") or guess_source_ip(parsed.board_host)
    parsed.ssh_timeout = resolve_int_option(parsed.ssh_timeout, env_names=("RKVOICE_SSH_TIMEOUT", "TTS_SSH_TIMEOUT"), local_settings=local_settings, default=30)
    return parsed


def prepare_board(args: argparse.Namespace) -> BoardPreparationSummary:
    remote_stage_binary = f"{args.remote_stage_dir.rstrip('/')}/adbd"
    adbd_backup_path: str | None = None
    adbd_replaced = False
    rknn_server_restarted = False

    client = open_ssh_client(args.board_host, args.board_username, args.board_password, source_ip=args.source_ip, timeout=args.ssh_timeout)
    try:
        before_state = query_socket_state(client, timeout=args.ssh_timeout)

        if not args.skip_adbd_repair and (args.force_adbd_replace or not before_state.has_adbd_5037):
            local_adbd = ensure_local_adbd_binary(cache_dir=args.cache_dir, adbd_zip_url=args.adbd_zip_url, archive_member=args.archive_member)
            adbd_backup_path = backup_remote_adbd_path()
            run_remote_command(client, f"mkdir -p {sh_quote(args.remote_stage_dir)}", timeout=args.ssh_timeout)
            upload_file(client, local_adbd, remote_stage_binary)
            run_remote_command(client, f"chmod +x {sh_quote(remote_stage_binary)}", timeout=args.ssh_timeout)
            replace_command = build_replace_adbd_command(
                staged_binary=remote_stage_binary,
                target_binary=args.remote_adbd_path,
                backup_path=adbd_backup_path,
            )
            run_remote_command(client, replace_command, timeout=args.ssh_timeout)
            run_remote_command(client, "pkill -x adbd >/dev/null 2>&1 || true", timeout=args.ssh_timeout)
            time.sleep(3.0)
            adbd_replaced = True

        after_adbd_state = query_socket_state(client, timeout=args.ssh_timeout)
        if not after_adbd_state.has_adbd_5037:
            raise BoardPreparationError("Board adbd is still not listening on 5037 after repair")
        if not after_adbd_state.has_adbd_5555:
            raise BoardPreparationError("Board adbd is no longer listening on 5555 after repair")

        if not args.skip_rknn_server_restart:
            start_command = build_start_rknn_server_command(
                remote_rknn_server_path=args.remote_rknn_server_path,
                loglevel=args.rknn_server_loglevel,
                log_path=args.remote_rknn_server_log_path,
            )
            run_remote_command(client, start_command, timeout=args.ssh_timeout)
            time.sleep(2.0)
            rknn_server_restarted = True

        rknn_server_running = remote_process_running(client, "rknn_server", timeout=args.ssh_timeout)
        transfer_proxy_ready = remote_transfer_proxy_ready(client, timeout=args.ssh_timeout)

        return BoardPreparationSummary(
            generated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            board_host=args.board_host,
            remote_adbd_path=args.remote_adbd_path,
            remote_rknn_server_path=args.remote_rknn_server_path,
            adbd_backup_path=adbd_backup_path,
            adbd_replaced=adbd_replaced,
            had_adbd_5037_before=before_state.has_adbd_5037,
            has_adbd_5037_after=after_adbd_state.has_adbd_5037,
            has_adbd_5555_after=after_adbd_state.has_adbd_5555,
            rknn_server_restarted=rknn_server_restarted,
            rknn_server_running=rknn_server_running,
            transfer_proxy_ready=transfer_proxy_ready,
            remote_rknn_server_log_path=args.remote_rknn_server_log_path,
        )
    finally:
        client.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = prepare_board(args)
    except BoardPreparationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.summary_path is not None:
        write_json(args.summary_path, asdict(summary))

    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())