from __future__ import annotations

import socket
import tarfile
import time
from pathlib import Path

import paramiko

from .runtime_bundle import runtime_bundle_required_paths
from .shared import ensure_parent, fail, log, write_text
from .source_bundle import materialize_runtime_support_files


def create_bundle_tarball(local_dir: Path, remote_dir: str) -> Path:
    tarball_path = local_dir.parent / f"{local_dir.name}.tar.gz"
    root_name = Path(remote_dir.rstrip("/")).name
    if tarball_path.exists():
        tarball_path.unlink()
    with tarfile.open(tarball_path, "w:gz") as archive:
        archive.add(local_dir, arcname=root_name)
    return tarball_path


def guess_source_ip(host: str) -> str | None:
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        udp_socket.connect((host, 1))
        return udp_socket.getsockname()[0]
    except OSError:
        return None
    finally:
        udp_socket.close()


def open_ssh_client(host: str, username: str, password: str, *, source_ip: str | None, timeout: int) -> paramiko.SSHClient:
    sock = None
    if source_ip:
        sock = socket.create_connection((host, 22), timeout=timeout, source_address=(source_ip, 0))
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        username=username,
        password=password,
        timeout=timeout,
        banner_timeout=timeout,
        auth_timeout=timeout,
        sock=sock,
    )
    return client


def upload_file(client: paramiko.SSHClient, local_path: Path, remote_path: str) -> None:
    sftp = client.open_sftp()
    try:
        sftp.put(str(local_path), remote_path)
    finally:
        sftp.close()


def download_file(client: paramiko.SSHClient, remote_path: str, local_path: Path) -> None:
    ensure_parent(local_path)
    sftp = client.open_sftp()
    try:
        sftp.get(remote_path, str(local_path))
    finally:
        sftp.close()


def read_channel(channel: paramiko.Channel, *, timeout: int) -> tuple[int, str]:
    start_time = time.time()
    output_parts: list[str] = []
    while True:
        if channel.recv_ready():
            chunk = channel.recv(4096).decode("utf-8", "ignore")
            print(chunk, end="", flush=True)
            output_parts.append(chunk)
        if channel.recv_stderr_ready():
            chunk = channel.recv_stderr(4096).decode("utf-8", "ignore")
            print(chunk, end="", flush=True)
            output_parts.append(chunk)
        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
            break
        if timeout and time.time() - start_time > timeout:
            channel.close()
            fail("Remote command timed out")
        time.sleep(0.1)
    status = channel.recv_exit_status()
    return status, "".join(output_parts)


def run_remote_command(client: paramiko.SSHClient, command: str, *, timeout: int) -> str:
    transport = client.get_transport()
    if transport is None:
        fail("SSH transport is not available")
    channel = transport.open_session()
    channel.get_pty()
    channel.exec_command(command)
    status, output = read_channel(channel, timeout=timeout)
    if status != 0:
        fail(f"Remote command failed with exit code {status}: {command}")
    return output


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def deploy_runtime_bundle(
    runtime_dir: Path,
    remote_dir: str,
    *,
    host: str,
    username: str,
    password: str,
    source_ip: str | None,
    ssh_timeout: int,
    remote_timeout: int,
    text: str,
    skip_smoketest: bool,
    enable_rknn_smoketest: bool,
) -> None:
    for required_path in runtime_bundle_required_paths(runtime_dir):
        if not required_path.exists():
            fail(f"Runtime bundle is missing required artifact: {required_path}")

    materialize_runtime_support_files(runtime_dir)

    client = open_ssh_client(host, username, password, source_ip=source_ip, timeout=ssh_timeout)
    local_output_dir = runtime_dir / "output"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = create_bundle_tarball(runtime_dir, remote_dir)
    remote_parent = str(Path(remote_dir).parent).replace("\\", "/")
    remote_tarball = f"{remote_parent}/{Path(tarball_path.name).name}"
    try:
        log(f"Uploading runtime tarball to {remote_tarball}")
        run_remote_command(client, f"mkdir -p {sh_quote(remote_parent)}", timeout=remote_timeout)
        upload_file(client, tarball_path, remote_tarball)
        run_remote_command(
            client,
            f"rm -rf {sh_quote(remote_dir)} && tar -xzf {sh_quote(remote_tarball)} -C {sh_quote(remote_parent)}",
            timeout=remote_timeout,
        )
        run_remote_command(
            client,
            (
                f"find {sh_quote(remote_dir)} -type f -name '*.sh' -exec chmod +x {{}} + && "
                f"find {sh_quote(remote_dir)}/bin -maxdepth 1 -type f -exec chmod +x {{}} +"
            ),
            timeout=remote_timeout,
        )
        if not skip_smoketest:
            log("Running remote smoke test")
            remote_output_wav = "./output/smoke_test_tts.wav"
            remote_output_log = "./output/smoke_test_summary.log"
            rknn_flag = "1" if enable_rknn_smoketest else "0"
            smoke_test_command = (
                "set -o pipefail; "
                "mkdir -p ./output; "
                f"RKVOICE_ENABLE_RKNN_SMOKETEST={rknn_flag} ./smoketest.sh {sh_quote(text)} {sh_quote(remote_output_wav)} "
                f"2>&1 | tee {sh_quote(remote_output_log)}"
            )
            smoke_test_output = run_remote_command(
                client,
                f"cd {sh_quote(remote_dir)} && bash -lc {sh_quote(smoke_test_command)}",
                timeout=remote_timeout,
            )
            local_log_path = local_output_dir / "smoke_test_summary.log"
            write_text(local_log_path, smoke_test_output)
            remote_output_dir = f"{remote_dir.rstrip('/')}/output"
            local_wav_path = local_output_dir / "smoke_test_tts.wav"
            download_file(client, f"{remote_output_dir}/smoke_test_tts.wav", local_wav_path)
            download_file(client, f"{remote_output_dir}/smoke_test_summary.log", local_log_path)
            for optional_name in (
                "board_profile_capabilities.txt",
                "rknpu_load.log",
                "rknn_profile.log",
                "rknn_runtime.log",
                "rknn_eval_perf.txt",
                "rknn_query_perf_detail.txt",
                "rknn_perf_detail.txt",
                "rknn_perf_run.json",
                "rknn_query_perf_run.txt",
                "rknn_perf_run.txt",
                "rknn_memory_profile.txt",
                "rknn_eval_memory.txt",
                "rknn_query_mem_size.json",
            ):
                try:
                    download_file(client, f"{remote_output_dir}/{optional_name}", local_output_dir / optional_name)
                except OSError:
                    pass
            log(f"Downloaded smoke test audio to {local_wav_path}")
            log(f"Downloaded smoke test log to {local_log_path}")
    finally:
        client.close()