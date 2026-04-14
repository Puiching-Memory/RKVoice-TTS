from __future__ import annotations

import socket
import tarfile
import time
from pathlib import Path

import paramiko

from .shared import ensure_parent, fail, log, write_text


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
