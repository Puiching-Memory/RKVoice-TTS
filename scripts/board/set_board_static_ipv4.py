from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import paramiko


SCRIPT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_ROOT.parent.parent
CONFIG_LOCAL_DIR = WORKSPACE_ROOT / "config" / "local"
BOARD_ENV_FILE = CONFIG_LOCAL_DIR / "board.local.env"


def strip_matching_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_env_file(path: Path) -> dict[str, str]:
    settings: dict[str, str] = {}
    if not path.exists():
        return settings
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        settings[key] = strip_matching_quotes(value.strip())
    return settings


def load_local_settings() -> dict[str, str]:
    return parse_env_file(BOARD_ENV_FILE)


def resolve_text_option(
    explicit_value: str | None,
    *,
    env_names: tuple[str, ...],
    local_settings: dict[str, str],
    default: str | None = None,
) -> str | None:
    if explicit_value is not None and explicit_value != "":
        return explicit_value
    for env_name in env_names:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    for env_name in env_names:
        value = local_settings.get(env_name, "").strip()
        if value:
            return value
    return default


def resolve_required_text_option(
    explicit_value: str | None,
    *,
    env_names: tuple[str, ...],
    local_settings: dict[str, str],
    option_name: str,
) -> str:
    resolved = resolve_text_option(explicit_value, env_names=env_names, local_settings=local_settings)
    if resolved is None or resolved == "":
        fail(f"Missing {option_name}. Set it on the command line, via environment variables {', '.join(env_names)}, or in {BOARD_ENV_FILE}.")
    return resolved


def resolve_int_option(
    explicit_value: int | None,
    *,
    env_names: tuple[str, ...],
    local_settings: dict[str, str],
    default: int,
) -> int:
    if explicit_value is not None:
        return explicit_value
    for env_name in env_names:
        value = os.environ.get(env_name, "").strip()
        if value:
            return int(value)
    for env_name in env_names:
        value = local_settings.get(env_name, "").strip()
        if value:
            return int(value)
    return default


@dataclass
class AdapterInfo:
    name: str
    mac: str
    if_index: int
    ipv4_addresses: list[str]
    description: str = ""
    interface_guid: str = ""


@dataclass
class BoardInfo:
    mac: str
    ipv6: str
    hostname: str | None


def fail(message: str, exit_code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


def normalize_mac(value: str) -> str:
    return value.strip().lower().replace("-", ":")


def run_local(
    command: list[str],
    *,
    check: bool = True,
    description: str | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if check and result.returncode != 0:
        detail = description or "命令执行失败"
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        extra = stderr or stdout or f"退出码: {result.returncode}"
        fail(f"{detail}: {extra}")
    return result


def run_powershell_json(script: str) -> object:
    script = (
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "$OutputEncoding = [Console]::OutputEncoding; "
        + script
    )
    result = run_local(
        ["powershell.exe", "-NoProfile", "-Command", script],
        description="读取 Windows 网卡信息失败",
    )
    payload = result.stdout.strip()
    if not payload:
        fail("Windows 网卡信息为空")
    return json.loads(payload)


def get_adapter_info_from_raw(raw: dict[str, object]) -> AdapterInfo:
    ipv4_raw = raw.get("IPv4", [])
    if isinstance(ipv4_raw, str):
        ipv4_addresses = [ipv4_raw]
    elif isinstance(ipv4_raw, list):
        ipv4_addresses = [item for item in ipv4_raw if item]
    else:
        ipv4_addresses = []
    return AdapterInfo(
        name=str(raw["Name"]),
        mac=normalize_mac(str(raw["MacAddress"])),
        if_index=int(raw["IfIndex"]),
        ipv4_addresses=ipv4_addresses,
        description=str(raw.get("InterfaceDescription", "")),
        interface_guid=str(raw.get("InterfaceGuid", "")),
    )


def get_adapter_info(adapter_name: str) -> AdapterInfo:
    escaped_name = adapter_name.replace("'", "''")
    script = (
        f"$adapter = Get-NetAdapter -Name '{escaped_name}' -ErrorAction Stop; "
        "$ipv4 = @(Get-NetIPAddress -InterfaceIndex $adapter.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty IPAddress); "
        "[pscustomobject]@{"
        "Name=$adapter.Name;"
        "MacAddress=$adapter.MacAddress;"
        "IfIndex=$adapter.ifIndex;"
        "InterfaceDescription=$adapter.InterfaceDescription;"
        "InterfaceGuid=$adapter.InterfaceGuid;"
        "IPv4=$ipv4"
        "} | ConvertTo-Json -Compress"
    )
    raw = run_powershell_json(script)
    return get_adapter_info_from_raw(raw)


def get_connected_wired_adapter_info() -> AdapterInfo:
    script = (
        "$adapters = Get-NetAdapter | ForEach-Object { "
        "$ipv4 = @(Get-NetIPAddress -InterfaceIndex $_.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty IPAddress); "
        "[pscustomobject]@{"
        "Name=$_.Name;"
        "MacAddress=$_.MacAddress;"
        "IfIndex=$_.ifIndex;"
        "InterfaceDescription=$_.InterfaceDescription;"
        "InterfaceGuid=$_.InterfaceGuid;"
        "Status=$_.Status;"
        "HardwareInterface=$_.HardwareInterface;"
        "Virtual=$_.Virtual;"
        "IPv4=$ipv4"
        "} }; $adapters | ConvertTo-Json -Compress"
    )
    raw = run_powershell_json(script)
    if isinstance(raw, dict):
        candidates = [raw]
    else:
        candidates = list(raw)

    def score(adapter: dict[str, object]) -> int:
        name = str(adapter.get("Name", "")).lower()
        description = str(adapter.get("InterfaceDescription", "")).lower()
        if str(adapter.get("Status", "")).lower() != "up":
            return -999
        if not str(adapter.get("MacAddress", "")).strip():
            return -999
        text = f"{name} {description}"
        score_value = 0
        if adapter.get("HardwareInterface"):
            score_value += 5
        if "ethernet" in text or "gbe" in text or "lan" in text or "realtek" in text:
            score_value += 8
        if "wi-fi" in text or "wireless" in text or name == "wlan":
            score_value -= 100
        if "hyper-v" in text or "vmware" in text or "virtual" in text or "loopback" in text:
            score_value -= 100
        if "tunnel" in text or "vpn" in text or "meta" in text or "safetywork" in text:
            score_value -= 100
        return score_value

    ranked = sorted(candidates, key=score, reverse=True)
    if not ranked or score(ranked[0]) < 0:
        fail("没有自动识别到可用的直连有线网卡，请显式传入 --adapter-name")
    chosen = get_adapter_info_from_raw(ranked[0])
    print(f"自动选择 Windows 网卡: {chosen.name} ({chosen.description})")
    return chosen


def get_tshark_interface_index(tshark_path: Path, adapter_name: str) -> str:
    fail("内部错误: 不应再通过网卡名匹配 tshark 接口")


def get_tshark_interface_target(adapter: AdapterInfo) -> str:
    guid = adapter.interface_guid.strip()
    if not guid:
        fail(f"Windows 网卡 {adapter.name or adapter.description} 缺少 InterfaceGuid")
    return f"\\Device\\NPF_{guid.upper()}"


def capture_once(tshark_path: Path, interface_target: str, seconds: int) -> list[list[str]]:
    command = [
        str(tshark_path),
        "-i",
        interface_target,
        "-a",
        f"duration:{seconds}",
        "-n",
        "-l",
        "-f",
        "icmp6 or (udp port 67 or 68) or arp",
        "-T",
        "fields",
        "-E",
        "separator=,",
        "-E",
        "quote=d",
        "-e",
        "eth.src",
        "-e",
        "ipv6.src",
        "-e",
        "ip.src",
        "-e",
        "bootp.option.hostname",
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode not in (0, 1):
        stderr = result.stderr.strip() or result.stdout.strip()
        fail(f"tshark 抓包失败: {stderr or result.returncode}")
    rows: list[list[str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        row = next(csv.reader([line]))
        if row:
            rows.append(row)
    return rows


def discover_board(
    tshark_path: Path,
    interface_target: str,
    local_mac: str,
    discovery_timeout: int,
    hostname_hint: str,
) -> BoardInfo:
    deadline = time.time() + discovery_timeout
    candidates: dict[str, dict[str, str | None]] = {}
    while time.time() < deadline:
        remaining = max(1, int(deadline - time.time()))
        window = min(10, remaining)
        rows = capture_once(tshark_path, interface_target, window)
        for row in rows:
            src_mac = normalize_mac(row[0]) if len(row) > 0 else ""
            ipv6 = row[1].strip() if len(row) > 1 else ""
            ipv4 = row[2].strip() if len(row) > 2 else ""
            hostname = row[3].strip() if len(row) > 3 else ""
            if not src_mac or src_mac == local_mac:
                continue
            entry = candidates.setdefault(src_mac, {"ipv6": None, "hostname": None, "ipv4": None})
            if hostname:
                entry["hostname"] = hostname
            if ipv4:
                entry["ipv4"] = ipv4
            if ipv6.startswith("fe80::"):
                entry["ipv6"] = ipv6.split("%", 1)[0]
        for mac, entry in candidates.items():
            ipv6 = entry.get("ipv6")
            hostname = entry.get("hostname")
            if not ipv6:
                continue
            if hostname_hint and hostname and hostname_hint.lower() in hostname.lower():
                return BoardInfo(mac=mac, ipv6=str(ipv6), hostname=hostname)
            if len(candidates) == 1:
                return BoardInfo(mac=mac, ipv6=str(ipv6), hostname=hostname)
    hint = hostname_hint or "开发板"
    fail(f"在 {discovery_timeout} 秒内没有抓到 {hint} 的可用 IPv6 链路本地地址")


def connect_ipv6(ipv6: str, if_index: int, username: str, password: str, timeout: int) -> paramiko.SSHClient:
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((ipv6, 22, 0, if_index))
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=ipv6,
        username=username,
        password=password,
        sock=sock,
        look_for_keys=False,
        allow_agent=False,
        timeout=timeout,
        banner_timeout=timeout,
        auth_timeout=timeout,
    )
    return client


def connect_ipv4(host: str, username: str, password: str, timeout: int) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        username=username,
        password=password,
        look_for_keys=False,
        allow_agent=False,
        timeout=timeout,
        banner_timeout=timeout,
        auth_timeout=timeout,
    )
    return client


def run_remote(client: paramiko.SSHClient, command: str, *, check: bool = True, timeout: int = 20) -> tuple[str, str, int]:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", "ignore").strip()
    err = stderr.read().decode("utf-8", "ignore").strip()
    if check and exit_code != 0:
        detail = err or out or f"退出码 {exit_code}"
        fail(f"远端命令失败: {command}\n{detail}")
    return out, err, exit_code


def get_remote_iface_for_ipv6(client: paramiko.SSHClient, ipv6: str) -> str:
    out, _, _ = run_remote(client, "ip -6 -o addr show scope link")
    for line in out.splitlines():
        if ipv6 in line:
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    fail(f"没有在开发板上找到携带 {ipv6} 的网卡")


def get_nm_connection_name(client: paramiko.SSHClient, iface: str) -> str:
    out, _, code = run_remote(client, "command -v nmcli >/dev/null 2>&1", check=False)
    if code != 0:
        fail("开发板上没有 nmcli，当前脚本只支持 NetworkManager 管理的网口")
    out, _, _ = run_remote(client, "nmcli -t -f NAME,DEVICE connection show --active")
    for line in out.splitlines():
        if not line:
            continue
        name, device = line.rsplit(":", 1)
        if device == iface:
            return name
    out, _, _ = run_remote(client, "nmcli -t -f NAME,DEVICE connection show")
    for line in out.splitlines():
        if not line:
            continue
        name, device = line.rsplit(":", 1)
        if device == iface:
            return name
    fail(f"没有在 NetworkManager 里找到绑定到 {iface} 的连接配置")


def install_persistent_ipv4(client: paramiko.SSHClient, conn_name: str, iface: str, static_cidr: str) -> None:
    remote_script = f"""#!/bin/sh
set -eu

CONN={shlex.quote(conn_name)}
IFACE={shlex.quote(iface)}
CIDR={shlex.quote(static_cidr)}

nmcli connection modify "$CONN" \
  ipv4.method manual \
  ipv4.addresses "$CIDR" \
  ipv4.gateway "" \
  ipv4.dns "" \
  ipv4.never-default yes \
  ipv6.method auto \
  connection.autoconnect yes

nmcli device reapply "$IFACE" || nmcli connection up "$CONN" ifname "$IFACE"
ip -4 -br addr show dev "$IFACE" > /tmp/set_board_static_ipv4.result
"""
    sftp = client.open_sftp()
    remote_path = "/tmp/set_board_static_ipv4.sh"
    with sftp.file(remote_path, "w") as handle:
        handle.write(remote_script)
    sftp.chmod(remote_path, 0o755)
    sftp.close()
    run_remote(client, f"nohup {shlex.quote(remote_path)} >/tmp/set_board_static_ipv4.log 2>&1 &", check=False)


def verify_ipv4(host: str, username: str, password: str, iface: str, verify_timeout: int, ssh_timeout: int) -> str:
    deadline = time.time() + verify_timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            client = connect_ipv4(host, username, password, ssh_timeout)
            out, _, _ = run_remote(client, f"ip -4 -br addr show dev {shlex.quote(iface)}", check=False)
            client.close()
            return out or "SSH 已恢复，但未读取到 IPv4 地址输出"
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    detail = str(last_error) if last_error else "超时"
    fail(f"开发板 IPv4 {host} 在 {verify_timeout} 秒内未验证成功: {detail}", exit_code=2)


def ensure_same_subnet(local_ipv4_addresses: list[str], static_cidr: str) -> None:
    target = ipaddress.ip_interface(static_cidr)
    if not local_ipv4_addresses:
        print("警告: 当前 PC 侧有线网卡还没有 IPv4 地址，后续可能需要重插一次网线。")
        return
    for address in local_ipv4_addresses:
        try:
            current = ipaddress.ip_interface(f"{address}/{target.network.prefixlen}")
        except ValueError:
            continue
        if current.ip in target.network:
            return
    joined = ", ".join(local_ipv4_addresses)
    print(f"警告: 当前 PC 侧 IPv4 {joined} 不在 {target.network} 网段，配置完成后可能还需要调整 PC 侧 IPv4。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重插网线后自动发现 RK3588，并给直连网口设置固定 IPv4")
    parser.add_argument("--tshark-path", default=None)
    parser.add_argument("--adapter-name", default=None)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--hostname-hint", default=None)
    parser.add_argument("--static-cidr", default=None)
    parser.add_argument("--discovery-timeout", type=int, default=None)
    parser.add_argument("--ssh-timeout", type=int, default=None)
    parser.add_argument("--verify-timeout", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_settings = load_local_settings()
    tshark_path = Path(
        resolve_text_option(
            args.tshark_path,
            env_names=("TTS_TSHARK_PATH",),
            local_settings=local_settings,
            default=r"C:\workspace\apps\Wireshark\tshark.exe",
        )
    )
    if not tshark_path.exists():
        fail(f"找不到 tshark: {tshark_path}")
    adapter_name = resolve_text_option(
        args.adapter_name,
        env_names=("TTS_ADAPTER_NAME",),
        local_settings=local_settings,
        default="",
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
    hostname_hint = resolve_text_option(
        args.hostname_hint,
        env_names=("TTS_HOSTNAME_HINT",),
        local_settings=local_settings,
        default="RK3588",
    )
    static_cidr = resolve_text_option(
        args.static_cidr,
        env_names=("TTS_STATIC_CIDR",),
        local_settings=local_settings,
        default="169.254.46.2/16",
    )
    discovery_timeout = resolve_int_option(
        args.discovery_timeout,
        env_names=("TTS_DISCOVERY_TIMEOUT",),
        local_settings=local_settings,
        default=120,
    )
    ssh_timeout = resolve_int_option(
        args.ssh_timeout,
        env_names=("TTS_SSH_TIMEOUT",),
        local_settings=local_settings,
        default=8,
    )
    verify_timeout = resolve_int_option(
        args.verify_timeout,
        env_names=("TTS_VERIFY_TIMEOUT",),
        local_settings=local_settings,
        default=40,
    )

    adapter = get_adapter_info(adapter_name) if adapter_name else get_connected_wired_adapter_info()
    ensure_same_subnet(adapter.ipv4_addresses, static_cidr)
    interface_target = get_tshark_interface_target(adapter)
    print(f"等待开发板在 {adapter.description or adapter.name} 上出现，抓包接口 {interface_target}，超时 {discovery_timeout} 秒...")
    board = discover_board(
        tshark_path=tshark_path,
        interface_target=interface_target,
        local_mac=adapter.mac,
        discovery_timeout=discovery_timeout,
        hostname_hint=hostname_hint,
    )
    print(f"发现开发板: MAC={board.mac} IPv6={board.ipv6} 主机名={board.hostname or '未知'}")
    client = connect_ipv6(board.ipv6, adapter.if_index, username, password, ssh_timeout)
    out, _, _ = run_remote(client, "whoami; hostname")
    print("SSH 登录成功:")
    print(out)
    iface = get_remote_iface_for_ipv6(client, board.ipv6)
    conn_name = get_nm_connection_name(client, iface)
    print(f"目标网口: {iface}")
    print(f"NetworkManager 连接: {conn_name}")
    current_cfg, _, _ = run_remote(
        client,
        f"nmcli -f connection.id,ipv4.method,ipv4.addresses,ipv6.method connection show {shlex.quote(conn_name)}",
        check=False,
    )
    if current_cfg:
        print("当前连接配置:")
        print(current_cfg)
    install_persistent_ipv4(client, conn_name, iface, static_cidr)
    client.close()
    host = str(ipaddress.ip_interface(static_cidr).ip)
    print(f"已下发固定 IPv4 {static_cidr}，等待开发板切换配置...")
    verify_output = verify_ipv4(host, username, password, iface, verify_timeout, ssh_timeout)
    print("验证成功，开发板当前 IPv4:")
    print(verify_output)
    print(f"后续可直接使用: ssh {username}@{host}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())