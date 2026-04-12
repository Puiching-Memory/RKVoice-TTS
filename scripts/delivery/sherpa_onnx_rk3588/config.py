from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .shared import fail


DELIVERY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = DELIVERY_ROOT / "templates" / "sherpa_onnx_rk3588"
WORKSPACE_ROOT = DELIVERY_ROOT.parent.parent
DEFAULT_STAGE_DIR = WORKSPACE_ROOT / "artifacts" / "source-bundles" / "sherpa_onnx_rk3588"
DEFAULT_RUNTIME_DIR = WORKSPACE_ROOT / "artifacts" / "runtime" / "sherpa_onnx_rk3588_runtime"
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / "artifacts" / "cache" / "sherpa_onnx_rk3588"
CONFIG_LOCAL_DIR = WORKSPACE_ROOT / "config" / "local"
BOARD_ENV_FILE = CONFIG_LOCAL_DIR / "board.local.env"
DELIVERY_ENV_FILE = CONFIG_LOCAL_DIR / "delivery.local.env"
DEFAULT_REMOTE_DIR = "/root/rkvoice/sherpa_onnx_rk3588_runtime"
DEFAULT_TTS_TEXT = "你好，欢迎使用 RKVoice sherpa-onnx 离线语音服务。"

SHERPA_ONNX_RELEASE_TAG = "v1.12.37"
SHERPA_ONNX_RUNTIME_ASSET_VERSION = "v1.12.36"
SENSE_VOICE_VERSION = "2025-09-09"
SENSE_VOICE_RKNN_VERSION = "2025-09-09"

PREBUILT_RUNTIME_DIR_NAME = "sherpa-onnx-runtime"
CPU_ASR_DIR_NAME = "sense-voice"
RKNN_ASR_DIR_NAME = "sense-voice-rk3588-20s"
TTS_DIR_NAME = "vits-icefall-zh-aishell3"


@lru_cache(maxsize=None)
def load_template(relative_path: str) -> str:
    template_path = TEMPLATE_ROOT / Path(relative_path)
    try:
        return template_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing delivery template: {template_path}") from exc


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
    settings: dict[str, str] = {}
    for env_file in (BOARD_ENV_FILE, DELIVERY_ENV_FILE):
        settings.update(parse_env_file(env_file))
    return settings


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
        fail(
            f"Missing {option_name}. Set it on the command line, via environment variables {', '.join(env_names)}, "
            f"or in {BOARD_ENV_FILE} / {DELIVERY_ENV_FILE}."
        )
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


def resolve_path_option(
    explicit_value: Path | None,
    *,
    env_names: tuple[str, ...],
    local_settings: dict[str, str],
    default: Path,
) -> Path:
    if explicit_value is not None:
        return explicit_value.resolve()
    for env_name in env_names:
        value = os.environ.get(env_name, "").strip()
        if value:
            return Path(value).resolve()
    for env_name in env_names:
        value = local_settings.get(env_name, "").strip()
        if value:
            return Path(value).resolve()
    return default.resolve()


@dataclass(frozen=True)
class Artifact:
    name: str
    url: str
    target_subdir: str
    strip_top_level: bool = False
    extracted_dir_name: str | None = None


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact(
        name=f"sherpa-onnx-{SHERPA_ONNX_RUNTIME_ASSET_VERSION}-rknn-linux-aarch64-shared.tar.bz2",
        url=(
            f"https://github.com/k2-fsa/sherpa-onnx/releases/download/{SHERPA_ONNX_RELEASE_TAG}/"
            f"sherpa-onnx-{SHERPA_ONNX_RUNTIME_ASSET_VERSION}-rknn-linux-aarch64-shared.tar.bz2"
        ),
        target_subdir="prebuilt",
        strip_top_level=True,
        extracted_dir_name=PREBUILT_RUNTIME_DIR_NAME,
    ),
    Artifact(
        name=f"sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-{SENSE_VOICE_VERSION}.tar.bz2",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            f"sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-{SENSE_VOICE_VERSION}.tar.bz2"
        ),
        target_subdir="models/asr/cpu",
        strip_top_level=True,
        extracted_dir_name=CPU_ASR_DIR_NAME,
    ),
    Artifact(
        name=f"sherpa-onnx-rk3588-20-seconds-sense-voice-zh-en-ja-ko-yue-{SENSE_VOICE_RKNN_VERSION}.tar.bz2",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            f"sherpa-onnx-rk3588-20-seconds-sense-voice-zh-en-ja-ko-yue-{SENSE_VOICE_RKNN_VERSION}.tar.bz2"
        ),
        target_subdir="models/asr/rknn",
        strip_top_level=True,
        extracted_dir_name=RKNN_ASR_DIR_NAME,
    ),
    Artifact(
        name="vits-icefall-zh-aishell3.tar.bz2",
        url="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2",
        target_subdir="models/tts",
        strip_top_level=True,
        extracted_dir_name=TTS_DIR_NAME,
    ),
)


RUNTIME_README = load_template("runtime/README_SDK.md")
RUNTIME_RUN_ASR_SH = load_template("runtime/run_asr.sh")
RUNTIME_RUN_TTS_SH = load_template("runtime/run_tts.sh")
RUNTIME_SMOKETEST_SH = load_template("runtime/smoketest.sh")
RUNTIME_CHECK_RKNN_ENV_SH = load_template("runtime/tools/check_rknn_env.sh")
RUNTIME_BOARD_PROFILE_CAPABILITIES_SH = load_template("runtime/tools/board_profile_capabilities.sh")
RUNTIME_PROFILE_ASR_INFERENCE_SH = load_template("runtime/tools/profile_asr_inference.sh")
RUNTIME_PROFILE_TTS_INFERENCE_SH = load_template("runtime/tools/profile_tts_inference.sh")