from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .shared import fail


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

DELIVERY_ROOT = Path(__file__).resolve().parent
TEMPLATE_ROOT = DELIVERY_ROOT / "templates"
WORKSPACE_ROOT = DELIVERY_ROOT.parent.parent
CONFIG_LOCAL_DIR = WORKSPACE_ROOT / "config" / "local"
BOARD_ENV_FILE = CONFIG_LOCAL_DIR / "board.local.env"
DELIVERY_ENV_FILE = CONFIG_LOCAL_DIR / "delivery.local.env"
AUDIOS_DIR = WORKSPACE_ROOT / "audios"

# ---------------------------------------------------------------------------
# ASR constants
# ---------------------------------------------------------------------------

ASR_TEMPLATE_ROOT = TEMPLATE_ROOT / "sherpa_onnx_rk3588"
ASR_DEFAULT_STAGE_DIR = WORKSPACE_ROOT / "artifacts" / "source-bundles" / "sherpa_onnx_rk3588"
ASR_DEFAULT_RUNTIME_DIR = WORKSPACE_ROOT / "artifacts" / "runtime" / "sherpa_onnx_rk3588_runtime"
ASR_DEFAULT_CACHE_DIR = WORKSPACE_ROOT / "artifacts" / "cache" / "sherpa_onnx_rk3588"
ASR_DEFAULT_REMOTE_DIR = "/root/rkvoice/sherpa_onnx_rk3588_runtime"

SHERPA_ONNX_RELEASE_TAG = "v1.12.37"
SHERPA_ONNX_RUNTIME_ASSET_VERSION = "v1.12.36"
STREAMING_RKNN_ZIPFORMER_VERSION = "2023-02-16"

PREBUILT_RUNTIME_DIR_NAME = "sherpa-onnx-runtime"
STREAMING_RKNN_ASR_DIR_NAME = "streaming-zipformer-rk3588-small"

# ---------------------------------------------------------------------------
# TTS constants
# ---------------------------------------------------------------------------

TTS_TEMPLATE_ROOT = TEMPLATE_ROOT / "melotts_rknn2"
TTS_DEFAULT_STAGE_DIR = WORKSPACE_ROOT / "artifacts" / "source-bundles" / "melotts_rknn2"
TTS_DEFAULT_RUNTIME_DIR = WORKSPACE_ROOT / "artifacts" / "runtime" / "melotts_rknn2_runtime"
TTS_DEFAULT_CACHE_DIR = WORKSPACE_ROOT / "artifacts" / "cache" / "melotts_rknn2"
TTS_DEFAULT_REMOTE_DIR = "/root/rkvoice/melotts_rknn2_runtime"
TTS_DEFAULT_TEXT = "你好，欢迎使用 RKVoice MeloTTS-RKNN2 离线语音服务。"

MELOTTS_REPO_REF = "master"
SOURCE_ROOT_DIR_NAME = "melotts_rknn2_upstream"
SOURCE_ROOT_RELATIVE_PATH = Path("snapshot") / SOURCE_ROOT_DIR_NAME
WHEELHOUSE_DIR_NAME = "wheels"
WHEELHOUSE_RELATIVE_PATH = Path(WHEELHOUSE_DIR_NAME)
BOARD_WHEEL_PLATFORM = "manylinux2014_aarch64"
BOARD_PYTHON_VERSION = "310"
BOARD_PYTHON_ABI = "cp310"
COMMON_PYTHON_DEPENDENCIES: tuple[str, ...] = (
    "pip",
    "numpy==1.24.4",
    "onnxruntime==1.16.0",
    "soundfile",
    "cn2an",
    "inflect",
    "psutil",
    "ruamel.yaml",
)
RKNN_LITE_DEPENDENCY = "rknn-toolkit-lite2==2.3.2"
REQUIRED_WHEEL_PATTERNS: tuple[str, ...] = (
    "pip-*.whl",
    "numpy-1.24.4-*.whl",
    "onnxruntime-1.16.0-*.whl",
    "soundfile-*.whl",
    "cn2an-*.whl",
    "inflect-*.whl",
    "psutil-*.whl",
    "ruamel_yaml-*.whl",
    "rknn_toolkit_lite2-2.3.2-*.whl",
)


# ---------------------------------------------------------------------------
# Artifact descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Artifact:
    name: str
    url: str
    target_subdir: str
    strip_top_level: bool = False
    extracted_dir_name: str | None = None


@dataclass(frozen=True)
class DirectFile:
    name: str
    url: str
    relative_path: str
    min_size_bytes: int = 1


ASR_ARTIFACTS: tuple[Artifact, ...] = (
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
        name=f"sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-{STREAMING_RKNN_ZIPFORMER_VERSION}.tar.bz2",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            f"sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-{STREAMING_RKNN_ZIPFORMER_VERSION}.tar.bz2"
        ),
        target_subdir="models/asr/streaming-rknn",
        strip_top_level=True,
        extracted_dir_name=STREAMING_RKNN_ASR_DIR_NAME,
    ),
)

TTS_ARTIFACTS: tuple[Artifact, ...] = (
    Artifact(
        name=f"MeloTTS-RKNN2-{MELOTTS_REPO_REF}.tar.gz",
        url=f"https://github.com/happyme531/MeloTTS-RKNN2/archive/refs/heads/{MELOTTS_REPO_REF}.tar.gz",
        target_subdir="snapshot",
        strip_top_level=True,
        extracted_dir_name=SOURCE_ROOT_DIR_NAME,
    ),
)

TTS_DIRECT_FILES: tuple[DirectFile, ...] = (
    DirectFile(
        name="melotts_rknn.py",
        url="https://huggingface.co/happyme531/MeloTTS-RKNN2/resolve/main/melotts_rknn.py?download=true",
        relative_path="melotts_rknn.py",
        min_size_bytes=512,
    ),
    DirectFile(
        name="utils.py",
        url="https://huggingface.co/happyme531/MeloTTS-RKNN2/resolve/main/utils.py?download=true",
        relative_path="utils.py",
        min_size_bytes=512,
    ),
    DirectFile(
        name="requirements.txt",
        url="https://huggingface.co/happyme531/MeloTTS-RKNN2/resolve/main/requirements.txt?download=true",
        relative_path="requirements.txt",
        min_size_bytes=64,
    ),
    DirectFile(
        name="encoder.onnx",
        url="https://huggingface.co/happyme531/MeloTTS-RKNN2/resolve/main/encoder.onnx?download=true",
        relative_path="encoder.onnx",
        min_size_bytes=1024 * 1024,
    ),
    DirectFile(
        name="decoder.rknn",
        url="https://huggingface.co/happyme531/MeloTTS-RKNN2/resolve/main/decoder.rknn?download=true",
        relative_path="decoder.rknn",
        min_size_bytes=1024 * 1024,
    ),
    DirectFile(
        name="g.bin",
        url="https://huggingface.co/happyme531/MeloTTS-RKNN2/resolve/main/g.bin?download=true",
        relative_path="g.bin",
        min_size_bytes=64,
    ),
    DirectFile(
        name="lexicon.txt",
        url="https://huggingface.co/happyme531/MeloTTS-RKNN2/resolve/main/lexicon.txt?download=true",
        relative_path="lexicon.txt",
        min_size_bytes=1024,
    ),
    DirectFile(
        name="tokens.txt",
        url="https://huggingface.co/happyme531/MeloTTS-RKNN2/resolve/main/tokens.txt?download=true",
        relative_path="tokens.txt",
        min_size_bytes=64,
    ),
)


# ---------------------------------------------------------------------------
# Template loaders
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_asr_template(relative_path: str) -> str:
    template_path = ASR_TEMPLATE_ROOT / Path(relative_path)
    try:
        return template_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing delivery template: {template_path}") from exc


@lru_cache(maxsize=None)
def load_tts_template(relative_path: str) -> str:
    template_path = TTS_TEMPLATE_ROOT / Path(relative_path)
    try:
        return template_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing delivery template: {template_path}") from exc


# ---------------------------------------------------------------------------
# Config file parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Option resolution helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Eagerly-loaded template content
# ---------------------------------------------------------------------------

ASR_RUNTIME_README = load_asr_template("runtime/README_SDK.md")
ASR_RUNTIME_RUN_SH = load_asr_template("runtime/run_asr.sh")
ASR_RUNTIME_SMOKETEST_SH = load_asr_template("runtime/smoketest.sh")
ASR_RUNTIME_CHECK_RKNN_ENV_SH = load_asr_template("runtime/tools/check_rknn_env.sh")
ASR_RUNTIME_BOARD_PROFILE_CAPABILITIES_SH = load_asr_template("runtime/tools/board_profile_capabilities.sh")
ASR_RUNTIME_PROFILE_INFERENCE_SH = load_asr_template("runtime/tools/profile_asr_inference.sh")

TTS_RUNTIME_README = load_tts_template("runtime/README_SDK.md")
TTS_RUNTIME_RUN_SH = load_tts_template("runtime/run_tts.sh")
TTS_RUNTIME_SMOKETEST_SH = load_tts_template("runtime/smoketest.sh")
TTS_RUNTIME_BOARD_PROFILE_CAPABILITIES_SH = load_tts_template("runtime/tools/board_profile_capabilities.sh")
TTS_RUNTIME_CHECK_PYTHON_ENV_SH = load_tts_template("runtime/tools/check_python_env.sh")
TTS_RUNTIME_INSTALL_PYTHON_DEPS_SH = load_tts_template("runtime/tools/install_python_deps.sh")
TTS_RUNTIME_PROFILE_INFERENCE_SH = load_tts_template("runtime/tools/profile_tts_inference.sh")
