from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .shared import fail


DELIVERY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = DELIVERY_ROOT / "templates" / "melotts_rknn2"
WORKSPACE_ROOT = DELIVERY_ROOT.parent.parent
DEFAULT_STAGE_DIR = WORKSPACE_ROOT / "artifacts" / "source-bundles" / "melotts_rknn2"
DEFAULT_RUNTIME_DIR = WORKSPACE_ROOT / "artifacts" / "runtime" / "melotts_rknn2_runtime"
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / "artifacts" / "cache" / "melotts_rknn2"
CONFIG_LOCAL_DIR = WORKSPACE_ROOT / "config" / "local"
BOARD_ENV_FILE = CONFIG_LOCAL_DIR / "board.local.env"
DELIVERY_ENV_FILE = CONFIG_LOCAL_DIR / "delivery.local.env"
DEFAULT_REMOTE_DIR = "/root/rkvoice/melotts_rknn2_runtime"
DEFAULT_TTS_TEXT = "你好，欢迎使用 RKVoice MeloTTS-RKNN2 离线语音服务。"

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


@dataclass(frozen=True)
class DirectFile:
    name: str
    url: str
    relative_path: str
    min_size_bytes: int = 1


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact(
        name=f"MeloTTS-RKNN2-{MELOTTS_REPO_REF}.tar.gz",
        url=f"https://github.com/happyme531/MeloTTS-RKNN2/archive/refs/heads/{MELOTTS_REPO_REF}.tar.gz",
        target_subdir="snapshot",
        strip_top_level=True,
        extracted_dir_name=SOURCE_ROOT_DIR_NAME,
    ),
)


DIRECT_FILES: tuple[DirectFile, ...] = (
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


RUNTIME_README = load_template("runtime/README_SDK.md")
RUNTIME_RUN_TTS_SH = load_template("runtime/run_tts.sh")
RUNTIME_SMOKETEST_SH = load_template("runtime/smoketest.sh")
RUNTIME_BOARD_PROFILE_CAPABILITIES_SH = load_template("runtime/tools/board_profile_capabilities.sh")
RUNTIME_CHECK_PYTHON_ENV_SH = load_template("runtime/tools/check_python_env.sh")
RUNTIME_INSTALL_PYTHON_DEPS_SH = load_template("runtime/tools/install_python_deps.sh")
RUNTIME_PROFILE_TTS_INFERENCE_SH = load_template("runtime/tools/profile_tts_inference.sh")