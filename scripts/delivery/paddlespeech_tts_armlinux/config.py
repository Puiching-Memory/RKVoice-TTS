from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .shared import fail


DELIVERY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = DELIVERY_ROOT / "templates" / "paddlespeech_tts_armlinux"
WORKSPACE_ROOT = DELIVERY_ROOT.parent.parent
DEFAULT_STAGE_DIR = WORKSPACE_ROOT / "artifacts" / "source-bundles" / "paddlespeech_tts_armlinux_offline"
DEFAULT_RUNTIME_DIR = WORKSPACE_ROOT / "artifacts" / "runtime" / "paddlespeech_tts_armlinux_runtime"
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / "artifacts" / "cache" / "paddlespeech_tts_armlinux"
CONFIG_LOCAL_DIR = WORKSPACE_ROOT / "config" / "local"
BOARD_ENV_FILE = CONFIG_LOCAL_DIR / "board.local.env"
DELIVERY_ENV_FILE = CONFIG_LOCAL_DIR / "delivery.local.env"
DEFAULT_REMOTE_DIR = "/root/tts/paddlespeech_tts_armlinux_runtime"
DEFAULT_SENTENCE = "你好，欢迎使用离线语音合成服务。"
DEFAULT_DOCKER_IMAGE = "ubuntu:22.04"
DEFAULT_DOCKER_PLATFORM = "linux/amd64"
DEFAULT_DOCKER_BUILDER_IMAGE = ""
PADDLESPEECH_ARCHIVE_URL = "https://github.com/PaddlePaddle/PaddleSpeech/archive/refs/heads/develop.tar.gz"
PADDLESPEECH_ARCHIVE_ROOT = "PaddleSpeech-develop"
PADDLESPEECH_TTS_SUBTREE = "demos/TTSArmLinux"
PADDLESPEECH_FRONTEND_SUBTREE = "demos/TTSCppFrontend"
PADDLE_LITE_LIB_DIR = "./libs/inference_lite_lib.armlinux.armv8.gcc.with_extra.with_cv/cxx/lib"
DOCKER_BUILD_PACKAGES = "build-essential cmake pkg-config ca-certificates file gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"


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
    md5: str | None = None
    strip_top_level: bool = False
    extracted_dir_name: str | None = None


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact(
        name="inference_lite_lib.armlinux.armv8.gcc.with_extra.with_cv.tar.gz",
        url="https://paddlespeech.cdn.bcebos.com/demos/TTSArmLinux/inference_lite_lib.armlinux.armv8.gcc.with_extra.with_cv.tar.gz",
        target_subdir="libs",
        md5="39e0c6604f97c70f5d13c573d7e709b9",
    ),
    Artifact(
        name="fs2cnn_mbmelgan_cpu_v1.3.0.tar.gz",
        url="https://paddlespeech.cdn.bcebos.com/demos/TTSAndroid/fs2cnn_mbmelgan_cpu_v1.3.0.tar.gz",
        target_subdir="models",
        md5="93ef17d44b498aff3bea93e2c5c09a1e",
    ),
    Artifact(
        name="fastspeech2_nosil_baker_ckpt_0.4.tar.gz",
        url="https://paddlespeech.cdn.bcebos.com/t2s/text_frontend/fastspeech2_nosil_baker_ckpt_0.4.tar.gz",
        target_subdir="dict",
        md5="7bf1bab1737375fa123c413eb429c573",
    ),
    Artifact(
        name="speedyspeech_nosil_baker_ckpt_0.5.tar.gz",
        url="https://paddlespeech.cdn.bcebos.com/t2s/text_frontend/speedyspeech_nosil_baker_ckpt_0.5.tar.gz",
        target_subdir="dict",
        md5="0b7754b21f324789aef469c61f4d5b8f",
    ),
    Artifact(
        name="jieba.tar.gz",
        url="https://paddlespeech.cdn.bcebos.com/t2s/text_frontend/jieba.tar.gz",
        target_subdir="dict",
        md5="6d30f426bd8c0025110a483f051315ca",
    ),
    Artifact(
        name="tranditional_to_simplified.tar.gz",
        url="https://paddlespeech.cdn.bcebos.com/t2s/text_frontend/tranditional_to_simplified.tar.gz",
        target_subdir="dict",
        md5="258f5b59d5ebfe96d02007ca1d274a7f",
    ),
    Artifact(
        name="cmake-3.31.6-linux-aarch64.tar.gz",
        url="https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-aarch64.tar.gz",
        target_subdir="tools",
        strip_top_level=True,
        extracted_dir_name="cmake",
    ),
    Artifact(
        name="gflags-v2.2.2.tar.gz",
        url="https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.tar.gz",
        target_subdir="src/TTSCppFrontend/third-party/vendor",
        strip_top_level=True,
        extracted_dir_name="gflags",
    ),
    Artifact(
        name="glog-v0.6.0.tar.gz",
        url="https://github.com/google/glog/archive/refs/tags/v0.6.0.tar.gz",
        target_subdir="src/TTSCppFrontend/third-party/vendor",
        strip_top_level=True,
        extracted_dir_name="glog",
    ),
    Artifact(
        name="abseil-cpp-20230125.1.tar.gz",
        url="https://github.com/abseil/abseil-cpp/archive/refs/tags/20230125.1.tar.gz",
        target_subdir="src/TTSCppFrontend/third-party/vendor",
        strip_top_level=True,
        extracted_dir_name="abseil-cpp",
    ),
    Artifact(
        name="cppjieba-v5.0.3.tar.gz",
        url="https://github.com/yanyiwu/cppjieba/archive/refs/tags/v5.0.3.tar.gz",
        target_subdir="src/TTSCppFrontend/third-party/vendor",
        strip_top_level=True,
        extracted_dir_name="cppjieba",
    ),
    Artifact(
        name="limonp-v0.6.6.tar.gz",
        url="https://github.com/yanyiwu/limonp/archive/refs/tags/v0.6.6.tar.gz",
        target_subdir="src/TTSCppFrontend/third-party/vendor",
        strip_top_level=True,
        extracted_dir_name="limonp",
    ),
)


OFFLINE_THIRD_PARTY_CMAKELISTS = load_template("offline/third-party/CMakeLists.txt")
OFFLINE_ENV_SH = load_template("offline/offline_env.sh")
OFFLINE_BUILD_SH = load_template("offline/offline_build.sh")
OFFLINE_RUN_SH = load_template("offline/offline_run.sh")
OFFLINE_SMOKETEST_SH = load_template("offline/offline_smoketest.sh")
RKVOICE_TTS_PUBLIC_HEADER = load_template("sdk/rkvoice_tts_api.h")
RKVOICE_TTS_CORE_CC = load_template("sdk/rkvoice_tts_core.cc")
RKVOICE_TTS_DEMO_MAIN_CC = load_template("sdk/rkvoice_tts_demo_main.cc")
RKVOICE_TTS_CMAKELISTS = load_template("sdk/CMakeLists.txt")
RUNTIME_SDK_README = load_template("runtime/README_SDK.md")
RUNTIME_C_API_DEMO_SOURCE = load_template("runtime/examples/c_api_demo.c")
ROOT_BUILD_DEPENDS_SH = load_template("offline/build-depends.sh")
RUNTIME_RUN_SH = load_template("runtime/run_tts.sh")
RUNTIME_SMOKETEST_SH = load_template("runtime/smoketest.sh")
RUNTIME_PROFILE_CAPABILITIES_SH = load_template("runtime/tools/board_profile_capabilities.sh")
RUNTIME_PROFILE_TTS_INFERENCE_SH = load_template("runtime/tools/profile_tts_inference.sh")
