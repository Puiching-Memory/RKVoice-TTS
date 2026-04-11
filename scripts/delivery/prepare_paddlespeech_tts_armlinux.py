from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import paramiko


SCRIPT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_ROOT.parent.parent
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


OFFLINE_THIRD_PARTY_CMAKELISTS = textwrap.dedent(
    """\
    cmake_minimum_required(VERSION 3.10)
    project(tts_third_party_libs)

    include(ExternalProject)

    set(VENDOR_DIR ${CMAKE_CURRENT_LIST_DIR}/vendor)
    set(COMMON_CROSS_CMAKE_ARGS
        -DCMAKE_SYSTEM_NAME=Linux
        -DCMAKE_SYSTEM_PROCESSOR=aarch64
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    )

    macro(assert_vendor_dir name)
        if(NOT EXISTS ${VENDOR_DIR}/${name})
            message(FATAL_ERROR "Missing vendored dependency: ${VENDOR_DIR}/${name}")
        endif()
    endmacro()

    assert_vendor_dir(gflags)
    assert_vendor_dir(glog)
    assert_vendor_dir(abseil-cpp)
    assert_vendor_dir(cppjieba)
    assert_vendor_dir(limonp)

    ExternalProject_Add(gflags
        SOURCE_DIR      ${VENDOR_DIR}/gflags
        PREFIX          ${CMAKE_CURRENT_BINARY_DIR}
        INSTALL_DIR     ${CMAKE_CURRENT_BINARY_DIR}
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND   ""
        CMAKE_ARGS
            ${COMMON_CROSS_CMAKE_ARGS}
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBUILD_STATIC_LIBS=OFF
            -DBUILD_SHARED_LIBS=ON
            -DBUILD_TESTING=OFF
    )

    ExternalProject_Add(glog
        SOURCE_DIR      ${VENDOR_DIR}/glog
        PREFIX          ${CMAKE_CURRENT_BINARY_DIR}
        INSTALL_DIR     ${CMAKE_CURRENT_BINARY_DIR}
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND   ""
        CMAKE_ARGS
            ${COMMON_CROSS_CMAKE_ARGS}
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBUILD_TESTING=OFF
        DEPENDS         gflags
    )

    ExternalProject_Add(abseil
        SOURCE_DIR      ${VENDOR_DIR}/abseil-cpp
        PREFIX          ${CMAKE_CURRENT_BINARY_DIR}
        INSTALL_DIR     ${CMAKE_CURRENT_BINARY_DIR}
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND   ""
        CMAKE_ARGS
            ${COMMON_CROSS_CMAKE_ARGS}
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DABSL_PROPAGATE_CXX_STD=ON
            -DABSL_ENABLE_INSTALL=ON
            -DBUILD_TESTING=OFF
            -DABSL_BUILD_TESTING=OFF
    )

    ExternalProject_Add(cppjieba
        SOURCE_DIR       ${VENDOR_DIR}/cppjieba
        PREFIX           ${CMAKE_CURRENT_BINARY_DIR}
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND   ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
    )

    ExternalProject_Add(limonp
        SOURCE_DIR       ${VENDOR_DIR}/limonp
        PREFIX           ${CMAKE_CURRENT_BINARY_DIR}
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND   ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
    )
    """
)


OFFLINE_ENV_SH = textwrap.dedent(
    """\
    #!/bin/bash
    set -e

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export PATH="$SCRIPT_DIR/tools/cmake/bin:$PATH"

    THIRD_PARTY_LIB_DIR="$SCRIPT_DIR/src/TTSCppFrontend/third-party/build/lib"
    THIRD_PARTY_LIB64_DIR="$SCRIPT_DIR/src/TTSCppFrontend/third-party/build/lib64"
    PADDLE_LITE_LIB_DIR="$SCRIPT_DIR/libs/inference_lite_lib.armlinux.armv8.gcc.with_extra.with_cv/cxx/lib"

    export LD_LIBRARY_PATH="$PADDLE_LITE_LIB_DIR:$THIRD_PARTY_LIB_DIR:$THIRD_PARTY_LIB64_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    """
)


OFFLINE_BUILD_SH = textwrap.dedent(
    """\
    #!/bin/bash
    set -euo pipefail

    cd "$(dirname "$(realpath "$0")")"
    . ./offline_env.sh
    ./build.sh "$@"
    """
)


OFFLINE_RUN_SH = textwrap.dedent(
    """\
    #!/bin/bash
    set -euo pipefail

    cd "$(dirname "$(realpath "$0")")"
    . ./offline_env.sh
    ./run.sh "$@"
    """
)


OFFLINE_SMOKETEST_SH = textwrap.dedent(
    """\
    #!/bin/bash
    set -euo pipefail

    cd "$(dirname "$(realpath "$0")")"
    . ./offline_env.sh

    sentence="${1:-你好，欢迎使用离线语音合成服务。}"
    output_wav="${2:-./output/smoke_test.wav}"

    ./build.sh
    ./run.sh --sentence "$sentence" --output_wav "$output_wav"
    echo "Smoke test WAV saved to: $output_wav"
    """
)


ROOT_BUILD_DEPENDS_SH = textwrap.dedent(
    """\
    #!/bin/bash
    set -euo pipefail

    cd "$(dirname "$(realpath "$0")")"
    ./src/TTSCppFrontend/build-depends.sh "$@"
    """
)


RUNTIME_RUN_SH = textwrap.dedent(
    """\
    #!/bin/bash
    set -euo pipefail

    SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
    cd "$SCRIPT_DIR"

    export LD_LIBRARY_PATH="$SCRIPT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    mkdir -p "$SCRIPT_DIR/output"

    exec "$SCRIPT_DIR/bin/paddlespeech_tts_demo" \
        --front_conf "$SCRIPT_DIR/front.conf" \
        --acoustic_model "$SCRIPT_DIR/models/cpu/fastspeech2_csmsc_arm.nb" \
        --vocoder "$SCRIPT_DIR/models/cpu/mb_melgan_csmsc_arm.nb" \
        "$@"
    """
)


RUNTIME_SMOKETEST_SH = textwrap.dedent(
    """\
    #!/bin/bash
    set -euo pipefail

    SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
    cd "$SCRIPT_DIR"

    sentence="${1:-}"
    output_wav="${2:-./output/smoke_test.wav}"

    if [ -z "$sentence" ]; then
        echo "Missing sentence argument"
        exit 1
    fi

    ./run_tts.sh --sentence "$sentence" --output_wav "$output_wav"
    ls -lh "$output_wav"
    echo "Smoke test WAV saved to: $output_wav"
    """
)


def log(message: str) -> None:
    print(message, flush=True)



def fail(message: str, exit_code: int = 1) -> None:
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(exit_code)



def md5sum(file_path: Path) -> str:
    digest = hashlib.md5()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)



def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8", newline="\n")



def download_file(url: str, destination: Path) -> None:
    ensure_parent(destination)
    tmp_destination = destination.with_suffix(destination.suffix + ".part")
    log(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, tmp_destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    tmp_destination.replace(destination)



def download_artifact(cache_dir: Path, artifact: Artifact) -> Path:
    destination = cache_dir / artifact.name
    if destination.exists() and artifact.md5:
        if md5sum(destination) == artifact.md5:
            log(f"Using cached artifact {artifact.name}")
            return destination
        destination.unlink()
    elif destination.exists():
        log(f"Using cached artifact {artifact.name}")
        return destination

    download_file(artifact.url, destination)
    if artifact.md5 and md5sum(destination) != artifact.md5:
        destination.unlink(missing_ok=True)
        fail(f"MD5 mismatch for {artifact.name}")
    return destination



def extract_tar_members(archive_path: Path, prefix: str, destination: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.name.startswith(prefix):
                continue
            relative_name = member.name[len(prefix):].lstrip("/")
            if not relative_name:
                continue
            target_path = destination / relative_name
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym() or member.islnk():
                continue
            file_object = archive.extractfile(member)
            if file_object is None:
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("wb") as output:
                shutil.copyfileobj(file_object, output)



def extract_tarball(
    archive_path: Path,
    destination: Path,
    *,
    strip_top_level: bool = False,
    extracted_dir_name: str | None = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if not strip_top_level:
            archive.extractall(destination)
            return

        top_level_names = []
        for member in members:
            first_part = Path(member.name).parts[0] if Path(member.name).parts else ""
            if first_part and first_part not in top_level_names:
                top_level_names.append(first_part)
        if len(top_level_names) != 1:
            fail(f"Archive {archive_path.name} does not have a single top-level directory")

        top_level = top_level_names[0]
        root_destination = destination / (extracted_dir_name or top_level)
        root_destination.mkdir(parents=True, exist_ok=True)

        prefix = f"{top_level}/"
        for member in members:
            if member.name == top_level:
                continue
            if not member.name.startswith(prefix):
                continue
            relative_name = member.name[len(prefix):]
            if not relative_name:
                continue
            target_path = root_destination / relative_name
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym() or member.islnk():
                continue
            file_object = archive.extractfile(member)
            if file_object is None:
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("wb") as output:
                shutil.copyfileobj(file_object, output)



def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)



def materialize_repo_sources(stage_dir: Path, cache_dir: Path) -> None:
    repo_archive = cache_dir / "PaddleSpeech-develop.tar.gz"
    if not repo_archive.exists():
        download_file(PADDLESPEECH_ARCHIVE_URL, repo_archive)

    extract_tar_members(repo_archive, f"{PADDLESPEECH_ARCHIVE_ROOT}/{PADDLESPEECH_TTS_SUBTREE}", stage_dir)
    extract_tar_members(
        repo_archive,
        f"{PADDLESPEECH_ARCHIVE_ROOT}/{PADDLESPEECH_FRONTEND_SUBTREE}",
        stage_dir / "src" / "TTSCppFrontend",
    )



def populate_artifacts(stage_dir: Path, cache_dir: Path) -> None:
    for artifact in ARTIFACTS:
        archive_path = download_artifact(cache_dir, artifact)
        extract_target = stage_dir / artifact.target_subdir
        log(f"Extracting {artifact.name}")
        extract_tarball(
            archive_path,
            extract_target,
            strip_top_level=artifact.strip_top_level,
            extracted_dir_name=artifact.extracted_dir_name,
        )



def patch_for_offline_build(stage_dir: Path) -> None:
    third_party_cmakelists = stage_dir / "src" / "TTSCppFrontend" / "third-party" / "CMakeLists.txt"
    write_text(third_party_cmakelists, OFFLINE_THIRD_PARTY_CMAKELISTS)

    frontend_cmakelists = stage_dir / "src" / "TTSCppFrontend" / "CMakeLists.txt"
    frontend_cmakelists_text = frontend_cmakelists.read_text(encoding="utf-8")
    frontend_cmakelists_text = frontend_cmakelists_text.replace(
        'set(ENV{PKG_CONFIG_PATH} "${CMAKE_SOURCE_DIR}/third-party/build/lib/pkgconfig:${CMAKE_SOURCE_DIR}/third-party/build/lib64/pkgconfig")',
        'set(ENV{PKG_CONFIG_PATH} "${CMAKE_CURRENT_LIST_DIR}/third-party/build/lib/pkgconfig:${CMAKE_CURRENT_LIST_DIR}/third-party/build/lib64/pkgconfig")',
    )
    frontend_cmakelists_text = frontend_cmakelists_text.replace(
        '    ${CMAKE_SOURCE_DIR}/third-party/build/src/cppjieba/include\n    ${CMAKE_SOURCE_DIR}/third-party/build/src/limonp/include\n',
        '    ${CMAKE_CURRENT_LIST_DIR}/third-party/vendor/cppjieba/include\n    ${CMAKE_CURRENT_LIST_DIR}/third-party/vendor/limonp/include\n',
    )
    frontend_cmakelists_text = frontend_cmakelists_text.replace(
        '    ${CMAKE_CURRENT_LIST_DIR}/third-party/build/src/cppjieba/include\n    ${CMAKE_CURRENT_LIST_DIR}/third-party/build/src/limonp/include\n',
        '    ${CMAKE_CURRENT_LIST_DIR}/third-party/vendor/cppjieba/include\n    ${CMAKE_CURRENT_LIST_DIR}/third-party/vendor/limonp/include\n',
    )
    write_text(frontend_cmakelists, frontend_cmakelists_text)

    root_cmakelists = stage_dir / "src" / "CMakeLists.txt"
    root_cmakelists_text = root_cmakelists.read_text(encoding="utf-8")
    root_cmakelists_text = root_cmakelists_text.replace(
        "    third-party/build/src/cppjieba/include\n    third-party/build/src/limonp/include\n",
        "    TTSCppFrontend/third-party/vendor/cppjieba/include\n    TTSCppFrontend/third-party/vendor/limonp/include\n",
    )
    root_cmakelists_text = root_cmakelists_text.replace(
        "    TTSCppFrontend/third-party/build/src/cppjieba/include\n    TTSCppFrontend/third-party/build/src/limonp/include\n",
        "    TTSCppFrontend/third-party/vendor/cppjieba/include\n    TTSCppFrontend/third-party/vendor/limonp/include\n",
    )
    write_text(root_cmakelists, root_cmakelists_text)

    dict_source = stage_dir / "dict"
    dict_target = stage_dir / "src" / "TTSCppFrontend" / "front_demo" / "dict"
    copy_tree(dict_source, dict_target)

    write_text(stage_dir / "offline_env.sh", OFFLINE_ENV_SH)
    write_text(stage_dir / "offline_build.sh", OFFLINE_BUILD_SH)
    write_text(stage_dir / "offline_run.sh", OFFLINE_RUN_SH)
    write_text(stage_dir / "offline_smoketest.sh", OFFLINE_SMOKETEST_SH)
    write_text(stage_dir / "build-depends.sh", ROOT_BUILD_DEPENDS_SH)



def prepare_source_bundle(stage_dir: Path, *, force: bool = False) -> Path:
    cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    if force and stage_dir.exists():
        shutil.rmtree(stage_dir)

    if not stage_dir.exists():
        stage_dir.mkdir(parents=True, exist_ok=True)
        materialize_repo_sources(stage_dir, cache_dir)
        populate_artifacts(stage_dir, cache_dir)
    else:
        log(f"Reusing existing source bundle: {stage_dir}")

    patch_for_offline_build(stage_dir)
    return stage_dir



def run_local_command(command: list[str], *, timeout: int, cwd: Path | None = None) -> None:
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    assert process.stdout is not None
    start_time = time.time()
    try:
        while True:
            line = process.stdout.readline()
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
            elif process.poll() is not None:
                break
            if timeout and time.time() - start_time > timeout:
                process.kill()
                fail(f"Local command timed out: {' '.join(command)}")
        remaining_output = process.stdout.read()
        if remaining_output:
            sys.stdout.write(remaining_output)
            sys.stdout.flush()
    finally:
        process.stdout.close()

    if process.returncode != 0:
        fail(f"Local command failed with exit code {process.returncode}: {' '.join(command)}")



def docker_mount_path(path: Path) -> str:
    return path.resolve().as_posix()


def sanitize_docker_tag_part(value: str) -> str:
    sanitized = value.lower()
    for old, new in (("/", "-"), (":", "-"), ("@", "-")):
        sanitized = sanitized.replace(old, new)
    return sanitized


def get_docker_builder_image_name(base_image: str, docker_platform: str, builder_image: str) -> str:
    if builder_image:
        return builder_image
    base_part = sanitize_docker_tag_part(base_image)
    platform_part = sanitize_docker_tag_part(docker_platform)
    return f"paddlespeech-tts-builder:{base_part}-{platform_part}"


def ensure_docker_builder_image(
    *,
    base_image: str,
    builder_image: str,
    docker_platform: str,
    docker_timeout: int,
) -> str:
    resolved_builder_image = get_docker_builder_image_name(base_image, docker_platform, builder_image)
    inspect = subprocess.run(
        ["docker", "image", "inspect", resolved_builder_image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if inspect.returncode == 0:
        log(f"Reusing local Docker builder image: {resolved_builder_image}")
        return resolved_builder_image

    log(f"Building local Docker builder image: {resolved_builder_image}")
    dockerfile = textwrap.dedent(
        f"""\
        FROM {base_image}
        ENV DEBIAN_FRONTEND=noninteractive
        RUN apt-get update \
            && apt-get install -y --no-install-recommends {DOCKER_BUILD_PACKAGES} \
            && rm -rf /var/lib/apt/lists/*
        """
    )
    with tempfile.TemporaryDirectory(prefix="paddlespeech_tts_builder_") as temp_dir:
        dockerfile_path = Path(temp_dir) / "Dockerfile"
        write_text(dockerfile_path, dockerfile)
        command = [
            "docker",
            "build",
            "--platform",
            docker_platform,
            "-t",
            resolved_builder_image,
            temp_dir,
        ]
        run_local_command(command, timeout=docker_timeout)

    return resolved_builder_image


def build_dir_needs_reset(build_dir: Path) -> bool:
    if not build_dir.exists():
        return False

    cache_file = build_dir / "CMakeCache.txt"
    if not cache_file.exists():
        return True

    cache_text = cache_file.read_text(encoding="utf-8", errors="ignore")
    return "aarch64-linux-gnu-gcc" not in cache_text or "aarch64-linux-gnu-g++" not in cache_text


def reset_incompatible_build_dirs(stage_dir: Path) -> None:
    build_dirs = (
        stage_dir / "build",
        stage_dir / "src" / "TTSCppFrontend" / "third-party" / "build",
    )
    for build_dir in build_dirs:
        if build_dir_needs_reset(build_dir):
            log(f"Removing incompatible build cache: {build_dir}")
            shutil.rmtree(build_dir)



def build_runtime_bundle_with_docker(
    stage_dir: Path,
    runtime_dir: Path,
    *,
    force: bool,
    docker_image: str,
    docker_builder_image: str,
    docker_platform: str,
    docker_timeout: int,
) -> Path:
    binary_path = runtime_dir / "bin" / "paddlespeech_tts_demo"
    if force and runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    if binary_path.exists():
        log(f"Reusing existing runtime bundle: {runtime_dir}")
        return runtime_dir

    runtime_dir.mkdir(parents=True, exist_ok=True)
    reset_incompatible_build_dirs(stage_dir)
    builder_image = ensure_docker_builder_image(
        base_image=docker_image,
        builder_image=docker_builder_image,
        docker_platform=docker_platform,
        docker_timeout=docker_timeout,
    )

    docker_script = "\n".join(
        [
            "set -euo pipefail",
            "export CC=aarch64-linux-gnu-gcc",
            "export CXX=aarch64-linux-gnu-g++",
            "cd /workspace/src",
            "./build.sh",
            "rm -rf /workspace/out/*",
            "mkdir -p /workspace/out/bin /workspace/out/lib /workspace/out/models /workspace/out/output /workspace/out/dict",
            "cp ./build/paddlespeech_tts_demo /workspace/out/bin/",
            "cp ./front.conf /workspace/out/front.conf",
            "cp -r ./models/. /workspace/out/models/",
            "cp -r ./src/TTSCppFrontend/front_demo/dict/. /workspace/out/dict/",
            "copy_shared() {",
            "  src_dir=\"$1\"",
            "  if [ ! -d \"$src_dir\" ]; then",
            "    return 0",
            "  fi",
            "  find \"$src_dir\" -maxdepth 1 \\( -type f -o -type l \\) \\( -name 'libpaddle*.so*' -o -name 'libgflags*.so*' -o -name 'libglog*.so*' -o -name 'libabsl_*.so*' \\) -print0 | while IFS= read -r -d '' item; do",
            "    cp -L \"$item\" \"/workspace/out/lib/$(basename \"$item\")\"",
            "  done",
            "}",
            f"copy_shared \"{PADDLE_LITE_LIB_DIR}\"",
            "copy_shared ./src/TTSCppFrontend/third-party/build/lib",
            "copy_shared ./src/TTSCppFrontend/third-party/build/lib64",
            "cat > /workspace/out/run_tts.sh <<'EOF'",
            RUNTIME_RUN_SH.rstrip(),
            "EOF",
            "cat > /workspace/out/smoketest.sh <<'EOF'",
            RUNTIME_SMOKETEST_SH.rstrip(),
            "EOF",
            "chmod +x /workspace/out/run_tts.sh /workspace/out/smoketest.sh /workspace/out/bin/paddlespeech_tts_demo",
            "file /workspace/out/bin/paddlespeech_tts_demo",
            "ls -lah /workspace/out/lib",
        ]
    )

    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        docker_platform,
        "--mount",
        f"type=bind,source={docker_mount_path(stage_dir)},target=/workspace/src",
        "--mount",
        f"type=bind,source={docker_mount_path(runtime_dir)},target=/workspace/out",
        builder_image,
        "bash",
        "-lc",
        docker_script,
    ]
    run_local_command(command, timeout=docker_timeout)

    if not binary_path.exists():
        fail(f"Docker build finished but binary was not produced: {binary_path}")
    log(f"Runtime bundle prepared at {runtime_dir}")
    return runtime_dir



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
            sys.stdout.write(chunk)
            sys.stdout.flush()
            output_parts.append(chunk)
        if channel.recv_stderr_ready():
            chunk = channel.recv_stderr(4096).decode("utf-8", "ignore")
            sys.stdout.write(chunk)
            sys.stdout.flush()
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
    sentence: str,
    skip_smoketest: bool,
) -> None:
    if not (runtime_dir / "bin" / "paddlespeech_tts_demo").exists():
        fail(f"Runtime bundle is missing binary: {runtime_dir / 'bin' / 'paddlespeech_tts_demo'}")

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
            f"find {sh_quote(remote_dir)} -type f -name '*.sh' -exec chmod +x {{}} + && chmod +x {sh_quote(remote_dir)}/bin/paddlespeech_tts_demo",
            timeout=remote_timeout,
        )
        if not skip_smoketest:
            log("Running remote smoke test")
            remote_output_wav = "./output/smoke_test.wav"
            remote_output_log = "./output/smoke_test.log"
            smoke_test_command = (
                "set -o pipefail; "
                "mkdir -p ./output; "
                f"./smoketest.sh {sh_quote(sentence)} {sh_quote(remote_output_wav)} 2>&1 | tee {sh_quote(remote_output_log)}"
            )
            smoke_test_output = run_remote_command(
                client,
                f"cd {sh_quote(remote_dir)} && bash -lc {sh_quote(smoke_test_command)}",
                timeout=remote_timeout,
            )
            local_log_path = local_output_dir / "smoke_test.log"
            write_text(local_log_path, smoke_test_output)
            remote_output_dir = f"{remote_dir.rstrip('/')}/output"
            local_wav_path = local_output_dir / "smoke_test.wav"
            download_file(client, f"{remote_output_dir}/smoke_test.wav", local_wav_path)
            download_file(client, f"{remote_output_dir}/smoke_test.log", local_log_path)
            log(f"Downloaded smoke test audio to {local_wav_path}")
            log(f"Downloaded smoke test log to {local_log_path}")
    finally:
        client.close()



def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, build, and deploy PaddleSpeech TTSArmLinux runtime bundle")
    subparsers = parser.add_subparsers(dest="action", required=True)

    def add_source_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--stage-dir", type=Path, default=None, help="Local source bundle directory")
        subparser.add_argument("--force", action="store_true", help="Recreate local bundle directories")

    def add_build_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--runtime-dir", type=Path, default=None, help="Local runtime bundle directory")
        subparser.add_argument("--docker-image", default=None, help="Base Docker image used to create the cached build image")
        subparser.add_argument("--docker-builder-image", default=None, help="Optional cached Docker builder image tag")
        subparser.add_argument("--docker-platform", default=None, help="Docker platform used for build")
        subparser.add_argument("--docker-timeout", type=int, default=None, help="Docker build timeout in seconds")

    def add_remote_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--host", default=None, help="Board SSH host")
        subparser.add_argument("--username", default=None, help="Board SSH username")
        subparser.add_argument("--password", default=None, help="Board SSH password")
        subparser.add_argument("--source-ip", default=None, help="Optional local source IP used to reach the board")
        subparser.add_argument("--remote-dir", default=None, help="Remote runtime directory on the board")
        subparser.add_argument("--ssh-timeout", type=int, default=None, help="SSH connect timeout in seconds")
        subparser.add_argument("--remote-timeout", type=int, default=None, help="Remote command timeout in seconds")
        subparser.add_argument("--sentence", default=None, help="Smoke test sentence")
        subparser.add_argument("--skip-smoketest", action="store_true", help="Skip remote smoke test")

    download_parser = subparsers.add_parser("download", help="Prepare the local source bundle")
    add_source_args(download_parser)

    build_parser = subparsers.add_parser("build", help="Prepare source bundle and build ARM64 runtime with Docker")
    add_source_args(build_parser)
    add_build_args(build_parser)

    upload_parser = subparsers.add_parser("upload", help="Upload an existing runtime bundle to the board")
    upload_parser.add_argument("--runtime-dir", type=Path, default=None, help="Local runtime bundle directory")
    add_remote_args(upload_parser)

    all_parser = subparsers.add_parser("all", help="Prepare source bundle, build ARM64 runtime, upload, and smoke test")
    add_source_args(all_parser)
    add_build_args(all_parser)
    add_remote_args(all_parser)
    all_parser.add_argument("--skip-build", action="store_true", help="Reuse an existing runtime bundle and skip local Docker build")

    return parser.parse_args()



def main() -> None:
    args = parse_args()
    local_settings = load_local_settings()

    stage_dir = resolve_path_option(
        getattr(args, "stage_dir", None),
        env_names=("TTS_STAGE_DIR",),
        local_settings=local_settings,
        default=DEFAULT_STAGE_DIR,
    )
    runtime_dir = resolve_path_option(
        getattr(args, "runtime_dir", None),
        env_names=("TTS_RUNTIME_DIR",),
        local_settings=local_settings,
        default=DEFAULT_RUNTIME_DIR,
    )

    if args.action in {"download", "build", "all"}:
        stage_dir = prepare_source_bundle(stage_dir, force=getattr(args, "force", False))
        log(f"Source bundle prepared at {stage_dir}")

    if args.action in {"build", "all"} and not getattr(args, "skip_build", False):
        docker_image = resolve_text_option(
            args.docker_image,
            env_names=("TTS_DOCKER_IMAGE",),
            local_settings=local_settings,
            default=DEFAULT_DOCKER_IMAGE,
        )
        docker_builder_image = resolve_text_option(
            args.docker_builder_image,
            env_names=("TTS_DOCKER_BUILDER_IMAGE",),
            local_settings=local_settings,
            default=DEFAULT_DOCKER_BUILDER_IMAGE,
        )
        docker_platform = resolve_text_option(
            args.docker_platform,
            env_names=("TTS_DOCKER_PLATFORM",),
            local_settings=local_settings,
            default=DEFAULT_DOCKER_PLATFORM,
        )
        docker_timeout = resolve_int_option(
            args.docker_timeout,
            env_names=("TTS_DOCKER_TIMEOUT",),
            local_settings=local_settings,
            default=7200,
        )
        runtime_dir = build_runtime_bundle_with_docker(
            stage_dir,
            runtime_dir,
            force=getattr(args, "force", False),
            docker_image=docker_image,
            docker_builder_image=docker_builder_image or "",
            docker_platform=docker_platform,
            docker_timeout=docker_timeout,
        )

    if args.action == "upload":
        if not (runtime_dir / "bin" / "paddlespeech_tts_demo").exists():
            fail(f"Runtime bundle does not exist: {runtime_dir}")

    if args.action == "all" and getattr(args, "skip_build", False):
        if not (runtime_dir / "bin" / "paddlespeech_tts_demo").exists():
            fail(f"Requested --skip-build but runtime bundle is missing: {runtime_dir}")

    if args.action in {"upload", "all"}:
        host = resolve_required_text_option(
            args.host,
            env_names=("TTS_BOARD_HOST",),
            local_settings=local_settings,
            option_name="board host",
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
        remote_dir = resolve_text_option(
            args.remote_dir,
            env_names=("TTS_REMOTE_DIR",),
            local_settings=local_settings,
            default=DEFAULT_REMOTE_DIR,
        )
        ssh_timeout = resolve_int_option(
            args.ssh_timeout,
            env_names=("TTS_SSH_TIMEOUT",),
            local_settings=local_settings,
            default=8,
        )
        remote_timeout = resolve_int_option(
            args.remote_timeout,
            env_names=("TTS_REMOTE_TIMEOUT",),
            local_settings=local_settings,
            default=1800,
        )
        sentence = resolve_text_option(
            args.sentence,
            env_names=("TTS_SMOKE_TEST_SENTENCE", "TTS_SENTENCE"),
            local_settings=local_settings,
            default=DEFAULT_SENTENCE,
        )
        source_ip_value = resolve_text_option(
            args.source_ip,
            env_names=("TTS_SOURCE_IP",),
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
            sentence=sentence,
            skip_smoketest=args.skip_smoketest,
        )
        log(f"Runtime bundle deployed to {remote_dir}")


if __name__ == "__main__":
    main()
