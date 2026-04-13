from __future__ import annotations

import importlib.util
import subprocess
import shutil
import sys
from pathlib import Path

from .config import (
    ARTIFACTS,
    BOARD_PYTHON_ABI,
    BOARD_PYTHON_VERSION,
    BOARD_WHEEL_PLATFORM,
    COMMON_PYTHON_DEPENDENCIES,
    DEFAULT_CACHE_DIR,
    DIRECT_FILES,
    REQUIRED_WHEEL_PATTERNS,
    RUNTIME_BOARD_PROFILE_CAPABILITIES_SH,
    RUNTIME_CHECK_PYTHON_ENV_SH,
    RUNTIME_INSTALL_PYTHON_DEPS_SH,
    RUNTIME_PROFILE_TTS_INFERENCE_SH,
    RUNTIME_README,
    RKNN_LITE_DEPENDENCY,
    RUNTIME_RUN_TTS_SH,
    RUNTIME_SMOKETEST_SH,
    SOURCE_ROOT_RELATIVE_PATH,
    WHEELHOUSE_RELATIVE_PATH,
    Artifact,
    DirectFile,
)
from .shared import download_http_file, extract_tarball, fail, log, write_text


def download_artifact(cache_dir: Path, artifact: Artifact) -> Path:
    destination = cache_dir / artifact.name
    if destination.exists():
        log(f"Using cached artifact {artifact.name}")
        return destination
    download_http_file(artifact.url, destination)
    return destination


def download_direct_file(cache_dir: Path, direct_file: DirectFile) -> Path:
    destination = cache_dir / direct_file.name
    if destination.exists() and destination.stat().st_size >= direct_file.min_size_bytes:
        log(f"Using cached file {direct_file.name}")
        return destination
    download_http_file(direct_file.url, destination)
    if destination.stat().st_size < direct_file.min_size_bytes:
        fail(f"Downloaded file is unexpectedly small: {destination}")
    return destination


def artifact_output_path(stage_dir: Path, artifact: Artifact) -> Path:
    base_path = stage_dir / artifact.target_subdir
    if artifact.strip_top_level:
        return base_path / (artifact.extracted_dir_name or "")
    return base_path


def source_root(stage_dir: Path) -> Path:
    return stage_dir / SOURCE_ROOT_RELATIVE_PATH


def stage_wheelhouse_dir(stage_dir: Path) -> Path:
    return stage_dir / WHEELHOUSE_RELATIVE_PATH


def cache_wheelhouse_dir(cache_dir: Path) -> Path:
    return cache_dir / WHEELHOUSE_RELATIVE_PATH


def required_wheels_present(wheelhouse_dir: Path) -> bool:
    if not wheelhouse_dir.exists():
        return False
    return all(any(wheelhouse_dir.glob(pattern)) for pattern in REQUIRED_WHEEL_PATTERNS)


def ensure_local_pip_available() -> None:
    if importlib.util.find_spec("pip") is not None:
        return
    log("pip is missing in the local environment; bootstrapping it with ensurepip")
    subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)


def download_python_wheels(cache_dir: Path) -> None:
    wheelhouse_dir = cache_wheelhouse_dir(cache_dir)
    if required_wheels_present(wheelhouse_dir):
        log(f"Using cached Python wheelhouse: {wheelhouse_dir}")
        return

    wheelhouse_dir.mkdir(parents=True, exist_ok=True)
    ensure_local_pip_available()

    base_command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--dest",
        str(wheelhouse_dir),
        "--only-binary=:all:",
        "--platform",
        BOARD_WHEEL_PLATFORM,
        "--implementation",
        "cp",
        "--python-version",
        BOARD_PYTHON_VERSION,
        "--abi",
        BOARD_PYTHON_ABI,
    ]

    log("Downloading offline Python wheels for MeloTTS-RKNN2")
    subprocess.run([*base_command, *COMMON_PYTHON_DEPENDENCIES], check=True)
    subprocess.run([*base_command, "--no-deps", RKNN_LITE_DEPENDENCY], check=True)

    if not required_wheels_present(wheelhouse_dir):
        fail(f"Python wheelhouse is incomplete after download: {wheelhouse_dir}")


def materialize_runtime_support_files(runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    tools_dir = runtime_dir / "tools"
    output_dir = runtime_dir / "output"
    bin_dir = runtime_dir / "bin"
    tools_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    write_text(runtime_dir / "README_SDK.md", RUNTIME_README)
    write_text(runtime_dir / "run_tts.sh", RUNTIME_RUN_TTS_SH)
    write_text(runtime_dir / "smoketest.sh", RUNTIME_SMOKETEST_SH)
    write_text(tools_dir / "board_profile_capabilities.sh", RUNTIME_BOARD_PROFILE_CAPABILITIES_SH)
    write_text(tools_dir / "check_python_env.sh", RUNTIME_CHECK_PYTHON_ENV_SH)
    write_text(tools_dir / "install_python_deps.sh", RUNTIME_INSTALL_PYTHON_DEPS_SH)
    write_text(tools_dir / "profile_tts_inference.sh", RUNTIME_PROFILE_TTS_INFERENCE_SH)


def populate_archive_artifacts(stage_dir: Path, cache_dir: Path) -> None:
    for artifact in ARTIFACTS:
        output_path = artifact_output_path(stage_dir, artifact)
        if output_path.exists():
            log(f"Reusing existing artifact contents: {output_path}")
            continue
        archive_path = download_artifact(cache_dir, artifact)
        extract_target = stage_dir / artifact.target_subdir
        log(f"Extracting {artifact.name}")
        extract_tarball(
            archive_path,
            extract_target,
            strip_top_level=artifact.strip_top_level,
            extracted_dir_name=artifact.extracted_dir_name,
        )


def populate_direct_files(stage_dir: Path, cache_dir: Path) -> None:
    root_dir = source_root(stage_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    for direct_file in DIRECT_FILES:
        destination = root_dir / direct_file.relative_path
        if destination.exists() and destination.stat().st_size >= direct_file.min_size_bytes:
            log(f"Reusing existing direct file: {destination}")
            continue
        cached_file = download_direct_file(cache_dir, direct_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_file, destination)


def populate_python_wheels(stage_dir: Path, cache_dir: Path) -> None:
    download_python_wheels(cache_dir)
    source_wheelhouse_dir = cache_wheelhouse_dir(cache_dir)
    destination_dir = stage_wheelhouse_dir(stage_dir)
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    for wheel_file in source_wheelhouse_dir.glob("*.whl"):
        shutil.copy2(wheel_file, destination_dir / wheel_file.name)


def validate_source_bundle(stage_dir: Path) -> None:
    root_dir = source_root(stage_dir)
    wheelhouse_dir = stage_wheelhouse_dir(stage_dir)
    required_paths = (
        root_dir / "english_utils",
        root_dir / "text",
        root_dir / "melotts_rknn.py",
        root_dir / "utils.py",
        root_dir / "requirements.txt",
        root_dir / "encoder.onnx",
        root_dir / "decoder.rknn",
        root_dir / "g.bin",
        root_dir / "lexicon.txt",
        root_dir / "tokens.txt",
    )
    for required_path in required_paths:
        if not required_path.exists():
            fail(f"Source bundle is missing required content: {required_path}")
    if not required_wheels_present(wheelhouse_dir):
        fail(f"Source bundle is missing required Python wheels: {wheelhouse_dir}")


def prepare_source_bundle(stage_dir: Path, *, force: bool = False) -> Path:
    cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    if force and stage_dir.exists():
        shutil.rmtree(stage_dir)

    stage_dir.mkdir(parents=True, exist_ok=True)
    populate_archive_artifacts(stage_dir, cache_dir)
    populate_direct_files(stage_dir, cache_dir)
    populate_python_wheels(stage_dir, cache_dir)
    validate_source_bundle(stage_dir)
    return stage_dir