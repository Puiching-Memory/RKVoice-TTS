from __future__ import annotations

import shutil
from pathlib import Path

from .config import (
    ARTIFACTS,
    DEFAULT_CACHE_DIR,
    RUNTIME_BOARD_PROFILE_CAPABILITIES_SH,
    RUNTIME_CHECK_RKNN_ENV_SH,
    RUNTIME_PROFILE_ASR_INFERENCE_SH,
    RUNTIME_PROFILE_TTS_INFERENCE_SH,
    RUNTIME_README,
    RUNTIME_RUN_ASR_SH,
    RUNTIME_RUN_TTS_SH,
    RUNTIME_SMOKETEST_SH,
    Artifact,
)
from .shared import download_http_file, extract_tarball, log, write_text


def download_artifact(cache_dir: Path, artifact: Artifact) -> Path:
    destination = cache_dir / artifact.name
    if destination.exists():
        log(f"Using cached artifact {artifact.name}")
        return destination
    download_http_file(artifact.url, destination)
    return destination


def artifact_output_path(stage_dir: Path, artifact: Artifact) -> Path:
    base_path = stage_dir / artifact.target_subdir
    if artifact.strip_top_level:
        return base_path / (artifact.extracted_dir_name or "")
    return base_path


def materialize_runtime_support_files(runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    tools_dir = runtime_dir / "tools"
    output_dir = runtime_dir / "output"
    tools_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(runtime_dir / "README_SDK.md", RUNTIME_README)
    write_text(runtime_dir / "run_asr.sh", RUNTIME_RUN_ASR_SH)
    write_text(runtime_dir / "run_tts.sh", RUNTIME_RUN_TTS_SH)
    write_text(runtime_dir / "smoketest.sh", RUNTIME_SMOKETEST_SH)
    write_text(tools_dir / "check_rknn_env.sh", RUNTIME_CHECK_RKNN_ENV_SH)
    write_text(tools_dir / "board_profile_capabilities.sh", RUNTIME_BOARD_PROFILE_CAPABILITIES_SH)
    write_text(tools_dir / "profile_asr_inference.sh", RUNTIME_PROFILE_ASR_INFERENCE_SH)
    write_text(tools_dir / "profile_tts_inference.sh", RUNTIME_PROFILE_TTS_INFERENCE_SH)


def populate_artifacts(stage_dir: Path, cache_dir: Path) -> None:
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


def prepare_source_bundle(stage_dir: Path, *, force: bool = False) -> Path:
    cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    if force and stage_dir.exists():
        shutil.rmtree(stage_dir)

    stage_dir.mkdir(parents=True, exist_ok=True)
    populate_artifacts(stage_dir, cache_dir)
    return stage_dir