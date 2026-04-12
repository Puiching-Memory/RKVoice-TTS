from __future__ import annotations

import shutil
from pathlib import Path

from .config import (
    ARTIFACTS,
    DEFAULT_CACHE_DIR,
    OFFLINE_BUILD_SH,
    OFFLINE_ENV_SH,
    OFFLINE_RUN_SH,
    OFFLINE_SMOKETEST_SH,
    OFFLINE_THIRD_PARTY_CMAKELISTS,
    PADDLESPEECH_ARCHIVE_ROOT,
    PADDLESPEECH_ARCHIVE_URL,
    PADDLESPEECH_FRONTEND_SUBTREE,
    PADDLESPEECH_TTS_SUBTREE,
    ROOT_BUILD_DEPENDS_SH,
    RKVOICE_TTS_CMAKELISTS,
    RKVOICE_TTS_CORE_CC,
    RKVOICE_TTS_DEMO_MAIN_CC,
    RKVOICE_TTS_PUBLIC_HEADER,
    RUNTIME_C_API_DEMO_SOURCE,
    RUNTIME_PROFILE_CAPABILITIES_SH,
    RUNTIME_PROFILE_TTS_INFERENCE_SH,
    RUNTIME_RUN_SH,
    RUNTIME_SDK_README,
    RUNTIME_SMOKETEST_SH,
    Artifact,
)
from .shared import copy_tree, download_http_file, extract_tar_members, extract_tarball, fail, log, md5sum, write_text


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

    download_http_file(artifact.url, destination)
    if artifact.md5 and md5sum(destination) != artifact.md5:
        destination.unlink(missing_ok=True)
        fail(f"MD5 mismatch for {artifact.name}")
    return destination


def materialize_runtime_support_files(runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    tools_dir = runtime_dir / "tools"
    include_dir = runtime_dir / "include"
    examples_dir = runtime_dir / "examples"
    tools_dir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)
    write_text(runtime_dir / "run_tts.sh", RUNTIME_RUN_SH)
    write_text(runtime_dir / "smoketest.sh", RUNTIME_SMOKETEST_SH)
    write_text(runtime_dir / "README_SDK.md", RUNTIME_SDK_README)
    write_text(include_dir / "rkvoice_tts_api.h", RKVOICE_TTS_PUBLIC_HEADER)
    write_text(examples_dir / "c_api_demo.c", RUNTIME_C_API_DEMO_SOURCE)
    write_text(tools_dir / "board_profile_capabilities.sh", RUNTIME_PROFILE_CAPABILITIES_SH)
    write_text(tools_dir / "profile_tts_inference.sh", RUNTIME_PROFILE_TTS_INFERENCE_SH)


def materialize_repo_sources(stage_dir: Path, cache_dir: Path) -> None:
    repo_archive = cache_dir / "PaddleSpeech-develop.tar.gz"
    if not repo_archive.exists():
        download_http_file(PADDLESPEECH_ARCHIVE_URL, repo_archive)

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

    write_text(stage_dir / "src" / "CMakeLists.txt", RKVOICE_TTS_CMAKELISTS)
    write_text(stage_dir / "src" / "rkvoice_tts_api.h", RKVOICE_TTS_PUBLIC_HEADER)
    write_text(stage_dir / "src" / "rkvoice_tts_core.cc", RKVOICE_TTS_CORE_CC)
    write_text(stage_dir / "src" / "rkvoice_tts_demo_main.cc", RKVOICE_TTS_DEMO_MAIN_CC)

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
