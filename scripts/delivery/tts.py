from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

from .config import (
    BOARD_PYTHON_ABI,
    BOARD_PYTHON_VERSION,
    BOARD_WHEEL_PLATFORM,
    COMMON_PYTHON_DEPENDENCIES,
    REQUIRED_WHEEL_PATTERNS,
    RKNN_LITE_DEPENDENCY,
    SOURCE_ROOT_RELATIVE_PATH,
    TTS_ARTIFACTS,
    TTS_DEFAULT_CACHE_DIR,
    TTS_DIRECT_FILES,
    TTS_RUNTIME_SUBDIR_NAME,
    TTS_RUNTIME_BOARD_PROFILE_CAPABILITIES_SH,
    TTS_RUNTIME_CHECK_PYTHON_ENV_SH,
    TTS_RUNTIME_INSTALL_PYTHON_DEPS_SH,
    TTS_RUNTIME_PROFILE_INFERENCE_SH,
    TTS_RUNTIME_README,
    TTS_RUNTIME_RUN_SH,
    TTS_RUNTIME_SMOKETEST_SH,
    UNIFIED_RUNTIME_README,
    WHEELHOUSE_RELATIVE_PATH,
    Artifact,
    DirectFile,
)
from .remote import (
    create_bundle_tarball,
    download_file,
    open_ssh_client,
    run_remote_command,
    sh_quote,
    upload_file,
)
from .shared import download_http_file, extract_tarball, fail, log, merge_tree, write_text


# ---------------------------------------------------------------------------
# Source bundle
# ---------------------------------------------------------------------------

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


def populate_archive_artifacts(stage_dir: Path, cache_dir: Path) -> None:
    for artifact in TTS_ARTIFACTS:
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
    for direct_file in TTS_DIRECT_FILES:
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
    cache_dir = TTS_DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    if force and stage_dir.exists():
        shutil.rmtree(stage_dir)

    stage_dir.mkdir(parents=True, exist_ok=True)
    populate_archive_artifacts(stage_dir, cache_dir)
    populate_direct_files(stage_dir, cache_dir)
    populate_python_wheels(stage_dir, cache_dir)
    validate_source_bundle(stage_dir)
    return stage_dir


# ---------------------------------------------------------------------------
# Runtime bundle
# ---------------------------------------------------------------------------

ROOT_FILES = (
    "melotts_rknn.py",
    "utils.py",
    "requirements.txt",
    "encoder.onnx",
    "decoder.rknn",
    "g.bin",
    "lexicon.txt",
    "tokens.txt",
)


def runtime_component_dir(runtime_dir: Path) -> Path:
    return runtime_dir / TTS_RUNTIME_SUBDIR_NAME


def materialize_runtime_root(runtime_dir: Path) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_text(runtime_dir / "README_SDK.md", UNIFIED_RUNTIME_README)
    return runtime_component_dir(runtime_dir)


def materialize_runtime_support_files(runtime_dir: Path) -> None:
    component_dir = materialize_runtime_root(runtime_dir)
    tools_dir = component_dir / "tools"
    output_dir = component_dir / "output"
    bin_dir = component_dir / "bin"
    tools_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    write_text(component_dir / "README_SDK.md", TTS_RUNTIME_README)
    write_text(component_dir / "run_tts.sh", TTS_RUNTIME_RUN_SH)
    write_text(component_dir / "smoketest.sh", TTS_RUNTIME_SMOKETEST_SH)
    write_text(tools_dir / "board_profile_capabilities.sh", TTS_RUNTIME_BOARD_PROFILE_CAPABILITIES_SH)
    write_text(tools_dir / "check_python_env.sh", TTS_RUNTIME_CHECK_PYTHON_ENV_SH)
    write_text(tools_dir / "install_python_deps.sh", TTS_RUNTIME_INSTALL_PYTHON_DEPS_SH)
    write_text(tools_dir / "profile_tts_inference.sh", TTS_RUNTIME_PROFILE_INFERENCE_SH)


def runtime_bundle_required_paths(runtime_dir: Path) -> tuple[Path, ...]:
    component_dir = runtime_component_dir(runtime_dir)
    return (
        component_dir / "melotts_rknn.py",
        component_dir / "utils.py",
        component_dir / "requirements.txt",
        component_dir / "encoder.onnx",
        component_dir / "decoder.rknn",
        component_dir / "g.bin",
        component_dir / "lexicon.txt",
        component_dir / "tokens.txt",
        component_dir / "english_utils",
        component_dir / "text",
        component_dir / "wheels",
        component_dir / "run_tts.sh",
        component_dir / "smoketest.sh",
        component_dir / "tools" / "install_python_deps.sh",
    )


def build_runtime_bundle(stage_dir: Path, runtime_dir: Path, *, force: bool) -> Path:
    component_dir = runtime_component_dir(runtime_dir)
    if force and component_dir.exists():
        shutil.rmtree(component_dir)

    required_runtime_paths = runtime_bundle_required_paths(runtime_dir)
    if all(path.exists() for path in required_runtime_paths):
        materialize_runtime_support_files(runtime_dir)
        log(f"Reusing existing TTS runtime component: {component_dir}")
        return runtime_dir

    source_dir = stage_dir / SOURCE_ROOT_RELATIVE_PATH
    for required_path in (
        source_dir / "english_utils",
        source_dir / "text",
        stage_dir / WHEELHOUSE_RELATIVE_PATH,
        *(source_dir / file_name for file_name in ROOT_FILES),
    ):
        if not required_path.exists():
            fail(f"Source bundle is missing required content: {required_path}")

    if component_dir.exists():
        shutil.rmtree(component_dir)
    component_dir.mkdir(parents=True, exist_ok=True)

    for file_name in ROOT_FILES:
        shutil.copy2(source_dir / file_name, component_dir / file_name)
    merge_tree(source_dir / "english_utils", component_dir / "english_utils")
    merge_tree(source_dir / "text", component_dir / "text")
    merge_tree(stage_dir / WHEELHOUSE_RELATIVE_PATH, component_dir / "wheels")
    (component_dir / "output").mkdir(parents=True, exist_ok=True)
    (component_dir / "bin").mkdir(parents=True, exist_ok=True)

    materialize_runtime_support_files(runtime_dir)

    for required_path in required_runtime_paths:
        if not required_path.exists():
            fail(f"Runtime bundle is missing required artifact after assembly: {required_path}")
    log(f"TTS runtime component prepared at {component_dir}")
    return runtime_dir


# ---------------------------------------------------------------------------
# Remote deploy
# ---------------------------------------------------------------------------

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
    text: str,
    install_python_deps: bool,
    skip_smoketest: bool,
) -> None:
    for required_path in runtime_bundle_required_paths(runtime_dir):
        if not required_path.exists():
            fail(f"Runtime bundle is missing required artifact: {required_path}")

    materialize_runtime_support_files(runtime_dir)

    component_dir = runtime_component_dir(runtime_dir)
    component_remote_dir = f"{remote_dir.rstrip('/')}/{TTS_RUNTIME_SUBDIR_NAME}"
    client = open_ssh_client(host, username, password, source_ip=source_ip, timeout=ssh_timeout)
    local_output_dir = component_dir / "output"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = create_bundle_tarball(component_dir, component_remote_dir)
    remote_parent = str(Path(component_remote_dir).parent).replace("\\", "/")
    remote_tarball = f"{remote_parent}/{Path(tarball_path.name).name}"
    try:
        log(f"Uploading runtime tarball to {remote_tarball}")
        run_remote_command(client, f"mkdir -p {sh_quote(remote_parent)}", timeout=remote_timeout)
        upload_file(client, tarball_path, remote_tarball)
        run_remote_command(
            client,
            f"rm -rf {sh_quote(component_remote_dir)} && tar -xzf {sh_quote(remote_tarball)} -C {sh_quote(remote_parent)}",
            timeout=remote_timeout,
        )
        run_remote_command(
            client,
            f"find {sh_quote(component_remote_dir)} -type f \\( -name '*.sh' -o -name '*.py' \\) -exec chmod +x {{}} +",
            timeout=remote_timeout,
        )
        if install_python_deps:
            log("Installing board-side Python dependencies")
            install_output = run_remote_command(
                client,
                f"cd {sh_quote(component_remote_dir)} && ./tools/install_python_deps.sh",
                timeout=remote_timeout,
            )
            write_text(local_output_dir / "python_deps_install.log", install_output)
        if not skip_smoketest:
            log("Running remote smoke test")
            remote_output_wav = "./output/smoke_test_tts.wav"
            remote_output_log = "./output/smoke_test_summary.log"
            smoke_test_command = (
                "set -o pipefail; "
                "mkdir -p ./output; "
                f"./smoketest.sh {sh_quote(text)} {sh_quote(remote_output_wav)} 2>&1 | tee {sh_quote(remote_output_log)}"
            )
            smoke_test_output = run_remote_command(
                client,
                f"cd {sh_quote(component_remote_dir)} && bash -lc {sh_quote(smoke_test_command)}",
                timeout=remote_timeout,
            )
            local_log_path = local_output_dir / "smoke_test_summary.log"
            write_text(local_log_path, smoke_test_output)
            remote_output_dir = f"{component_remote_dir.rstrip('/')}/output"
            local_wav_path = local_output_dir / "smoke_test_tts.wav"
            download_file(client, f"{remote_output_dir}/smoke_test_tts.wav", local_wav_path)
            download_file(client, f"{remote_output_dir}/smoke_test_summary.log", local_log_path)
            for optional_name in (
                "python_deps_install.log",
                "python_deps_path.txt",
                "board_profile_capabilities.txt",
                "warm_run_tts.wav",
                "profile_tts.wav",
                "profile-samples.csv",
                "rknn_runtime.log",
            ):
                try:
                    download_file(client, f"{remote_output_dir}/{optional_name}", local_output_dir / optional_name)
                except OSError:
                    pass
            log(f"Downloaded smoke test audio to {local_wav_path}")
            log(f"Downloaded smoke test log to {local_log_path}")
    finally:
        client.close()
