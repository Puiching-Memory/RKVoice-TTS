from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

from .asr_rknn_export import materialize_streaming_zipformer_rknn
from .config import (
    ASR_ARTIFACTS,
    ASR_DEFAULT_CACHE_DIR,
    ASR_RUNTIME_SUBDIR_NAME,
    ASR_RUNTIME_BOARD_PROFILE_CAPABILITIES_SH,
    ASR_RUNTIME_CHECK_RKNN_ENV_SH,
    ASR_RUNTIME_PROFILE_INFERENCE_SH,
    ASR_RUNTIME_README,
    ASR_RUNTIME_RUN_SH,
    ASR_RUNTIME_SMOKETEST_SH,
    AUDIOS_DIR,
    LEGACY_STREAMING_RKNN_ASSET_NAME,
    LEGACY_STREAMING_RKNN_ASSET_URL,
    PREBUILT_RUNTIME_DIR_NAME,
    RKNN_TOOLKIT2_TARGET_PLATFORM,
    STREAMING_ONNX_ASR_SOURCE_DIR_NAME,
    STREAMING_RKNN_ASR_DIR_NAME,
    UNIFIED_RUNTIME_README,
    WORKSPACE_ROOT,
    Artifact,
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
# Path helpers
# ---------------------------------------------------------------------------

PREBUILT_RUNTIME_RELATIVE_PATH = Path("prebuilt") / PREBUILT_RUNTIME_DIR_NAME
STREAMING_ONNX_ASR_SOURCE_RELATIVE_PATH = Path("source-models") / "asr" / "streaming-onnx" / STREAMING_ONNX_ASR_SOURCE_DIR_NAME
STREAMING_RKNN_ASR_RELATIVE_PATH = Path("models") / "asr" / "streaming-rknn" / STREAMING_RKNN_ASR_DIR_NAME


def runtime_component_dir(runtime_dir: Path) -> Path:
    return runtime_dir / ASR_RUNTIME_SUBDIR_NAME


def materialize_runtime_root(runtime_dir: Path) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_text(runtime_dir / "README_SDK.md", UNIFIED_RUNTIME_README)
    return runtime_component_dir(runtime_dir)


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


def artifact_output_path(stage_dir: Path, artifact: Artifact) -> Path:
    base_path = stage_dir / artifact.target_subdir
    if artifact.strip_top_level:
        return base_path / (artifact.extracted_dir_name or "")
    return base_path


def populate_artifacts(stage_dir: Path, cache_dir: Path) -> None:
    for artifact in ASR_ARTIFACTS:
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


def extract_member_from_tarball(archive_path: Path, member_suffix: str, destination: Path) -> bool:
    normalized_suffix = member_suffix.replace("\\", "/")
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            normalized_name = member.name.replace("\\", "/")
            if not normalized_name.endswith(normalized_suffix):
                continue
            file_object = archive.extractfile(member)
            if file_object is None:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as output:
                shutil.copyfileobj(file_object, output)
            return True
    return False


def hydrate_streaming_source_tokens(stage_dir: Path, cache_dir: Path) -> Path:
    source_dir = stage_dir / STREAMING_ONNX_ASR_SOURCE_RELATIVE_PATH
    tokens_path = source_dir / "tokens.txt"
    if tokens_path.exists():
        return tokens_path

    legacy_archive = cache_dir / LEGACY_STREAMING_RKNN_ASSET_NAME
    if not legacy_archive.exists():
        download_http_file(LEGACY_STREAMING_RKNN_ASSET_URL, legacy_archive)

    if not extract_member_from_tarball(legacy_archive, "/tokens.txt", tokens_path):
        fail(f"Legacy streaming RKNN archive does not contain tokens.txt: {legacy_archive}")

    log(f"Bootstrapped streaming ASR tokens from legacy RKNN archive: {tokens_path}")
    return tokens_path


def prepare_source_bundle(stage_dir: Path, *, force: bool = False) -> Path:
    cache_dir = ASR_DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    if force and stage_dir.exists():
        shutil.rmtree(stage_dir)

    stage_dir.mkdir(parents=True, exist_ok=True)
    populate_artifacts(stage_dir, cache_dir)
    hydrate_streaming_source_tokens(stage_dir, cache_dir)
    return stage_dir


def materialize_streaming_rknn_models(stage_dir: Path, *, force: bool) -> Path:
    source_dir = stage_dir / STREAMING_ONNX_ASR_SOURCE_RELATIVE_PATH
    output_dir = stage_dir / STREAMING_RKNN_ASR_RELATIVE_PATH
    return materialize_streaming_zipformer_rknn(
        source_dir,
        output_dir,
        workspace_root=WORKSPACE_ROOT,
        target=RKNN_TOOLKIT2_TARGET_PLATFORM,
        force=force,
    )


# ---------------------------------------------------------------------------
# Runtime bundle
# ---------------------------------------------------------------------------

def materialize_runtime_support_files(runtime_dir: Path) -> None:
    component_dir = materialize_runtime_root(runtime_dir)
    tools_dir = component_dir / "tools"
    output_dir = component_dir / "output"
    tools_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(component_dir / "README_SDK.md", ASR_RUNTIME_README)
    write_text(component_dir / "run_asr.sh", ASR_RUNTIME_RUN_SH)
    write_text(component_dir / "smoketest.sh", ASR_RUNTIME_SMOKETEST_SH)
    write_text(tools_dir / "check_rknn_env.sh", ASR_RUNTIME_CHECK_RKNN_ENV_SH)
    write_text(tools_dir / "board_profile_capabilities.sh", ASR_RUNTIME_BOARD_PROFILE_CAPABILITIES_SH)
    write_text(tools_dir / "profile_asr_inference.sh", ASR_RUNTIME_PROFILE_INFERENCE_SH)


def runtime_bundle_required_paths(runtime_dir: Path) -> tuple[Path, ...]:
    component_dir = runtime_component_dir(runtime_dir)
    return (
        component_dir / "bin" / "sherpa-onnx",
        component_dir / "lib" / "libsherpa-onnx-c-api.so",
        component_dir / "models" / "asr" / "streaming-rknn" / STREAMING_RKNN_ASR_DIR_NAME / "encoder.rknn",
        component_dir / "audios",
        component_dir / "run_asr.sh",
        component_dir / "smoketest.sh",
    )


def build_runtime_bundle(stage_dir: Path, runtime_dir: Path, *, force: bool) -> Path:
    component_dir = runtime_component_dir(runtime_dir)
    if force and component_dir.exists():
        shutil.rmtree(component_dir)

    required_runtime_paths = runtime_bundle_required_paths(runtime_dir)
    if all(path.exists() for path in required_runtime_paths):
        materialize_runtime_support_files(runtime_dir)
        log(f"Reusing existing ASR runtime component: {component_dir}")
        return runtime_dir

    prebuilt_dir = stage_dir / PREBUILT_RUNTIME_RELATIVE_PATH
    streaming_rknn_asr_dir = materialize_streaming_rknn_models(stage_dir, force=force)

    for required_path in (
        prebuilt_dir / "bin",
        prebuilt_dir / "lib",
        streaming_rknn_asr_dir,
    ):
        if not required_path.exists():
            fail(f"Source bundle is missing required content: {required_path}")

    if component_dir.exists():
        shutil.rmtree(component_dir)
    component_dir.mkdir(parents=True, exist_ok=True)

    merge_tree(prebuilt_dir / "bin", component_dir / "bin")
    merge_tree(prebuilt_dir / "lib", component_dir / "lib")
    merge_tree(prebuilt_dir / "include", component_dir / "include")
    merge_tree(stage_dir / "models", component_dir / "models")
    if AUDIOS_DIR.exists():
        merge_tree(AUDIOS_DIR, component_dir / "audios")
    (component_dir / "output").mkdir(parents=True, exist_ok=True)

    materialize_runtime_support_files(runtime_dir)

    for required_path in required_runtime_paths:
        if not required_path.exists():
            fail(f"Runtime bundle is missing required artifact after assembly: {required_path}")
    log(f"ASR runtime component prepared at {component_dir}")
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
    skip_smoketest: bool,
    enable_rknn_smoketest: bool,
) -> None:
    for required_path in runtime_bundle_required_paths(runtime_dir):
        if not required_path.exists():
            fail(f"Runtime bundle is missing required artifact: {required_path}")

    materialize_runtime_support_files(runtime_dir)

    component_dir = runtime_component_dir(runtime_dir)
    component_remote_dir = f"{remote_dir.rstrip('/')}/{ASR_RUNTIME_SUBDIR_NAME}"
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
            (
                f"find {sh_quote(component_remote_dir)} -type f -name '*.sh' -exec chmod +x {{}} + && "
                f"find {sh_quote(component_remote_dir)}/bin -maxdepth 1 -type f -exec chmod +x {{}} +"
            ),
            timeout=remote_timeout,
        )
        if not skip_smoketest:
            log("Running remote smoke test")
            remote_output_log = "./output/smoke_test_summary.log"
            rknn_flag = "1" if enable_rknn_smoketest else "0"
            smoke_test_command = (
                "set -o pipefail; "
                "mkdir -p ./output; "
                f"RKVOICE_ENABLE_RKNN_SMOKETEST={rknn_flag} ./smoketest.sh "
                f"2>&1 | tee {sh_quote(remote_output_log)}"
            )
            local_log_path = local_output_dir / "smoke_test_summary.log"
            remote_output_dir = f"{component_remote_dir.rstrip('/')}/output"
            smoke_failure: SystemExit | None = None
            try:
                smoke_test_output = run_remote_command(
                    client,
                    f"cd {sh_quote(component_remote_dir)} && bash -lc {sh_quote(smoke_test_command)}",
                    timeout=remote_timeout,
                )
                write_text(local_log_path, smoke_test_output)
            except SystemExit as exc:
                smoke_failure = exc

            download_file(client, f"{remote_output_dir}/smoke_test_summary.log", local_log_path)
            for optional_name in (
                "board_profile_capabilities.txt",
                "rknpu_load.log",
                "rknn_profile.log",
                "rknn_runtime.log",
                "rknn_eval_perf.txt",
                "rknn_query_perf_detail.txt",
                "rknn_perf_detail.txt",
                "rknn_perf_run.json",
                "rknn_query_perf_run.txt",
                "rknn_perf_run.txt",
                "rknn_memory_profile.txt",
                "rknn_eval_memory.txt",
                "rknn_query_mem_size.json",
            ):
                try:
                    download_file(client, f"{remote_output_dir}/{optional_name}", local_output_dir / optional_name)
                except OSError:
                    pass
            log(f"Downloaded smoke test log to {local_log_path}")
            if smoke_failure is not None:
                raise smoke_failure
    finally:
        client.close()
