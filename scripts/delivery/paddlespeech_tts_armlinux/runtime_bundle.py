from __future__ import annotations

import shutil
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

from .config import DOCKER_BUILD_PACKAGES, PADDLE_LITE_LIB_DIR, RUNTIME_C_API_DEMO_SOURCE, RUNTIME_RUN_SH, RUNTIME_SDK_README, RUNTIME_SMOKETEST_SH
from .shared import fail, log, write_text
from .source_bundle import materialize_runtime_support_files


def runtime_bundle_required_paths(runtime_dir: Path) -> tuple[Path, ...]:
    return (
        runtime_dir / "bin" / "paddlespeech_tts_demo",
        runtime_dir / "bin" / "rkvoice_tts_demo",
        runtime_dir / "lib" / "librkvoice_tts.so",
        runtime_dir / "lib" / "librkvoice_tts.a",
        runtime_dir / "include" / "rkvoice_tts_api.h",
    )


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
                print(line, end="", flush=True)
            elif process.poll() is not None:
                break
            if timeout and time.time() - start_time > timeout:
                process.kill()
                fail(f"Local command timed out: {' '.join(command)}")
        remaining_output = process.stdout.read()
        if remaining_output:
            print(remaining_output, end="", flush=True)
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
    required_runtime_paths = runtime_bundle_required_paths(runtime_dir)
    if force and runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    if all(path.exists() for path in required_runtime_paths):
        materialize_runtime_support_files(runtime_dir)
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
            "mkdir -p /workspace/out/bin /workspace/out/lib /workspace/out/models /workspace/out/output /workspace/out/dict /workspace/out/include /workspace/out/examples",
            "cp ./build/paddlespeech_tts_demo /workspace/out/bin/",
            "cp ./build/rkvoice_tts_demo /workspace/out/bin/",
            "cp ./build/librkvoice_tts.so /workspace/out/lib/",
            "cp ./build/librkvoice_tts.a /workspace/out/lib/",
            "cp ./front.conf /workspace/out/front.conf",
            "cp -r ./models/. /workspace/out/models/",
            "cp -r ./src/TTSCppFrontend/front_demo/dict/. /workspace/out/dict/",
            "cp ./src/rkvoice_tts_api.h /workspace/out/include/",
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
            "cat > /workspace/out/README_SDK.md <<'EOF'",
            RUNTIME_SDK_README.rstrip(),
            "EOF",
            "cat > /workspace/out/examples/c_api_demo.c <<'EOF'",
            RUNTIME_C_API_DEMO_SOURCE.rstrip(),
            "EOF",
            "chmod +x /workspace/out/run_tts.sh /workspace/out/smoketest.sh /workspace/out/bin/paddlespeech_tts_demo /workspace/out/bin/rkvoice_tts_demo",
            "file /workspace/out/bin/paddlespeech_tts_demo",
            "file /workspace/out/bin/rkvoice_tts_demo",
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

    for required_path in required_runtime_paths:
        if not required_path.exists():
            fail(f"Docker build finished but required runtime artifact was not produced: {required_path}")
    materialize_runtime_support_files(runtime_dir)
    log(f"Runtime bundle prepared at {runtime_dir}")
    return runtime_dir
