from __future__ import annotations

import shutil
from pathlib import Path

from .config import CPU_ASR_DIR_NAME, PREBUILT_RUNTIME_DIR_NAME, RKNN_ASR_DIR_NAME, STREAMING_ASR_DIR_NAME, TTS_DIR_NAME
from .shared import fail, log, merge_tree
from .source_bundle import materialize_runtime_support_files


PREBUILT_RUNTIME_RELATIVE_PATH = Path("prebuilt") / PREBUILT_RUNTIME_DIR_NAME
CPU_ASR_RELATIVE_PATH = Path("models") / "asr" / "cpu" / CPU_ASR_DIR_NAME
RKNN_ASR_RELATIVE_PATH = Path("models") / "asr" / "rknn" / RKNN_ASR_DIR_NAME
STREAMING_ASR_RELATIVE_PATH = Path("models") / "asr" / "streaming" / STREAMING_ASR_DIR_NAME
TTS_RELATIVE_PATH = Path("models") / "tts" / TTS_DIR_NAME


def runtime_bundle_required_paths(runtime_dir: Path) -> tuple[Path, ...]:
    return (
        runtime_dir / "bin" / "sherpa-onnx",
        runtime_dir / "bin" / "sherpa-onnx-offline",
        runtime_dir / "bin" / "sherpa-onnx-offline-tts",
        runtime_dir / "lib" / "libsherpa-onnx-c-api.so",
        runtime_dir / "models" / "asr" / "cpu" / CPU_ASR_DIR_NAME / "model.int8.onnx",
        runtime_dir / "models" / "asr" / "rknn" / RKNN_ASR_DIR_NAME / "model.rknn",
        runtime_dir / "models" / "asr" / "streaming" / STREAMING_ASR_DIR_NAME / "encoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx",
        runtime_dir / "models" / "tts" / TTS_DIR_NAME / "model.onnx",
        runtime_dir / "run_asr.sh",
        runtime_dir / "run_tts.sh",
        runtime_dir / "smoketest.sh",
    )


def build_runtime_bundle(stage_dir: Path, runtime_dir: Path, *, force: bool) -> Path:
    if force and runtime_dir.exists():
        shutil.rmtree(runtime_dir)

    required_runtime_paths = runtime_bundle_required_paths(runtime_dir)
    if all(path.exists() for path in required_runtime_paths):
        materialize_runtime_support_files(runtime_dir)
        log(f"Reusing existing runtime bundle: {runtime_dir}")
        return runtime_dir

    prebuilt_dir = stage_dir / PREBUILT_RUNTIME_RELATIVE_PATH
    cpu_asr_dir = stage_dir / CPU_ASR_RELATIVE_PATH
    rknn_asr_dir = stage_dir / RKNN_ASR_RELATIVE_PATH
    streaming_asr_dir = stage_dir / STREAMING_ASR_RELATIVE_PATH
    tts_dir = stage_dir / TTS_RELATIVE_PATH

    for required_path in (
        prebuilt_dir / "bin",
        prebuilt_dir / "lib",
        cpu_asr_dir,
        rknn_asr_dir,
        streaming_asr_dir,
        tts_dir,
    ):
        if not required_path.exists():
            fail(f"Source bundle is missing required content: {required_path}")

    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    merge_tree(prebuilt_dir / "bin", runtime_dir / "bin")
    merge_tree(prebuilt_dir / "lib", runtime_dir / "lib")
    merge_tree(prebuilt_dir / "include", runtime_dir / "include")
    merge_tree(stage_dir / "models", runtime_dir / "models")
    (runtime_dir / "output").mkdir(parents=True, exist_ok=True)

    materialize_runtime_support_files(runtime_dir)

    for required_path in required_runtime_paths:
        if not required_path.exists():
            fail(f"Runtime bundle is missing required artifact after assembly: {required_path}")
    log(f"Runtime bundle prepared at {runtime_dir}")
    return runtime_dir