from __future__ import annotations

import shutil
from pathlib import Path

from .config import SOURCE_ROOT_RELATIVE_PATH, WHEELHOUSE_RELATIVE_PATH
from .shared import fail, merge_tree, log
from .source_bundle import materialize_runtime_support_files


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


def runtime_bundle_required_paths(runtime_dir: Path) -> tuple[Path, ...]:
    return (
        runtime_dir / "melotts_rknn.py",
        runtime_dir / "utils.py",
        runtime_dir / "requirements.txt",
        runtime_dir / "encoder.onnx",
        runtime_dir / "decoder.rknn",
        runtime_dir / "g.bin",
        runtime_dir / "lexicon.txt",
        runtime_dir / "tokens.txt",
        runtime_dir / "english_utils",
        runtime_dir / "text",
        runtime_dir / "wheels",
        runtime_dir / "run_tts.sh",
        runtime_dir / "smoketest.sh",
        runtime_dir / "tools" / "install_python_deps.sh",
    )


def build_runtime_bundle(stage_dir: Path, runtime_dir: Path, *, force: bool) -> Path:
    if force and runtime_dir.exists():
        shutil.rmtree(runtime_dir)

    required_runtime_paths = runtime_bundle_required_paths(runtime_dir)
    if all(path.exists() for path in required_runtime_paths):
        materialize_runtime_support_files(runtime_dir)
        log(f"Reusing existing runtime bundle: {runtime_dir}")
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

    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    for file_name in ROOT_FILES:
        shutil.copy2(source_dir / file_name, runtime_dir / file_name)
    merge_tree(source_dir / "english_utils", runtime_dir / "english_utils")
    merge_tree(source_dir / "text", runtime_dir / "text")
    merge_tree(stage_dir / WHEELHOUSE_RELATIVE_PATH, runtime_dir / "wheels")
    (runtime_dir / "output").mkdir(parents=True, exist_ok=True)
    (runtime_dir / "bin").mkdir(parents=True, exist_ok=True)

    materialize_runtime_support_files(runtime_dir)

    for required_path in required_runtime_paths:
        if not required_path.exists():
            fail(f"Runtime bundle is missing required artifact after assembly: {required_path}")
    log(f"Runtime bundle prepared at {runtime_dir}")
    return runtime_dir