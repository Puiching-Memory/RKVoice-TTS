from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

try:
    from .config import RKNN_TOOLKIT2_IMAGE_TAG, RKNN_TOOLKIT2_TARGET_PLATFORM
    from .shared import log, write_text
except ImportError:
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))
    from scripts.delivery.config import RKNN_TOOLKIT2_IMAGE_TAG, RKNN_TOOLKIT2_TARGET_PLATFORM
    from scripts.delivery.shared import log, write_text

DEFAULT_CONTAINER_WORKSPACE = Path("/workspace")
DEFAULT_DOCKERFILE = WORKSPACE_ROOT / "docker" / "toolkit2-profile" / "Dockerfile"
DEFAULT_BUILD_CONTEXT = DEFAULT_DOCKERFILE.parent
BUILD_MANIFEST_NAME = "rknn_build_manifest.json"
FIXED_SHAPE_SOURCE_VARIANTS = ("64", "96")
SYMBOLIC_DIM_DEFAULTS = {
    "N": 1,
}
ENCODER_CUSTOM_STRING_KEYS = (
    "model_type",
    "attention_dims",
    "encoder_dims",
    "T",
    "left_context_len",
    "decode_chunk_len",
    "cnn_module_kernels",
    "num_encoder_layers",
)


class ASRRKNNExportError(Exception):
    pass


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def resolve_rknn() -> Any:
    try:
        from rknn.api import RKNN  # type: ignore
    except ImportError as exc:
        raise ASRRKNNExportError(
            "rknn.api 不可用。请在已安装 rknn-toolkit2 的环境中运行，或让脚本自动通过 Docker 导出。"
        ) from exc
    return RKNN


def has_local_rknn_toolchain() -> bool:
    try:
        resolve_rknn()
    except ASRRKNNExportError:
        return False
    return True


def pick_first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def first_glob(path: Path, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(candidate for candidate in path.glob(pattern) if candidate.is_file())
        if matches:
            return matches[0]
    return None


def first_non_int8_glob(path: Path, pattern: str) -> Path | None:
    matches = sorted(
        candidate
        for candidate in path.glob(pattern)
        if candidate.is_file() and not candidate.name.endswith(".int8.onnx")
    )
    if matches:
        return matches[0]
    return None


def find_model_variant_root(source_dir: Path) -> Path:
    for variant_name in FIXED_SHAPE_SOURCE_VARIANTS:
        candidate = source_dir / variant_name
        if candidate.is_dir():
            return candidate
    return source_dir


def find_source_files(source_dir: Path) -> dict[str, Path]:
    if not source_dir.exists() or not source_dir.is_dir():
        raise ASRRKNNExportError(f"ASR ONNX 源模型目录不存在：{source_dir}")

    variant_root = find_model_variant_root(source_dir)

    encoder = pick_first_existing(
        [
            variant_root / "encoder.onnx",
            variant_root / "encoder.int8.onnx",
        ]
    ) or first_non_int8_glob(variant_root, "encoder-*.onnx") or first_glob(variant_root, ("encoder-*.int8.onnx",))
    decoder = pick_first_existing(
        [
            variant_root / "decoder.onnx",
            variant_root / "decoder.int8.onnx",
        ]
    ) or first_non_int8_glob(variant_root, "decoder-*.onnx") or first_glob(variant_root, ("decoder-*.int8.onnx",))
    joiner = pick_first_existing(
        [
            variant_root / "joiner.onnx",
            variant_root / "joiner.int8.onnx",
        ]
    ) or first_non_int8_glob(variant_root, "joiner-*.onnx") or first_glob(variant_root, ("joiner-*.int8.onnx",))
    tokens = variant_root / "tokens.txt"
    test_wavs = source_dir / "test_wavs"

    missing = [
        name
        for name, path in (("encoder", encoder), ("decoder", decoder), ("joiner", joiner))
        if path is None
    ]
    if missing:
        raise ASRRKNNExportError(
            f"ASR ONNX 源模型目录缺少必要文件：{', '.join(missing)}，目录：{source_dir}"
        )
    if not tokens.exists():
        raise ASRRKNNExportError(f"ASR ONNX 源模型目录缺少 tokens.txt：{source_dir}")

    result = {
        "variant_root": variant_root,
        "encoder": encoder,
        "decoder": decoder,
        "joiner": joiner,
        "tokens": tokens,
    }
    if test_wavs.exists() and test_wavs.is_dir():
        result["test_wavs"] = test_wavs
    return result


def resolve_dim_value(dim: Any) -> int:
    dim_value = getattr(dim, "dim_value", 0) or 0
    if dim_value > 0:
        return int(dim_value)

    dim_param = (getattr(dim, "dim_param", "") or "").strip()
    if dim_param:
        return SYMBOLIC_DIM_DEFAULTS.get(dim_param, 1)

    raise ASRRKNNExportError(f"ONNX 输入维度缺少可解析的静态值: {dim}")


def load_onnx_metadata(onnx_path: Path) -> dict[str, str]:
    try:
        onnx = importlib.import_module("onnx")
    except ImportError as exc:
        raise ASRRKNNExportError("缺少 onnx 依赖，无法解析 ONNX metadata。") from exc

    model = onnx.load(str(onnx_path))
    return {prop.key: prop.value for prop in model.metadata_props}


def build_encoder_custom_string(
    encoder_metadata: Mapping[str, str],
    decoder_metadata: Mapping[str, str],
) -> str:
    missing_encoder_keys = [key for key in ENCODER_CUSTOM_STRING_KEYS if not encoder_metadata.get(key)]
    if missing_encoder_keys:
        raise ASRRKNNExportError(
            f"encoder ONNX metadata 缺少字段：{', '.join(missing_encoder_keys)}"
        )

    context_size = (decoder_metadata.get("context_size") or "").strip()
    if not context_size:
        raise ASRRKNNExportError("decoder ONNX metadata 缺少 context_size")

    parts = [f"{key}={encoder_metadata[key]}" for key in ENCODER_CUSTOM_STRING_KEYS]
    parts.append(f"context_size={context_size}")
    return ";".join(parts)


def load_encoder_custom_string(encoder_onnx_path: Path, decoder_onnx_path: Path) -> str:
    encoder_metadata = load_onnx_metadata(encoder_onnx_path)
    decoder_metadata = load_onnx_metadata(decoder_onnx_path)
    return build_encoder_custom_string(encoder_metadata, decoder_metadata)


def load_onnx_input_spec(onnx_path: Path) -> tuple[list[str], list[list[int]]]:
    try:
        onnx = importlib.import_module("onnx")
    except ImportError as exc:
        raise ASRRKNNExportError("缺少 onnx 依赖，无法解析 ONNX 输入签名。") from exc

    model = onnx.load(str(onnx_path))
    input_names: list[str] = []
    input_size_list: list[list[int]] = []
    for tensor in model.graph.input:
        input_names.append(tensor.name)
        input_size_list.append([resolve_dim_value(dim) for dim in tensor.type.tensor_type.shape.dim])
    return input_names, input_size_list


def export_onnx_to_rknn(
    *,
    onnx_path: Path,
    output_path: Path,
    target: str,
    verbose: bool,
    custom_string: str | None = None,
) -> None:
    RKNN = resolve_rknn()
    rknn = RKNN(verbose=verbose)
    input_names, input_size_list = load_onnx_input_spec(onnx_path)
    try:
        config_result = rknn.config(target_platform=target, custom_string=custom_string)
        if config_result not in {None, 0}:
            raise ASRRKNNExportError(f"rknn.config 失败：{config_result} ({onnx_path.name})")

        load_result = rknn.load_onnx(
            model=str(onnx_path),
            inputs=input_names,
            input_size_list=input_size_list,
        )
        if load_result != 0:
            raise ASRRKNNExportError(f"rknn.load_onnx 失败：{load_result} ({onnx_path.name})")

        build_result = rknn.build(do_quantization=False)
        if build_result != 0:
            raise ASRRKNNExportError(f"rknn.build 失败：{build_result} ({onnx_path.name})")

        export_result = rknn.export_rknn(str(output_path))
        if export_result != 0:
            raise ASRRKNNExportError(f"rknn.export_rknn 失败：{export_result} ({output_path.name})")
    finally:
        rknn.release()


def export_streaming_zipformer_models(
    source_dir: Path,
    output_dir: Path,
    *,
    target: str = RKNN_TOOLKIT2_TARGET_PLATFORM,
    verbose: bool = False,
) -> dict[str, Any]:
    source_files = find_source_files(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_custom_string = load_encoder_custom_string(source_files["encoder"], source_files["decoder"])

    export_onnx_to_rknn(
        onnx_path=source_files["encoder"],
        output_path=output_dir / "encoder.rknn",
        target=target,
        verbose=verbose,
        custom_string=encoder_custom_string,
    )
    export_onnx_to_rknn(onnx_path=source_files["decoder"], output_path=output_dir / "decoder.rknn", target=target, verbose=verbose)
    export_onnx_to_rknn(onnx_path=source_files["joiner"], output_path=output_dir / "joiner.rknn", target=target, verbose=verbose)

    shutil.copy2(source_files["tokens"], output_dir / "tokens.txt")
    test_wavs_dir = source_files.get("test_wavs")
    if test_wavs_dir is not None:
        shutil.copytree(test_wavs_dir, output_dir / "test_wavs", dirs_exist_ok=True)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_platform": target,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "toolchain": {
            "mode": "local",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "source_variant_dir": str(source_files["variant_root"]),
        "inputs": {
            "encoder": str(source_files["encoder"]),
            "decoder": str(source_files["decoder"]),
            "joiner": str(source_files["joiner"]),
            "tokens": str(source_files["tokens"]),
        },
        "outputs": {
            "encoder": str(output_dir / "encoder.rknn"),
            "decoder": str(output_dir / "decoder.rknn"),
            "joiner": str(output_dir / "joiner.rknn"),
            "tokens": str(output_dir / "tokens.txt"),
        },
    }
    write_json(output_dir / BUILD_MANIFEST_NAME, manifest)
    return manifest


def format_docker_mount_source(path: Path) -> str:
    resolved = path.resolve()
    if os.name == "nt":
        return resolved.as_posix()
    return str(resolved)


def split_mount_anchor(path: Path, *, treat_as_file: bool) -> tuple[Path, Path]:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        if resolved.is_file() or treat_as_file:
            return resolved.parent, Path(resolved.name)
        return resolved, Path()

    suffix_parts: list[str] = []
    anchor = resolved
    while not anchor.exists():
        if anchor.parent == anchor:
            raise ASRRKNNExportError(f"无法为 Docker 映射路径：{resolved}")
        suffix_parts.append(anchor.name)
        anchor = anchor.parent
    suffix_parts.reverse()
    return anchor, Path(*suffix_parts)


def map_host_path_to_container(
    *,
    host_path: Path,
    workspace_root: Path,
    treat_as_file: bool,
    mount_index: int,
) -> tuple[list[str], str]:
    resolved = host_path.expanduser().resolve()
    try:
        relative = resolved.relative_to(workspace_root)
    except ValueError:
        anchor, suffix = split_mount_anchor(resolved, treat_as_file=treat_as_file)
        container_root = Path("/mnt/external") / str(mount_index)
        container_path = (container_root / suffix).as_posix()
        return ["-v", f"{format_docker_mount_source(anchor)}:{container_root.as_posix()}"], container_path

    return [], (DEFAULT_CONTAINER_WORKSPACE / relative).as_posix()


def build_docker_build_command(*, image_tag: str) -> list[str]:
    return [
        "docker",
        "build",
        "-t",
        image_tag,
        "-f",
        str(DEFAULT_DOCKERFILE),
        str(DEFAULT_BUILD_CONTEXT),
    ]


def build_docker_run_command(
    *,
    workspace_root: Path,
    image_tag: str,
    source_dir: Path,
    output_dir: Path,
    target: str,
    verbose: bool,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{format_docker_mount_source(workspace_root)}:{DEFAULT_CONTAINER_WORKSPACE.as_posix()}",
        "--workdir",
        DEFAULT_CONTAINER_WORKSPACE.as_posix(),
        "--entrypoint",
        "python",
    ]

    next_mount_index = 0
    source_mount_args, mapped_source_dir = map_host_path_to_container(
        host_path=source_dir,
        workspace_root=workspace_root,
        treat_as_file=False,
        mount_index=next_mount_index,
    )
    command.extend(source_mount_args)
    if source_mount_args:
        next_mount_index += 1

    output_mount_args, mapped_output_dir = map_host_path_to_container(
        host_path=output_dir,
        workspace_root=workspace_root,
        treat_as_file=False,
        mount_index=next_mount_index,
    )
    command.extend(output_mount_args)

    command.append(image_tag)
    command.extend(
        [
            "scripts/delivery/asr_rknn_export.py",
            "--source-dir",
            mapped_source_dir,
            "--output-dir",
            mapped_output_dir,
            "--target",
            target,
        ]
    )
    if verbose:
        command.append("--verbose")
    return command


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
    except OSError as exc:
        command_text = " ".join(str(part) for part in command)
        raise ASRRKNNExportError(f"命令执行失败：{command_text}\n{exc}") from exc


def docker_image_exists(image_tag: str) -> bool:
    completed = run_command(["docker", "image", "inspect", image_tag])
    return completed.returncode == 0


def ensure_docker_image(image_tag: str) -> None:
    if docker_image_exists(image_tag):
        return
    log(f"Docker 镜像不存在，开始构建：{image_tag}")
    completed = run_command(build_docker_build_command(image_tag=image_tag))
    if completed.returncode != 0:
        raise ASRRKNNExportError(
            "Docker 镜像构建失败：\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )


def export_streaming_zipformer_models_via_docker(
    source_dir: Path,
    output_dir: Path,
    *,
    workspace_root: Path = WORKSPACE_ROOT,
    image_tag: str = RKNN_TOOLKIT2_IMAGE_TAG,
    target: str = RKNN_TOOLKIT2_TARGET_PLATFORM,
    verbose: bool = False,
) -> None:
    ensure_docker_image(image_tag)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    command = build_docker_run_command(
        workspace_root=workspace_root.resolve(),
        image_tag=image_tag,
        source_dir=source_dir.resolve(),
        output_dir=output_dir.resolve(),
        target=target,
        verbose=verbose,
    )
    completed = run_command(command)
    if completed.returncode != 0:
        raise ASRRKNNExportError(
            "Docker RKNN 导出失败：\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )


def materialize_streaming_zipformer_rknn(
    source_dir: Path,
    output_dir: Path,
    *,
    workspace_root: Path = WORKSPACE_ROOT,
    target: str = RKNN_TOOLKIT2_TARGET_PLATFORM,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    required_outputs = (
        output_dir / "encoder.rknn",
        output_dir / "decoder.rknn",
        output_dir / "joiner.rknn",
        output_dir / "tokens.txt",
        output_dir / BUILD_MANIFEST_NAME,
    )
    if not force and all(path.exists() for path in required_outputs):
        log(f"Reusing existing local RKNN export: {output_dir}")
        return output_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)

    if has_local_rknn_toolchain():
        log(f"Using local rknn-toolkit2 to export ASR RKNN models from {source_dir}")
        export_streaming_zipformer_models(source_dir, output_dir, target=target, verbose=verbose)
    else:
        log(f"Local rknn-toolkit2 unavailable, exporting ASR RKNN models in Docker from {source_dir}")
        export_streaming_zipformer_models_via_docker(
            source_dir,
            output_dir,
            workspace_root=workspace_root,
            target=target,
            verbose=verbose,
        )
    return output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sherpa-onnx streaming Zipformer ONNX models to RKNN")
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory containing source ONNX model files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory receiving encoder/decoder/joiner RKNN files")
    parser.add_argument("--target", default=RKNN_TOOLKIT2_TARGET_PLATFORM, help="RKNN target platform")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose RKNN logging")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        export_streaming_zipformer_models(
            args.source_dir.resolve(),
            args.output_dir.resolve(),
            target=args.target,
            verbose=args.verbose,
        )
    except ASRRKNNExportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
