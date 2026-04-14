from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from scripts.delivery.asr_rknn_export import (
    BUILD_MANIFEST_NAME,
    DEFAULT_CONTAINER_WORKSPACE,
    ASRRKNNExportError,
    build_docker_run_command,
    build_encoder_custom_string,
    find_source_files,
    resolve_dim_value,
)
from tests.test_support import WorkspaceTestCase


class ASRRKNNExportTests(WorkspaceTestCase):
    def test_find_source_files_supports_sherpa_streaming_package_layout(self) -> None:
        with self.temp_dir("rkvoice_asr_export_source_") as temp_dir:
            source_dir = temp_dir / "sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16"
            source_dir.mkdir(parents=True)
            variant_dir = source_dir / "64"
            variant_dir.mkdir(parents=True)
            (variant_dir / "encoder-epoch-99-avg-1.onnx").write_text("encoder-fp32", encoding="utf-8")
            (variant_dir / "encoder-epoch-99-avg-1.int8.onnx").write_text("encoder", encoding="utf-8")
            (variant_dir / "decoder-epoch-99-avg-1.onnx").write_text("decoder", encoding="utf-8")
            (variant_dir / "joiner-epoch-99-avg-1.onnx").write_text("joiner-fp32", encoding="utf-8")
            (variant_dir / "joiner-epoch-99-avg-1.int8.onnx").write_text("joiner", encoding="utf-8")
            (variant_dir / "tokens.txt").write_text("tokens", encoding="utf-8")
            (source_dir / "test_wavs").mkdir()

            resolved = find_source_files(source_dir)

            self.assertEqual(resolved["encoder"].name, "encoder-epoch-99-avg-1.onnx")
            self.assertEqual(resolved["decoder"].name, "decoder-epoch-99-avg-1.onnx")
            self.assertEqual(resolved["joiner"].name, "joiner-epoch-99-avg-1.onnx")
            self.assertEqual(resolved["tokens"].name, "tokens.txt")
            self.assertEqual(resolved["test_wavs"].name, "test_wavs")
            self.assertEqual(resolved["variant_root"].name, "64")

    def test_resolve_dim_value_replaces_symbolic_dims(self) -> None:
        class FakeDim:
            def __init__(self, *, dim_value: int = 0, dim_param: str = "") -> None:
                self.dim_value = dim_value
                self.dim_param = dim_param

        self.assertEqual(resolve_dim_value(FakeDim(dim_value=103)), 103)
        self.assertEqual(resolve_dim_value(FakeDim(dim_param="N")), 1)
        self.assertEqual(resolve_dim_value(FakeDim(dim_param="UNKNOWN")), 1)

    def test_build_encoder_custom_string_uses_encoder_and_decoder_metadata(self) -> None:
        encoder_metadata = {
            "model_type": "zipformer",
            "attention_dims": "192,192,192,192,192",
            "encoder_dims": "256,256,256,256,256",
            "T": "71",
            "left_context_len": "128,64,32,16,64",
            "decode_chunk_len": "64",
            "cnn_module_kernels": "31,31,31,31,31",
            "num_encoder_layers": "2,2,2,2,2",
        }
        decoder_metadata = {
            "context_size": "2",
        }

        custom_string = build_encoder_custom_string(encoder_metadata, decoder_metadata)

        self.assertEqual(
            custom_string,
            "model_type=zipformer;attention_dims=192,192,192,192,192;encoder_dims=256,256,256,256,256;T=71;left_context_len=128,64,32,16,64;decode_chunk_len=64;cnn_module_kernels=31,31,31,31,31;num_encoder_layers=2,2,2,2,2;context_size=2",
        )

    def test_build_encoder_custom_string_requires_context_size(self) -> None:
        with self.assertRaises(ASRRKNNExportError):
            build_encoder_custom_string(
                {
                    "model_type": "zipformer",
                    "attention_dims": "192,192,192,192,192",
                    "encoder_dims": "256,256,256,256,256",
                    "T": "71",
                    "left_context_len": "128,64,32,16,64",
                    "decode_chunk_len": "64",
                    "cnn_module_kernels": "31,31,31,31,31",
                    "num_encoder_layers": "2,2,2,2,2",
                },
                {},
            )

    def test_build_docker_run_command_uses_python_entrypoint(self) -> None:
        workspace_root = Path(self.workspace_root)
        source_dir = workspace_root / "artifacts" / "source-bundles" / "sherpa_onnx_rk3588" / "source-models" / "asr" / "streaming-onnx" / "demo"
        output_dir = workspace_root / "artifacts" / "source-bundles" / "sherpa_onnx_rk3588" / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small"

        command = build_docker_run_command(
            workspace_root=workspace_root,
            image_tag="rkvoice/rknn-toolkit2-profile:test",
            source_dir=source_dir,
            output_dir=output_dir,
            target="rk3588",
            verbose=True,
        )

        self.assertEqual(command[:3], ["docker", "run", "--rm"])
        self.assertIn("--entrypoint", command)
        self.assertIn("python", command)
        self.assertIn("scripts/delivery/asr_rknn_export.py", command)
        self.assertIn("--source-dir", command)
        self.assertIn((DEFAULT_CONTAINER_WORKSPACE / "artifacts" / "source-bundles" / "sherpa_onnx_rk3588" / "source-models" / "asr" / "streaming-onnx" / "demo").as_posix(), command)
        self.assertIn("--output-dir", command)
        self.assertIn((DEFAULT_CONTAINER_WORKSPACE / "artifacts" / "source-bundles" / "sherpa_onnx_rk3588" / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small").as_posix(), command)
        self.assertIn("--target", command)
        self.assertIn("rk3588", command)
        self.assertIn("--verbose", command)

    def test_build_manifest_name_is_stable(self) -> None:
        self.assertEqual(BUILD_MANIFEST_NAME, "rknn_build_manifest.json")

    def test_importing_delivery_config_does_not_require_paramiko(self) -> None:
        code = """
import builtins
real_import = builtins.__import__

def blocked(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'paramiko':
        raise ModuleNotFoundError('No module named paramiko')
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked
from scripts.delivery.config import RKNN_TOOLKIT2_TARGET_PLATFORM
print(RKNN_TOOLKIT2_TARGET_PLATFORM)
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("rk3588", completed.stdout)
