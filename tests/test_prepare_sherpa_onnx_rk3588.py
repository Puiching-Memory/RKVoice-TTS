from __future__ import annotations

from pathlib import Path
import tarfile
from unittest.mock import patch

from tests.test_support import WorkspaceTestCase

from scripts.delivery.asr import build_runtime_bundle, deploy_runtime_bundle, hydrate_streaming_source_tokens, materialize_runtime_support_files
from scripts.delivery.config import LEGACY_STREAMING_RKNN_ASSET_NAME
from scripts.delivery.config import AUDIOS_DIR


class SherpaOnnxDeliveryTests(WorkspaceTestCase):
    def _create_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _create_stage_bundle(self, stage_dir: Path, *, include_built_models: bool = True) -> None:
        self._create_file(stage_dir / "prebuilt" / "sherpa-onnx-runtime" / "bin" / "sherpa-onnx", "binary\n")
        self._create_file(stage_dir / "prebuilt" / "sherpa-onnx-runtime" / "lib" / "libsherpa-onnx-c-api.so", "shared\n")
        self._create_file(stage_dir / "prebuilt" / "sherpa-onnx-runtime" / "lib" / "libonnxruntime.so", "shared\n")
        self._create_file(stage_dir / "prebuilt" / "sherpa-onnx-runtime" / "include" / "sherpa-onnx" / "c-api" / "c-api.h", "header\n")

        source_dir = stage_dir / "source-models" / "asr" / "streaming-onnx" / "sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16"
        self._create_file(source_dir / "encoder-epoch-99-avg-1.int8.onnx", "onnx\n")
        self._create_file(source_dir / "decoder-epoch-99-avg-1.onnx", "onnx\n")
        self._create_file(source_dir / "joiner-epoch-99-avg-1.int8.onnx", "onnx\n")
        self._create_file(source_dir / "tokens.txt", "tokens\n")
        self._create_file(source_dir / "test_wavs" / "0.wav", "wav\n")

        if include_built_models:
            self._create_file(stage_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "encoder.rknn", "model\n")
            self._create_file(stage_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "decoder.rknn", "model\n")
            self._create_file(stage_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "joiner.rknn", "model\n")
            self._create_file(stage_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "tokens.txt", "tokens\n")
            self._create_file(stage_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "rknn_build_manifest.json", "{}\n")

    def test_materialize_runtime_support_files_writes_runtime_layout(self) -> None:
        with self.temp_dir("rkvoice_sherpa_runtime_") as temp_dir:
            runtime_dir = temp_dir / "runtime"
            asr_runtime_dir = runtime_dir / "asr"

            materialize_runtime_support_files(runtime_dir)

            self.assertTrue((runtime_dir / "README_SDK.md").exists())
            self.assertTrue((asr_runtime_dir / "README_SDK.md").exists())
            self.assertTrue((asr_runtime_dir / "run_asr.sh").exists())
            self.assertTrue((asr_runtime_dir / "smoketest.sh").exists())
            self.assertTrue((asr_runtime_dir / "tools" / "check_rknn_env.sh").exists())
            self.assertTrue((asr_runtime_dir / "tools" / "board_profile_capabilities.sh").exists())

            run_asr = (asr_runtime_dir / "run_asr.sh").read_text(encoding="utf-8")
            smoketest = (asr_runtime_dir / "smoketest.sh").read_text(encoding="utf-8")
            profile_script = (asr_runtime_dir / "tools" / "profile_asr_inference.sh").read_text(encoding="utf-8")

            self.assertIn("--provider=rknn", run_asr)
            self.assertIn("streaming-zipformer-rk3588-small", run_asr)
            self.assertIn("audios", run_asr)
            self.assertIn("test_wavs", run_asr)
            self.assertIn("head -1", run_asr)
            self.assertIn("./tools/check_rknn_env.sh", smoketest)
            self.assertIn("head -1", smoketest)
            self.assertIn("Elapsed seconds:", smoketest)
            self.assertIn("Audio duration:", smoketest)
            self.assertIn("Real time factor (RTF):", smoketest)
            self.assertIn("run_asr.sh", profile_script)
            self.assertIn("head -1", profile_script)
            self.assertIn("RKNN_LOG_LEVEL", profile_script)
            self.assertIn("rknn_runtime.log", profile_script)
            self.assertIn("rknpu_load.log", profile_script)

    def test_hydrate_streaming_source_tokens_bootstraps_from_legacy_archive(self) -> None:
        with self.temp_dir("rkvoice_sherpa_tokens_") as temp_dir:
            stage_dir = temp_dir / "stage"
            cache_dir = temp_dir / "cache"
            source_dir = stage_dir / "source-models" / "asr" / "streaming-onnx" / "sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16"
            source_dir.mkdir(parents=True, exist_ok=True)

            legacy_root = temp_dir / "legacy" / "sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16"
            self._create_file(legacy_root / "tokens.txt", "legacy-tokens\n")
            cache_dir.mkdir(parents=True, exist_ok=True)
            legacy_archive = cache_dir / LEGACY_STREAMING_RKNN_ASSET_NAME
            with tarfile.open(legacy_archive, "w:bz2") as archive:
                archive.add(legacy_root.parent, arcname=legacy_root.parent.name)

            tokens_path = hydrate_streaming_source_tokens(stage_dir, cache_dir)

            self.assertEqual(tokens_path.read_text(encoding="utf-8"), "legacy-tokens\n")

    def test_build_runtime_bundle_assembles_prebuilt_assets(self) -> None:
        with self.temp_dir("rkvoice_sherpa_bundle_") as temp_dir:
            stage_dir = temp_dir / "stage"
            runtime_dir = temp_dir / "runtime"
            asr_runtime_dir = runtime_dir / "asr"
            self._create_stage_bundle(stage_dir)

            build_runtime_bundle(stage_dir, runtime_dir, force=False)

            self.assertTrue((runtime_dir / "README_SDK.md").exists())
            self.assertTrue((asr_runtime_dir / "bin" / "sherpa-onnx").exists())
            self.assertTrue((asr_runtime_dir / "lib" / "libsherpa-onnx-c-api.so").exists())
            self.assertTrue((asr_runtime_dir / "include" / "sherpa-onnx" / "c-api" / "c-api.h").exists())
            self.assertTrue((asr_runtime_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "encoder.rknn").exists())
            self.assertTrue((asr_runtime_dir / "audios").exists())
            self.assertTrue(any((asr_runtime_dir / "audios").glob("*.wav")))
            self.assertTrue((asr_runtime_dir / "smoketest.sh").exists())
            self.assertTrue((asr_runtime_dir / "tools" / "check_rknn_env.sh").exists())

    def test_build_runtime_bundle_exports_rknn_from_onnx_sources(self) -> None:
        with self.temp_dir("rkvoice_sherpa_export_") as temp_dir:
            stage_dir = temp_dir / "stage"
            runtime_dir = temp_dir / "runtime"
            self._create_stage_bundle(stage_dir, include_built_models=False)

            def fake_export(source_dir: Path, output_dir: Path, *, workspace_root: Path, target: str, force: bool) -> Path:
                self.assertIn("source-models", source_dir.as_posix())
                self.assertEqual(target, "rk3588")
                self._create_file(output_dir / "encoder.rknn", "model\n")
                self._create_file(output_dir / "decoder.rknn", "model\n")
                self._create_file(output_dir / "joiner.rknn", "model\n")
                self._create_file(output_dir / "tokens.txt", "tokens\n")
                self._create_file(output_dir / "test_wavs" / "0.wav", "wav\n")
                self._create_file(output_dir / "rknn_build_manifest.json", "{}\n")
                return output_dir

            with patch("scripts.delivery.asr.materialize_streaming_zipformer_rknn", side_effect=fake_export) as export_mock:
                build_runtime_bundle(stage_dir, runtime_dir, force=False)

            self.assertTrue(export_mock.called)
            self.assertTrue((runtime_dir / "asr" / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "encoder.rknn").exists())

    def test_deploy_runtime_bundle_downloads_smoke_log_even_when_remote_smoke_fails(self) -> None:
        with self.temp_dir("rkvoice_sherpa_upload_") as temp_dir:
            stage_dir = temp_dir / "stage"
            runtime_dir = temp_dir / "runtime"
            self._create_stage_bundle(stage_dir)
            build_runtime_bundle(stage_dir, runtime_dir, force=False)

            tarball_path = temp_dir / "asr.tar.gz"
            tarball_path.write_text("archive\n", encoding="utf-8")

            class DummyClient:
                def close(self) -> None:
                    return None

            def fake_run_remote_command(_client, command: str, *, timeout: int) -> str:
                if "smoketest.sh" in command:
                    raise SystemExit(1)
                return ""

            def fake_download_file(_client, remote_path: str, local_path: Path) -> None:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                if remote_path.endswith("smoke_test_summary.log"):
                    local_path.write_text("Elapsed seconds: 0.321 s\n", encoding="utf-8")
                    return
                if remote_path.endswith("board_profile_capabilities.txt"):
                    local_path.write_text("librknnrt version: 2.3.2\n", encoding="utf-8")
                    return
                raise OSError("missing remote artifact")

            with patch("scripts.delivery.asr.open_ssh_client", return_value=DummyClient()), \
                patch("scripts.delivery.asr.create_bundle_tarball", return_value=tarball_path), \
                patch("scripts.delivery.asr.upload_file"), \
                patch("scripts.delivery.asr.run_remote_command", side_effect=fake_run_remote_command), \
                patch("scripts.delivery.asr.download_file", side_effect=fake_download_file):
                with self.assertRaises(SystemExit):
                    deploy_runtime_bundle(
                        runtime_dir,
                        "/root/rkvoice/rkvoice_runtime",
                        host="169.254.46.2",
                        username="root",
                        password="123456",
                        source_ip=None,
                        ssh_timeout=8,
                        remote_timeout=60,
                        skip_smoketest=False,
                        enable_rknn_smoketest=True,
                    )

            local_output_dir = runtime_dir / "asr" / "output"
            self.assertEqual((local_output_dir / "smoke_test_summary.log").read_text(encoding="utf-8"), "Elapsed seconds: 0.321 s\n")
            self.assertEqual((local_output_dir / "board_profile_capabilities.txt").read_text(encoding="utf-8"), "librknnrt version: 2.3.2\n")