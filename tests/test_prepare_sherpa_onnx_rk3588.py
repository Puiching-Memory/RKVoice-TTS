from __future__ import annotations

from pathlib import Path

from tests.test_support import WorkspaceTestCase

from scripts.delivery.asr import build_runtime_bundle, materialize_runtime_support_files
from scripts.delivery.config import AUDIOS_DIR


class SherpaOnnxDeliveryTests(WorkspaceTestCase):
    def _create_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _create_stage_bundle(self, stage_dir: Path) -> None:
        self._create_file(stage_dir / "prebuilt" / "sherpa-onnx-runtime" / "bin" / "sherpa-onnx", "binary\n")
        self._create_file(stage_dir / "prebuilt" / "sherpa-onnx-runtime" / "lib" / "libsherpa-onnx-c-api.so", "shared\n")
        self._create_file(stage_dir / "prebuilt" / "sherpa-onnx-runtime" / "lib" / "libonnxruntime.so", "shared\n")
        self._create_file(stage_dir / "prebuilt" / "sherpa-onnx-runtime" / "include" / "sherpa-onnx" / "c-api" / "c-api.h", "header\n")

        self._create_file(stage_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "encoder.rknn", "model\n")
        self._create_file(stage_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "decoder.rknn", "model\n")
        self._create_file(stage_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "joiner.rknn", "model\n")
        self._create_file(stage_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "tokens.txt", "tokens\n")

    def test_materialize_runtime_support_files_writes_runtime_layout(self) -> None:
        with self.temp_dir("rkvoice_sherpa_runtime_") as temp_dir:
            runtime_dir = temp_dir / "runtime"

            materialize_runtime_support_files(runtime_dir)

            self.assertTrue((runtime_dir / "README_SDK.md").exists())
            self.assertTrue((runtime_dir / "run_asr.sh").exists())
            self.assertTrue((runtime_dir / "smoketest.sh").exists())
            self.assertTrue((runtime_dir / "tools" / "check_rknn_env.sh").exists())
            self.assertTrue((runtime_dir / "tools" / "board_profile_capabilities.sh").exists())

            run_asr = (runtime_dir / "run_asr.sh").read_text(encoding="utf-8")
            smoketest = (runtime_dir / "smoketest.sh").read_text(encoding="utf-8")
            profile_script = (runtime_dir / "tools" / "profile_asr_inference.sh").read_text(encoding="utf-8")

            self.assertIn("--provider=rknn", run_asr)
            self.assertIn("streaming-zipformer-rk3588-small", run_asr)
            self.assertIn("audios", run_asr)
            self.assertIn("./tools/check_rknn_env.sh", smoketest)
            self.assertIn("audios", smoketest)
            self.assertIn("run_asr.sh", profile_script)
            self.assertIn("RKNN_LOG_LEVEL", profile_script)
            self.assertIn("rknn_runtime.log", profile_script)
            self.assertIn("rknpu_load.log", profile_script)

    def test_build_runtime_bundle_assembles_prebuilt_assets(self) -> None:
        with self.temp_dir("rkvoice_sherpa_bundle_") as temp_dir:
            stage_dir = temp_dir / "stage"
            runtime_dir = temp_dir / "runtime"
            self._create_stage_bundle(stage_dir)

            build_runtime_bundle(stage_dir, runtime_dir, force=False)

            self.assertTrue((runtime_dir / "bin" / "sherpa-onnx").exists())
            self.assertTrue((runtime_dir / "lib" / "libsherpa-onnx-c-api.so").exists())
            self.assertTrue((runtime_dir / "include" / "sherpa-onnx" / "c-api" / "c-api.h").exists())
            self.assertTrue((runtime_dir / "models" / "asr" / "streaming-rknn" / "streaming-zipformer-rk3588-small" / "encoder.rknn").exists())
            self.assertTrue((runtime_dir / "audios").exists())
            self.assertTrue(any((runtime_dir / "audios").glob("*.wav")))
            self.assertTrue((runtime_dir / "smoketest.sh").exists())
            self.assertTrue((runtime_dir / "tools" / "check_rknn_env.sh").exists())