from __future__ import annotations

from pathlib import Path

from tests.test_support import WorkspaceTestCase

from scripts.delivery.sherpa_onnx_rk3588 import build_runtime_bundle, materialize_runtime_support_files


class SherpaOnnxDeliveryTests(WorkspaceTestCase):
    def _create_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _create_stage_bundle(self, stage_dir: Path) -> None:
        self._create_file(stage_dir / "prebuilt" / "sherpa-onnx-runtime" / "bin" / "sherpa-onnx-offline", "binary\n")
        self._create_file(stage_dir / "prebuilt" / "sherpa-onnx-runtime" / "bin" / "sherpa-onnx-offline-tts", "binary\n")
        self._create_file(stage_dir / "prebuilt" / "sherpa-onnx-runtime" / "lib" / "libsherpa-onnx-c-api.so", "shared\n")
        self._create_file(stage_dir / "prebuilt" / "sherpa-onnx-runtime" / "lib" / "libonnxruntime.so", "shared\n")
        self._create_file(stage_dir / "prebuilt" / "sherpa-onnx-runtime" / "include" / "sherpa-onnx" / "c-api" / "c-api.h", "header\n")

        self._create_file(stage_dir / "models" / "asr" / "cpu" / "sense-voice" / "model.int8.onnx", "model\n")
        self._create_file(stage_dir / "models" / "asr" / "cpu" / "sense-voice" / "tokens.txt", "tokens\n")
        self._create_file(stage_dir / "models" / "asr" / "cpu" / "sense-voice" / "test_wavs" / "zh.wav", "wav\n")

        self._create_file(stage_dir / "models" / "asr" / "rknn" / "sense-voice-rk3588-20s" / "model.rknn", "model\n")
        self._create_file(stage_dir / "models" / "asr" / "rknn" / "sense-voice-rk3588-20s" / "tokens.txt", "tokens\n")
        self._create_file(stage_dir / "models" / "asr" / "rknn" / "sense-voice-rk3588-20s" / "test_wavs" / "zh.wav", "wav\n")

        self._create_file(stage_dir / "models" / "tts" / "vits-icefall-zh-aishell3" / "model.onnx", "model\n")
        self._create_file(stage_dir / "models" / "tts" / "vits-icefall-zh-aishell3" / "lexicon.txt", "lexicon\n")
        self._create_file(stage_dir / "models" / "tts" / "vits-icefall-zh-aishell3" / "tokens.txt", "tokens\n")
        self._create_file(stage_dir / "models" / "tts" / "vits-icefall-zh-aishell3" / "phone.fst", "fst\n")
        self._create_file(stage_dir / "models" / "tts" / "vits-icefall-zh-aishell3" / "date.fst", "fst\n")
        self._create_file(stage_dir / "models" / "tts" / "vits-icefall-zh-aishell3" / "number.fst", "fst\n")

    def test_materialize_runtime_support_files_writes_runtime_layout(self) -> None:
        with self.temp_dir("rkvoice_sherpa_runtime_") as temp_dir:
            runtime_dir = temp_dir / "runtime"

            materialize_runtime_support_files(runtime_dir)

            self.assertTrue((runtime_dir / "README_SDK.md").exists())
            self.assertTrue((runtime_dir / "run_asr.sh").exists())
            self.assertTrue((runtime_dir / "run_tts.sh").exists())
            self.assertTrue((runtime_dir / "smoketest.sh").exists())
            self.assertTrue((runtime_dir / "tools" / "check_rknn_env.sh").exists())
            self.assertTrue((runtime_dir / "tools" / "board_profile_capabilities.sh").exists())
            self.assertTrue((runtime_dir / "tools" / "profile_tts_inference.sh").exists())

            run_asr = (runtime_dir / "run_asr.sh").read_text(encoding="utf-8")
            run_tts = (runtime_dir / "run_tts.sh").read_text(encoding="utf-8")
            smoketest = (runtime_dir / "smoketest.sh").read_text(encoding="utf-8")
            profile_script = (runtime_dir / "tools" / "profile_asr_inference.sh").read_text(encoding="utf-8")
            tts_profile_script = (runtime_dir / "tools" / "profile_tts_inference.sh").read_text(encoding="utf-8")

            self.assertIn("RKVOICE_ASR_PROVIDER", run_asr)
            self.assertIn("sherpa-onnx-offline", run_asr)
            self.assertIn("sherpa-onnx-offline-tts", run_tts)
            self.assertIn("vits-icefall-zh-aishell3", run_tts)
            self.assertIn("RKVOICE_ENABLE_RKNN_SMOKETEST", smoketest)
            self.assertIn("./tools/check_rknn_env.sh", smoketest)
            self.assertIn("RKVOICE_ASR_PROVIDER=rknn", profile_script)
            self.assertIn("RKVOICE_TTS_OUTPUT_WAV", tts_profile_script)
            self.assertIn('"$RUNTIME_DIR/run_tts.sh" "$sentence"', tts_profile_script)

    def test_build_runtime_bundle_assembles_prebuilt_assets(self) -> None:
        with self.temp_dir("rkvoice_sherpa_bundle_") as temp_dir:
            stage_dir = temp_dir / "stage"
            runtime_dir = temp_dir / "runtime"
            self._create_stage_bundle(stage_dir)

            build_runtime_bundle(stage_dir, runtime_dir, force=False)

            self.assertTrue((runtime_dir / "bin" / "sherpa-onnx-offline").exists())
            self.assertTrue((runtime_dir / "bin" / "sherpa-onnx-offline-tts").exists())
            self.assertTrue((runtime_dir / "lib" / "libsherpa-onnx-c-api.so").exists())
            self.assertTrue((runtime_dir / "include" / "sherpa-onnx" / "c-api" / "c-api.h").exists())
            self.assertTrue((runtime_dir / "models" / "asr" / "cpu" / "sense-voice" / "test_wavs" / "zh.wav").exists())
            self.assertTrue((runtime_dir / "models" / "asr" / "rknn" / "sense-voice-rk3588-20s" / "model.rknn").exists())
            self.assertTrue((runtime_dir / "models" / "tts" / "vits-icefall-zh-aishell3" / "model.onnx").exists())
            self.assertTrue((runtime_dir / "smoketest.sh").exists())
            self.assertTrue((runtime_dir / "tools" / "check_rknn_env.sh").exists())