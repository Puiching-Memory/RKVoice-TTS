from __future__ import annotations

from pathlib import Path

from tests.test_support import WorkspaceTestCase

from scripts.delivery.tts import build_runtime_bundle, materialize_runtime_support_files


class MeloTtsDeliveryTests(WorkspaceTestCase):
    def _create_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _create_stage_bundle(self, stage_dir: Path) -> None:
        source_dir = stage_dir / "snapshot" / "melotts_rknn2_upstream"
        wheel_dir = stage_dir / "wheels"
        self._create_file(source_dir / "melotts_rknn.py", "print('demo')\n")
        self._create_file(source_dir / "utils.py", "def helper():\n    return 1\n")
        self._create_file(source_dir / "requirements.txt", "numpy==1.24.4\n")
        self._create_file(source_dir / "encoder.onnx", "model\n")
        self._create_file(source_dir / "decoder.rknn", "model\n")
        self._create_file(source_dir / "g.bin", "blob\n")
        self._create_file(source_dir / "lexicon.txt", "lexicon\n")
        self._create_file(source_dir / "tokens.txt", "tokens\n")
        self._create_file(source_dir / "english_utils" / "__init__.py", "\n")
        self._create_file(source_dir / "text" / "__init__.py", "\n")
        self._create_file(wheel_dir / "pip-26.0.1-py3-none-any.whl", "wheel\n")
        self._create_file(wheel_dir / "numpy-1.24.4-cp310-cp310-manylinux2014_aarch64.whl", "wheel\n")
        self._create_file(wheel_dir / "onnxruntime-1.16.0-cp310-cp310-manylinux2014_aarch64.whl", "wheel\n")
        self._create_file(wheel_dir / "soundfile-0.13.1-py2.py3-none-any.whl", "wheel\n")
        self._create_file(wheel_dir / "cn2an-0.5.23-py3-none-any.whl", "wheel\n")
        self._create_file(wheel_dir / "inflect-7.5.0-py3-none-any.whl", "wheel\n")
        self._create_file(wheel_dir / "psutil-7.2.2-cp36-abi3-manylinux2014_aarch64.whl", "wheel\n")
        self._create_file(wheel_dir / "ruamel_yaml-0.19.1-py3-none-any.whl", "wheel\n")
        self._create_file(wheel_dir / "rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux2014_aarch64.whl", "wheel\n")

    def test_materialize_runtime_support_files_writes_runtime_layout(self) -> None:
        with self.temp_dir("rkvoice_melo_runtime_") as temp_dir:
            runtime_dir = temp_dir / "runtime"
            tts_runtime_dir = runtime_dir / "tts"

            materialize_runtime_support_files(runtime_dir)

            self.assertTrue((runtime_dir / "README_SDK.md").exists())
            self.assertTrue((tts_runtime_dir / "README_SDK.md").exists())
            self.assertTrue((tts_runtime_dir / "run_tts.sh").exists())
            self.assertTrue((tts_runtime_dir / "smoketest.sh").exists())
            self.assertTrue((tts_runtime_dir / "tools" / "board_profile_capabilities.sh").exists())
            self.assertTrue((tts_runtime_dir / "tools" / "check_python_env.sh").exists())
            self.assertTrue((tts_runtime_dir / "tools" / "install_python_deps.sh").exists())
            self.assertTrue((tts_runtime_dir / "tools" / "profile_tts_inference.sh").exists())

            run_script = (tts_runtime_dir / "run_tts.sh").read_text(encoding="utf-8")
            smoketest = (tts_runtime_dir / "smoketest.sh").read_text(encoding="utf-8")
            install_script = (tts_runtime_dir / "tools" / "install_python_deps.sh").read_text(encoding="utf-8")
            profile_script = (tts_runtime_dir / "tools" / "profile_tts_inference.sh").read_text(encoding="utf-8")
            self.assertIn("melotts_rknn.py", run_script)
            self.assertIn("RKVOICE_TTS_OUTPUT_WAV", run_script)
            self.assertIn("decoder.rknn", run_script)
            self.assertIn("pydeps", run_script)
            self.assertIn("check_python_env.sh", smoketest)
            self.assertIn("profile_tts_inference.sh", smoketest)
            self.assertIn("rknn-toolkit-lite2==2.3.2", install_script)
            self.assertIn("--no-deps", install_script)
            self.assertIn('"$RUNTIME_DIR/run_tts.sh" "$sentence"', profile_script)

    def test_build_runtime_bundle_assembles_upstream_assets(self) -> None:
        with self.temp_dir("rkvoice_melo_bundle_") as temp_dir:
            stage_dir = temp_dir / "stage"
            runtime_dir = temp_dir / "runtime"
            tts_runtime_dir = runtime_dir / "tts"
            self._create_stage_bundle(stage_dir)

            build_runtime_bundle(stage_dir, runtime_dir, force=False)

            self.assertTrue((runtime_dir / "README_SDK.md").exists())
            self.assertTrue((tts_runtime_dir / "melotts_rknn.py").exists())
            self.assertTrue((tts_runtime_dir / "utils.py").exists())
            self.assertTrue((tts_runtime_dir / "encoder.onnx").exists())
            self.assertTrue((tts_runtime_dir / "decoder.rknn").exists())
            self.assertTrue((tts_runtime_dir / "english_utils" / "__init__.py").exists())
            self.assertTrue((tts_runtime_dir / "text" / "__init__.py").exists())
            self.assertTrue((tts_runtime_dir / "wheels" / "pip-26.0.1-py3-none-any.whl").exists())
            self.assertTrue((tts_runtime_dir / "wheels" / "rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux2014_aarch64.whl").exists())
            self.assertTrue((tts_runtime_dir / "run_tts.sh").exists())
            self.assertTrue((tts_runtime_dir / "smoketest.sh").exists())
            self.assertTrue((tts_runtime_dir / "output").exists())