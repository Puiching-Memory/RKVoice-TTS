from __future__ import annotations

from tests.test_support import WorkspaceTestCase

from scripts.delivery.paddlespeech_tts_armlinux import (  # noqa: E402
    materialize_runtime_support_files,
    patch_for_offline_build,
)


class DeliverySdkTests(WorkspaceTestCase):
    def test_materialize_runtime_support_files_writes_sdk_layout(self) -> None:
        with self.temp_dir("rkvoice_tts_runtime_") as temp_dir:
            runtime_dir = temp_dir / "runtime"

            materialize_runtime_support_files(runtime_dir)

            self.assertTrue((runtime_dir / "README_SDK.md").exists())
            self.assertTrue((runtime_dir / "include" / "rkvoice_tts_api.h").exists())
            self.assertTrue((runtime_dir / "examples" / "c_api_demo.c").exists())
            self.assertTrue((runtime_dir / "tools" / "board_profile_capabilities.sh").exists())

            run_script = (runtime_dir / "run_tts.sh").read_text(encoding="utf-8")
            self.assertIn("RKVOICE_TTS_BACKEND", run_script)
            self.assertIn("bin/rkvoice_tts_demo", run_script)

            profile_script = (runtime_dir / "tools" / "profile_tts_inference.sh").read_text(encoding="utf-8")
            c_api_demo = (runtime_dir / "examples" / "c_api_demo.c").read_text(encoding="utf-8")
            self.assertIn("printf '%s\\n' \"$load_line\"", profile_script)
            self.assertIn("npu_core2_percent\\n", profile_script)
            self.assertIn('"create failed: %s\\n"', c_api_demo)
            self.assertIn('rtf=%.6f\\n"', c_api_demo)

    def test_patch_for_offline_build_writes_sdk_sources(self) -> None:
        with self.temp_dir("rkvoice_tts_stage_") as stage_dir:
            (stage_dir / "src" / "TTSCppFrontend" / "third-party").mkdir(parents=True)
            (stage_dir / "src" / "TTSCppFrontend" / "front_demo").mkdir(parents=True)
            (stage_dir / "dict").mkdir(parents=True)

            (stage_dir / "src" / "TTSCppFrontend" / "CMakeLists.txt").write_text(
                """
set(ENV{PKG_CONFIG_PATH} \"${CMAKE_SOURCE_DIR}/third-party/build/lib/pkgconfig:${CMAKE_SOURCE_DIR}/third-party/build/lib64/pkgconfig\")
include_directories(
    ${CMAKE_SOURCE_DIR}/third-party/build/src/cppjieba/include
    ${CMAKE_SOURCE_DIR}/third-party/build/src/limonp/include
    ${CMAKE_CURRENT_LIST_DIR}/third-party/build/src/cppjieba/include
    ${CMAKE_CURRENT_LIST_DIR}/third-party/build/src/limonp/include
)
""",
                encoding="utf-8",
            )
            (stage_dir / "dict" / "placeholder.txt").write_text("demo", encoding="utf-8")

            patch_for_offline_build(stage_dir)

            root_cmakelists = (stage_dir / "src" / "CMakeLists.txt").read_text(encoding="utf-8")
            core_source = (stage_dir / "src" / "rkvoice_tts_core.cc").read_text(encoding="utf-8")
            self.assertIn("rkvoice_tts_shared", root_cmakelists)
            self.assertIn("rkvoice_tts_demo", root_cmakelists)
            self.assertTrue((stage_dir / "src" / "rkvoice_tts_api.h").exists())
            self.assertTrue((stage_dir / "src" / "rkvoice_tts_core.cc").exists())
            self.assertTrue((stage_dir / "src" / "rkvoice_tts_demo_main.cc").exists())
            self.assertTrue((stage_dir / "offline_build.sh").exists())
            self.assertIn("value[0] != '\\0'", core_source)
            self.assertNotIn("\x00", core_source)
