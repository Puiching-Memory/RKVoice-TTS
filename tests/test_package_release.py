from __future__ import annotations

import zipfile
from pathlib import Path

from tests.test_support import WorkspaceTestCase
from scripts.release.package_release import build_release


class PackageReleaseTests(WorkspaceTestCase):
    def _create_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _create_workspace(self, workspace_root: Path) -> None:
        self._create_file(workspace_root / ".gitignore", "artifacts/\n")
        self._create_file(workspace_root / "README.md", "# demo\n")
        self._create_file(workspace_root / "pyproject.toml", "[project]\nname='demo'\n")
        self._create_file(workspace_root / "uv.lock", "version = 1\n")
        self._create_file(workspace_root / "config" / "examples" / "demo.env", "KEY=value\n")
        self._create_file(
            workspace_root / "docs" / "delivery" / "发布说明模板.md",
            "包={{PACKAGE_NAME}}\n版本={{VERSION}}\n时间={{BUILD_TIMESTAMP}}\n目录={{RELEASE_DIRECTORY}}\n压缩包={{ZIP_PATH}}\n",
        )
        self._create_file(workspace_root / "docs" / "delivery" / "发布手册.md", "manual\n")
        self._create_file(workspace_root / "scripts" / "release" / "package_release.py", "print('demo')\n")
        self._create_file(workspace_root / "scripts" / "delivery" / "helper.py", "print('helper')\n")

    def test_build_release_creates_manifest_release_notes_and_zip(self) -> None:
        with self.temp_dir("rkvoice_tts_release_") as temp_dir:
            workspace_root = temp_dir / "workspace"
            output_root = temp_dir / "out"
            self._create_workspace(workspace_root)

            result = build_release(
                workspace_root=workspace_root,
                output_root=output_root,
                package_name="rk3588-asr-tts-delivery",
                version="v1.0.0",
            )

            self.assertTrue(result.release_dir.exists())
            self.assertTrue(result.zip_path.exists())
            self.assertTrue((result.release_dir / "README.md").exists())
            self.assertTrue((result.release_dir / "scripts" / "release" / "package_release.py").exists())

            release_notes = (result.release_dir / "RELEASE_NOTES.md").read_text(encoding="utf-8")
            manifest = result.manifest_path.read_text(encoding="utf-8")
            self.assertIn("包=rk3588-asr-tts-delivery", release_notes)
            self.assertIn("版本=v1.0.0", release_notes)
            self.assertIn("Version=v1.0.0", manifest)
            self.assertIn("IncludeRuntimeBundle=False", manifest)
            self.assertIn("IncludeEvidence=False", manifest)
            self.assertIn("- scripts", manifest)

            with zipfile.ZipFile(result.zip_path) as archive:
                archive_names = set(archive.namelist())
            self.assertIn(f"{result.release_dir.name}/README.md", archive_names)
            self.assertIn(f"{result.release_dir.name}/RELEASE_NOTES.md", archive_names)

    def test_build_release_includes_optional_runtime_and_evidence(self) -> None:
        with self.temp_dir("rkvoice_tts_release_optional_") as temp_dir:
            workspace_root = temp_dir / "workspace"
            output_root = temp_dir / "out"
            self._create_workspace(workspace_root)
            self._create_file(
                workspace_root / "artifacts" / "runtime" / "rkvoice_runtime.tar.gz",
                "runtime bundle\n",
            )
            self._create_file(
                workspace_root / "artifacts" / "runtime" / "rkvoice_runtime" / "asr" / "output" / "smoke_test.log",
                "ok\n",
            )
            self._create_file(
                workspace_root / "artifacts" / "runtime" / "rkvoice_runtime" / "tts" / "run_tts.sh",
                "#!/bin/bash\n",
            )
            self._create_file(
                workspace_root / "artifacts" / "runtime" / "rkvoice_runtime" / "tts" / "output" / "smoke_test_tts.wav",
                "wav\n",
            )
            self._create_file(
                workspace_root / "custom_release_notes.md",
                "自定义版本={{VERSION}}\n",
            )

            result = build_release(
                workspace_root=workspace_root,
                output_root=output_root,
                package_name="demo package",
                version="release candidate",
                release_notes_path=str(workspace_root / "custom_release_notes.md"),
                include_runtime_bundle=True,
                include_evidence=True,
            )

            self.assertTrue(
                (
                    result.release_dir
                    / "artifacts"
                    / "runtime"
                    / "rkvoice_runtime.tar.gz"
                ).exists()
            )
            self.assertTrue(
                (
                    result.release_dir
                    / "artifacts"
                    / "runtime"
                    / "rkvoice_runtime"
                    / "asr"
                    / "output"
                    / "smoke_test.log"
                ).exists()
            )
            self.assertTrue(
                (
                    result.release_dir
                    / "artifacts"
                    / "runtime"
                    / "rkvoice_runtime"
                    / "tts"
                    / "output"
                    / "smoke_test_tts.wav"
                ).exists()
            )
            manifest = result.manifest_path.read_text(encoding="utf-8")
            release_notes = (result.release_dir / "RELEASE_NOTES.md").read_text(encoding="utf-8")
            self.assertIn("PackageLabel=demo-package", manifest)
            self.assertIn("IncludeRuntimeBundle=True", manifest)
            self.assertIn("IncludeEvidence=True", manifest)
            self.assertIn("自定义版本=release candidate", release_notes)