from __future__ import annotations

from pathlib import Path

from scripts.release.package_release_in_docker import (
    DEFAULT_BUILD_CONTEXT,
    DEFAULT_CONTAINER_WORKSPACE,
    DEFAULT_DOCKERFILE,
    build_docker_build_command,
    build_docker_run_command,
)
from tests.test_support import WorkspaceTestCase


class PackageReleaseDockerTests(WorkspaceTestCase):
    def test_build_docker_build_command_targets_dockerfile_context(self) -> None:
        command = build_docker_build_command(image_tag="rkvoice/test-release:py312")

        self.assertEqual(command[:4], ["docker", "build", "-t", "rkvoice/test-release:py312"])
        self.assertEqual(Path(command[5]).resolve(), DEFAULT_DOCKERFILE.resolve())
        self.assertEqual(Path(command[-1]).resolve(), DEFAULT_BUILD_CONTEXT.resolve())

    def test_build_docker_run_command_mounts_workspace_and_forwards_release_args(self) -> None:
        workspace_root = Path(self.workspace_root)

        command = build_docker_run_command(
            workspace_root=workspace_root,
            image_tag="rkvoice/test-release:py312",
            output_root=None,
            package_name="demo-package",
            version="v1.2.3",
            release_notes_path=None,
            include_runtime_bundle=True,
            include_evidence=True,
            include_melo_runtime_bundle=True,
            include_melo_evidence=True,
        )

        self.assertEqual(command[0:3], ["docker", "run", "--rm"])
        self.assertIn(f"{workspace_root.resolve().as_posix()}:{DEFAULT_CONTAINER_WORKSPACE.as_posix()}", command)
        self.assertIn("rkvoice/test-release:py312", command)
        self.assertIn("--package-name", command)
        self.assertIn("demo-package", command)
        self.assertIn("--version", command)
        self.assertIn("v1.2.3", command)
        self.assertIn("--include-runtime-bundle", command)
        self.assertIn("--include-evidence", command)
        self.assertIn("--include-melo-runtime-bundle", command)
        self.assertIn("--include-melo-evidence", command)

    def test_build_docker_run_command_mounts_external_paths(self) -> None:
        with self.temp_dir("rkvoice_release_docker_") as temp_dir:
            workspace_root = temp_dir / "workspace"
            workspace_root.mkdir(parents=True)

            external_release_notes = temp_dir / "notes" / "release.md"
            external_release_notes.parent.mkdir(parents=True)
            external_release_notes.write_text("demo", encoding="utf-8")

            external_output_root = temp_dir / "out" / "releases"

            command = build_docker_run_command(
                workspace_root=workspace_root,
                image_tag="rkvoice/test-release:py312",
                output_root=external_output_root,
                package_name="demo-package",
                version="",
                release_notes_path=external_release_notes,
                include_runtime_bundle=False,
                include_evidence=False,
                include_melo_runtime_bundle=False,
                include_melo_evidence=False,
            )

            self.assertIn("/mnt/external/0/out/releases", command)
            self.assertIn("/mnt/external/1/release.md", command)
            mount_args = [item for item in command if item.startswith(temp_dir.resolve().as_posix())]
            self.assertEqual(len(mount_args), 3)