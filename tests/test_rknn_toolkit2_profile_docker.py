from __future__ import annotations

from pathlib import Path

from scripts.testing.rknn_toolkit2_profile_in_docker import (
    DEFAULT_BOARD_PREPARE_SCRIPT,
    DEFAULT_BUILD_CONTEXT,
    DEFAULT_CONTAINER_WORKSPACE,
    DEFAULT_DOCKERFILE,
    build_prepare_board_command,
    build_docker_build_command,
    build_docker_run_command,
)
from tests.test_support import WorkspaceTestCase


class RKNNToolkit2ProfileDockerTests(WorkspaceTestCase):
    def test_build_prepare_board_command_targets_board_helper(self) -> None:
        command = build_prepare_board_command()

        self.assertTrue(command[0].lower().endswith("python.exe") or command[0].endswith("python"))
        self.assertEqual(Path(command[1]).resolve(), DEFAULT_BOARD_PREPARE_SCRIPT.resolve())

    def test_build_docker_build_command_targets_toolkit2_image(self) -> None:
        command = build_docker_build_command(image_tag="rkvoice/rknn-toolkit2-profile:test")

        self.assertEqual(command[:4], ["docker", "build", "-t", "rkvoice/rknn-toolkit2-profile:test"])
        self.assertEqual(Path(command[5]).resolve(), DEFAULT_DOCKERFILE.resolve())
        self.assertEqual(Path(command[-1]).resolve(), DEFAULT_BUILD_CONTEXT.resolve())

    def test_build_docker_run_command_mounts_workspace_and_model(self) -> None:
        workspace_root = Path(self.workspace_root)
        model_path = workspace_root / "artifacts" / "runtime" / "sherpa_onnx_rk3588_runtime" / "models" / "asr" / "rknn" / "sense-voice-rk3588-20s" / "model.rknn"
        output_dir = workspace_root / "artifacts" / "runtime" / "sherpa_onnx_rk3588_runtime" / "output"

        command = build_docker_run_command(
            workspace_root=workspace_root,
            image_tag="rkvoice/rknn-toolkit2-profile:test",
            model_path=model_path,
            output_dir=output_dir,
            target="rk3588",
            device_id="device-1",
            adb_connect="192.168.1.10:5555",
            adb_serial="SER123",
            verbose=True,
        )

        self.assertEqual(command[0:3], ["docker", "run", "--rm"])
        self.assertIn(f"{workspace_root.resolve().as_posix()}:{DEFAULT_CONTAINER_WORKSPACE.as_posix()}", command)
        self.assertIn("--model", command)
        self.assertIn("/workspace/artifacts/runtime/sherpa_onnx_rk3588_runtime/models/asr/rknn/sense-voice-rk3588-20s/model.rknn", command)
        self.assertIn("--output-dir", command)
        self.assertIn("/workspace/artifacts/runtime/sherpa_onnx_rk3588_runtime/output", command)
        self.assertIn("--device-id", command)
        self.assertIn("device-1", command)
        self.assertIn("--adb-connect", command)
        self.assertIn("192.168.1.10:5555", command)
        self.assertIn("--adb-serial", command)
        self.assertIn("SER123", command)
        self.assertIn("--verbose", command)

    def test_build_docker_run_command_mounts_external_model_and_output(self) -> None:
        with self.temp_dir("rkvoice_toolkit2_docker_") as temp_dir:
            workspace_root = temp_dir / "workspace"
            workspace_root.mkdir(parents=True)

            external_model = temp_dir / "models" / "demo" / "model.rknn"
            external_model.parent.mkdir(parents=True)
            external_model.write_text("model", encoding="utf-8")

            external_output = temp_dir / "output" / "profile"

            command = build_docker_run_command(
                workspace_root=workspace_root,
                image_tag="rkvoice/rknn-toolkit2-profile:test",
                model_path=external_model,
                output_dir=external_output,
                target="rk3588",
                device_id="",
                adb_connect="",
                adb_serial="",
                verbose=False,
            )

            self.assertIn("/mnt/external/0/model.rknn", command)
            self.assertIn("/mnt/external/1/output/profile", command)