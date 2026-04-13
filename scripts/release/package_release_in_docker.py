from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_TAG = "rkvoice/package-release:py312"
DEFAULT_CONTAINER_WORKSPACE = Path("/workspace")
DEFAULT_DOCKERFILE = WORKSPACE_ROOT / "docker" / "package-release" / "Dockerfile"
DEFAULT_BUILD_CONTEXT = DEFAULT_DOCKERFILE.parent

try:
    from .package_release import DEFAULT_PACKAGE_NAME
except ImportError:
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))
    from scripts.release.package_release import DEFAULT_PACKAGE_NAME


class DockerReleaseError(Exception):
    pass


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
            raise DockerReleaseError(f"无法为容器映射路径：{resolved}")
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


def build_release_args(
    *,
    output_root: str | None,
    package_name: str,
    version: str,
    release_notes_path: str | None,
    include_runtime_bundle: bool,
    include_evidence: bool,
) -> list[str]:
    args = ["--package-name", package_name]
    if version:
        args.extend(["--version", version])
    if output_root:
        args.extend(["--output-root", output_root])
    if release_notes_path:
        args.extend(["--release-notes-path", release_notes_path])
    if include_runtime_bundle:
        args.append("--include-runtime-bundle")
    if include_evidence:
        args.append("--include-evidence")
    return args


def build_docker_build_command(
    *,
    image_tag: str,
    dockerfile_path: Path = DEFAULT_DOCKERFILE,
    context_dir: Path = DEFAULT_BUILD_CONTEXT,
) -> list[str]:
    return [
        "docker",
        "build",
        "-t",
        image_tag,
        "-f",
        str(dockerfile_path),
        str(context_dir),
    ]


def build_docker_run_command(
    *,
    workspace_root: Path,
    image_tag: str,
    output_root: Path | None,
    package_name: str,
    version: str,
    release_notes_path: Path | None,
    include_runtime_bundle: bool,
    include_evidence: bool,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{format_docker_mount_source(workspace_root)}:{DEFAULT_CONTAINER_WORKSPACE.as_posix()}",
        "--workdir",
        DEFAULT_CONTAINER_WORKSPACE.as_posix(),
    ]

    if os.name != "nt" and hasattr(os, "getuid") and hasattr(os, "getgid"):
        command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])

    next_mount_index = 0
    mapped_output_root: str | None = None
    mapped_release_notes_path: str | None = None

    if output_root is not None:
        mount_args, mapped_output_root = map_host_path_to_container(
            host_path=output_root,
            workspace_root=workspace_root,
            treat_as_file=False,
            mount_index=next_mount_index,
        )
        command.extend(mount_args)
        if mount_args:
            next_mount_index += 1

    if release_notes_path is not None:
        mount_args, mapped_release_notes_path = map_host_path_to_container(
            host_path=release_notes_path,
            workspace_root=workspace_root,
            treat_as_file=True,
            mount_index=next_mount_index,
        )
        command.extend(mount_args)
        if mount_args:
            next_mount_index += 1

    command.append(image_tag)
    command.extend(
        build_release_args(
            output_root=mapped_output_root,
            package_name=package_name,
            version=version,
            release_notes_path=mapped_release_notes_path,
            include_runtime_bundle=include_runtime_bundle,
            include_evidence=include_evidence,
        )
    )
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RKVoice release packager inside Docker")
    parser.add_argument("--workspace-dir", type=Path, default=WORKSPACE_ROOT, help="Workspace root to mount into the container")
    parser.add_argument("--image-tag", default=DEFAULT_IMAGE_TAG, help="Docker image tag used for release packaging")
    parser.add_argument("--skip-image-build", action="store_true", help="Reuse an existing image and skip docker build")
    parser.add_argument("--output-root", type=Path, default=None, help="Release output root directory on the host")
    parser.add_argument("--package-name", default=DEFAULT_PACKAGE_NAME, help="Release package prefix")
    parser.add_argument("--version", default="", help="Release version label")
    parser.add_argument("--release-notes-path", type=Path, default=None, help="Custom release notes template or rendered notes source")
    parser.add_argument("--include-runtime-bundle", action="store_true", help="Include artifacts/runtime/sherpa_onnx_rk3588_runtime.tar.gz")
    parser.add_argument("--include-evidence", action="store_true", help="Include artifacts/runtime/sherpa_onnx_rk3588_runtime/output")
    return parser.parse_args(argv)


def run_command(command: Sequence[str]) -> int:
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    workspace_root = args.workspace_dir.resolve()
    if not workspace_root.exists():
        print(f"Workspace directory does not exist: {workspace_root}", file=sys.stderr)
        return 1

    if not args.skip_image_build:
        build_status = run_command(build_docker_build_command(image_tag=args.image_tag))
        if build_status != 0:
            return build_status

    run_status = run_command(
        build_docker_run_command(
            workspace_root=workspace_root,
            image_tag=args.image_tag,
            output_root=args.output_root,
            package_name=args.package_name,
            version=args.version,
            release_notes_path=args.release_notes_path,
            include_runtime_bundle=args.include_runtime_bundle,
            include_evidence=args.include_evidence,
        )
    )
    return run_status


if __name__ == "__main__":
    raise SystemExit(main())