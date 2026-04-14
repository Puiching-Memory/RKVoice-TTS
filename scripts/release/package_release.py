from __future__ import annotations

import argparse
import re
import shutil
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "artifacts" / "releases"
DEFAULT_PACKAGE_NAME = "rk3588-asr-tts-delivery"
DEFAULT_RELEASE_NOTES_RELATIVE_PATH = "docs\\delivery\\发布说明模板.md"
BASE_ITEMS = (
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "config\\examples",
    "docs",
    "scripts",
)
RUNTIME_DIR_RELATIVE_PATH = "artifacts\\runtime\\rkvoice_runtime"
RUNTIME_BUNDLE_RELATIVE_PATH = "artifacts\\runtime\\rkvoice_runtime.tar.gz"
RUNTIME_EVIDENCE_RELATIVE_PATHS = (
    "artifacts\\runtime\\rkvoice_runtime\\asr\\output",
    "artifacts\\runtime\\rkvoice_runtime\\tts\\output",
)
INVALID_PATH_SEGMENT_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WHITESPACE_PATTERN = re.compile(r"\s+")


class ReleasePackagingError(Exception):
    pass


@dataclass(frozen=True)
class ReleasePackageResult:
    package_name: str
    package_label: str
    version: str
    build_timestamp: str
    generated_at: str
    release_notes_source: Path
    release_dir: Path
    zip_path: Path
    manifest_path: Path
    included_items: tuple[str, ...]


def relative_path_to_path(relative_path: str) -> Path:
    return Path(*[part for part in re.split(r"[\\/]+", relative_path) if part])


def get_safe_path_segment(value: str) -> str:
    sanitized = value.strip()
    if not sanitized:
        return ""
    sanitized = INVALID_PATH_SEGMENT_PATTERN.sub("-", sanitized)
    sanitized = WHITESPACE_PATTERN.sub("-", sanitized)
    return sanitized.strip("-")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8", newline="\n")


def copy_workspace_item(*, relative_path: str, workspace_root: Path, release_dir: Path) -> bool:
    source = workspace_root / relative_path_to_path(relative_path)
    if not source.exists():
        return False

    destination = release_dir / relative_path_to_path(relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)
    return True


def copy_required_workspace_item(*, relative_path: str, workspace_root: Path, release_dir: Path) -> None:
    if not copy_workspace_item(relative_path=relative_path, workspace_root=workspace_root, release_dir=release_dir):
        raise ReleasePackagingError(f"发布内容缺失：{relative_path}")


def copy_or_archive_runtime_bundle(
    *,
    archive_relative_path: str,
    runtime_dir_relative_path: str,
    workspace_root: Path,
    release_dir: Path,
) -> None:
    archive_source = workspace_root / relative_path_to_path(archive_relative_path)
    destination = release_dir / relative_path_to_path(archive_relative_path)
    if archive_source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive_source, destination)
        return

    runtime_dir = workspace_root / relative_path_to_path(runtime_dir_relative_path)
    if not runtime_dir.exists():
        raise ReleasePackagingError(f"发布内容缺失：{archive_relative_path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(runtime_dir, arcname=runtime_dir.name)


def resolve_release_notes_source(*, workspace_root: Path, release_notes_path: str | None) -> Path:
    if release_notes_path:
        candidate = Path(release_notes_path).expanduser().resolve()
        if not candidate.exists():
            raise ReleasePackagingError(f"发布说明不存在：{candidate}")
        return candidate

    default_template = workspace_root / relative_path_to_path(DEFAULT_RELEASE_NOTES_RELATIVE_PATH)
    if not default_template.exists():
        raise ReleasePackagingError(f"缺少默认发布说明模板：{DEFAULT_RELEASE_NOTES_RELATIVE_PATH}")
    return default_template


def render_release_notes(
    *,
    source_path: Path,
    destination_path: Path,
    package_name: str,
    version: str,
    build_timestamp: str,
    generated_at: str,
    release_dir: Path,
    zip_path: Path,
) -> None:
    content = source_path.read_text(encoding="utf-8")
    rendered = (
        content.replace("{{PACKAGE_NAME}}", package_name)
        .replace("{{VERSION}}", version)
        .replace("{{BUILD_TIMESTAMP}}", build_timestamp)
        .replace("{{GENERATED_AT}}", generated_at)
        .replace("{{RELEASE_DIRECTORY}}", str(release_dir))
        .replace("{{ZIP_PATH}}", str(zip_path))
    )
    write_text(destination_path, rendered)


def write_release_manifest(
    *,
    destination_path: Path,
    generated_at: str,
    package_name: str,
    package_label: str,
    version: str,
    build_timestamp: str,
    release_notes_source: Path,
    release_dir: Path,
    zip_path: Path,
    include_runtime_bundle: bool,
    include_evidence: bool,
    included_items: Sequence[str],
) -> None:
    lines = [
        "# Release Manifest",
        "",
        f"GeneratedAt={generated_at}",
        f"PackageName={package_name}",
        f"PackageLabel={package_label}",
        f"Version={version}",
        f"BuildTimestamp={build_timestamp}",
        f"ReleaseNotesSource={release_notes_source}",
        f"ReleaseDirectory={release_dir}",
        f"ZipPath={zip_path}",
        f"IncludeRuntimeBundle={include_runtime_bundle}",
        f"IncludeEvidence={include_evidence}",
        "",
        "IncludedItems:",
    ]
    lines.extend(f"- {item}" for item in included_items)
    write_text(destination_path, "\n".join(lines) + "\n")


def create_zip_archive(*, release_dir: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()

    archive_path = Path(
        shutil.make_archive(
            base_name=str(zip_path.with_suffix("")),
            format="zip",
            root_dir=str(release_dir.parent),
            base_dir=release_dir.name,
        )
    )
    if archive_path != zip_path:
        if zip_path.exists():
            zip_path.unlink()
        archive_path.replace(zip_path)
    return zip_path


def build_release(
    *,
    workspace_root: Path = WORKSPACE_ROOT,
    output_root: Path | None = None,
    package_name: str = DEFAULT_PACKAGE_NAME,
    version: str = "",
    release_notes_path: str | None = None,
    include_runtime_bundle: bool = False,
    include_evidence: bool = False,
) -> ReleasePackageResult:
    workspace_root = workspace_root.resolve()
    resolved_output_root = (output_root or DEFAULT_OUTPUT_ROOT).resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    build_timestamp = now.strftime("%Y%m%d-%H%M%S")
    generated_at = now.strftime("%Y-%m-%d %H:%M:%S")
    resolved_version = version.strip() if version and version.strip() else "snapshot"
    package_label = get_safe_path_segment(package_name)
    version_label = get_safe_path_segment(resolved_version)

    if not package_label:
        raise ReleasePackagingError("PackageName 不能为空，也不能只包含非法文件名字符。")
    if not version_label:
        raise ReleasePackagingError("Version 不能为空，也不能只包含非法文件名字符。")

    release_dir = resolved_output_root / f"{package_label}-{version_label}-{build_timestamp}"
    zip_path = release_dir.with_suffix(".zip")
    release_dir.mkdir(parents=True, exist_ok=True)

    included_items: list[str] = []
    for item in BASE_ITEMS:
        copy_required_workspace_item(relative_path=item, workspace_root=workspace_root, release_dir=release_dir)
        included_items.append(item)

    release_notes_source = resolve_release_notes_source(
        workspace_root=workspace_root,
        release_notes_path=release_notes_path,
    )
    render_release_notes(
        source_path=release_notes_source,
        destination_path=release_dir / "RELEASE_NOTES.md",
        package_name=package_name,
        version=resolved_version,
        build_timestamp=build_timestamp,
        generated_at=generated_at,
        release_dir=release_dir,
        zip_path=zip_path,
    )
    included_items.append("RELEASE_NOTES.md")

    if include_runtime_bundle:
        copy_or_archive_runtime_bundle(
            archive_relative_path=RUNTIME_BUNDLE_RELATIVE_PATH,
            runtime_dir_relative_path=RUNTIME_DIR_RELATIVE_PATH,
            workspace_root=workspace_root,
            release_dir=release_dir,
        )
        included_items.append(RUNTIME_BUNDLE_RELATIVE_PATH)

    if include_evidence:
        copied_evidence: list[str] = []
        for relative_path in RUNTIME_EVIDENCE_RELATIVE_PATHS:
            if copy_workspace_item(relative_path=relative_path, workspace_root=workspace_root, release_dir=release_dir):
                copied_evidence.append(relative_path)
        if not copied_evidence:
            raise ReleasePackagingError("发布内容缺失：artifacts\\runtime\\rkvoice_runtime\\{asr,tts}\\output")
        included_items.extend(copied_evidence)

    manifest_path = release_dir / "RELEASE_MANIFEST.md"
    write_release_manifest(
        destination_path=manifest_path,
        generated_at=generated_at,
        package_name=package_name,
        package_label=package_label,
        version=resolved_version,
        build_timestamp=build_timestamp,
        release_notes_source=release_notes_source,
        release_dir=release_dir,
        zip_path=zip_path,
        include_runtime_bundle=include_runtime_bundle,
        include_evidence=include_evidence,
        included_items=included_items,
    )
    create_zip_archive(release_dir=release_dir, zip_path=zip_path)

    return ReleasePackageResult(
        package_name=package_name,
        package_label=package_label,
        version=resolved_version,
        build_timestamp=build_timestamp,
        generated_at=generated_at,
        release_notes_source=release_notes_source,
        release_dir=release_dir,
        zip_path=zip_path,
        manifest_path=manifest_path,
        included_items=tuple(included_items),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package a release bundle for RKVoice-TTS")
    parser.add_argument("--output-root", "--OutputRoot", default="", help="Release output root directory")
    parser.add_argument("--package-name", "--PackageName", default=DEFAULT_PACKAGE_NAME, help="Release package prefix")
    parser.add_argument("--version", "--Version", default="", help="Release version label")
    parser.add_argument(
        "--release-notes-path",
        "--ReleaseNotesPath",
        default="",
        help="Custom release notes template or rendered notes source",
    )
    parser.add_argument(
        "--include-runtime-bundle",
        "--IncludeRuntimeBundle",
        action="store_true",
        help="Include artifacts/runtime/rkvoice_runtime.tar.gz",
    )
    parser.add_argument(
        "--include-evidence",
        "--IncludeEvidence",
        action="store_true",
        help="Include artifacts/runtime/rkvoice_runtime/asr/output and artifacts/runtime/rkvoice_runtime/tts/output when present",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_release(
            output_root=Path(args.output_root).resolve() if args.output_root else None,
            package_name=args.package_name,
            version=args.version,
            release_notes_path=args.release_notes_path or None,
            include_runtime_bundle=args.include_runtime_bundle,
            include_evidence=args.include_evidence,
        )
    except ReleasePackagingError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Release version:   {result.version}")
    print(f"Release directory: {result.release_dir}")
    print(f"Release archive:   {result.zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())