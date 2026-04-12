from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path


def log(message: str) -> None:
    print(message, flush=True)


def fail(message: str, exit_code: int = 1) -> None:
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(exit_code)


def md5sum(file_path: Path) -> str:
    digest = hashlib.md5()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8", newline="\n")


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def download_http_file(url: str, destination: Path) -> None:
    ensure_parent(destination)
    tmp_destination = destination.with_suffix(destination.suffix + ".part")
    log(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, tmp_destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    tmp_destination.replace(destination)


def extract_tar_members(archive_path: Path, prefix: str, destination: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.name.startswith(prefix):
                continue
            relative_name = member.name[len(prefix):].lstrip("/")
            if not relative_name:
                continue
            target_path = destination / relative_name
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym() or member.islnk():
                continue
            file_object = archive.extractfile(member)
            if file_object is None:
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("wb") as output:
                shutil.copyfileobj(file_object, output)


def extract_tarball(
    archive_path: Path,
    destination: Path,
    *,
    strip_top_level: bool = False,
    extracted_dir_name: str | None = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if not strip_top_level:
            archive.extractall(destination)
            return

        top_level_names: list[str] = []
        for member in members:
            first_part = Path(member.name).parts[0] if Path(member.name).parts else ""
            if first_part and first_part not in top_level_names:
                top_level_names.append(first_part)
        if len(top_level_names) != 1:
            fail(f"Archive {archive_path.name} does not have a single top-level directory")

        top_level = top_level_names[0]
        root_destination = destination / (extracted_dir_name or top_level)
        root_destination.mkdir(parents=True, exist_ok=True)

        prefix = f"{top_level}/"
        for member in members:
            if member.name == top_level:
                continue
            if not member.name.startswith(prefix):
                continue
            relative_name = member.name[len(prefix):]
            if not relative_name:
                continue
            target_path = root_destination / relative_name
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym() or member.islnk():
                continue
            file_object = archive.extractfile(member)
            if file_object is None:
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("wb") as output:
                shutil.copyfileobj(file_object, output)
