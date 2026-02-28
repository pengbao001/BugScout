from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from bugscout.data.file_utils import is_binary_file

# Need to extend these.
EXCLUDED_DIR_NAMES = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "target",
    ".idea",
    ".vscode",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
}

TEST_DIR_NAMES = {
    "test",
    "tests",
    "testing",
    "testdata",
}

ALLOWED_EXTS = {
    ".py", ".pyi",
    ".java", ".kt", ".kts",
    ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs",
    ".c", ".h", ".cpp", ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".scala",
    ".sql",
    ".md", ".rst", ".txt",
    ".yml", ".yaml",
    ".json", ".toml", ".ini", ".cfg",
    ".xml", ".properties",
    ".sh", ".bash",
    ".gradle",
}

EXCLUDED_FILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
}

# Skip files larger than 300KB for now
MAX_FILE_BYTES = 300_000  


@dataclass
class FileRecord:
    rel_path: str
    size_bytes: int


def _should_exclude_dir(dir_name: str) -> bool:
    name = dir_name.lower()
    if name in EXCLUDED_DIR_NAMES:
        return True
    if name in TEST_DIR_NAMES:
        return True
    return False


def _should_include_file(path: Path) -> bool:
    """
    Decide if a file should be in the candidate set.
    """
    name = path.name
    if name in EXCLUDED_FILE_NAMES:
        return False

    if name.endswith(".min.js") or name.endswith(".min.css"):
        return False

    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= 0 or size > MAX_FILE_BYTES:
        return False

    ext = path.suffix.lower()
    if ext:
        if ext not in ALLOWED_EXTS:
            return False
    else:
        if name not in {"Makefile", "Dockerfile"}:
            return False

    if is_binary_file(path):
        return False

    return True


def collect_candidate_files(repo_root: Path) -> list[FileRecord]:
    """
    Walk the repo and collect candidate files. 
    Returns a list of FileRecord with relative paths.
    """
    repo_root = repo_root.resolve()
    out: list[FileRecord] = []

    for root, dirs, files in os.walk(repo_root):
        root_path = Path(root)

        dirs[:] = [d for d in dirs if not _should_exclude_dir(d)]

        for fn in files:
            abs_path = root_path / fn
            if not _should_include_file(abs_path):
                continue

            rel = abs_path.relative_to(repo_root).as_posix()
            size = abs_path.stat().st_size
            out.append(FileRecord(rel_path=rel, size_bytes=size))

    out.sort(key=lambda r: r.rel_path)
    return out