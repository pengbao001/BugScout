from __future__ import annotations

from pathlib import Path
from bugscout.data.file_utils import read_text_truncated

def build_file_input(repo_root: Path, rel_path: str, *, max_chars: int = 8000, include_path: bool = True) -> str:
    content = read_text_truncated(repo_root / rel_path, max_chars=max_chars)
    if include_path:
        return f"path: {rel_path}\n\n{content}"
    return content