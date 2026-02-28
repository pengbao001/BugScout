from __future__ import annotations

from pathlib import Path

def is_binary_file(path : Path, sample_size : int = 4096) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(sample_size)
        return b"\x00" in chunk
    except OSError:
        return True

def read_text_truncated(path : Path, max_chars : int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) > max_chars:
        return text[:max_chars]
    return text