from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+[0-9]+")

def tokenize(text : str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())

def normalize_rel_path(p : str)-> str:
    if not p:
        return ""
    
    p = p.replace("\\", "/")
    p = p.lstrip("./")

    while "//" in p:
        p = p.replace("//", "/")
    return p