from __future__ import annotations

from dataclasses import dataclass

@dataclass
class BugExample:
    example_id : str
    repo_id : str
    base_sha : str
    issue_text : str
    changed_files : tuple[str, ...]
