from __future__ import annotations

from dataclasses import dataclass
from datasets import load_dataset
import ast
import json
from typing import Iterable, Optional, Tuple
from bugscout.data.examples import BugExample
from collections import defaultdict
import random

def safe_str(x : object) -> str:
    return "" if x is None else str(x)

def parse_changed_files(value : object) -> list[str]:
    if value is None:
        return []
    
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass

        if "," in s:
            parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
            return [p for p in parts if p]
    
    return []

def build_issue_text(title : object, body : object) -> str:
    
    title_s = safe_str(title).strip()
    body_s = safe_str(body).strip()

    if title_s and body_s:
        return f"{title_s}\n\n{body_s}"
    
    return title_s or body_s

def load_lca_examples(*, configuration : str = "py", split : str = "dev", limit : Optional[int] = 500) -> list[BugExample]:
    ds = load_dataset("JetBrains-Research/lca-bug-localization", configuration, split=split)
    examples = []

    n = len(ds) if limit is None else min(limit, len(ds))

    for i in range(n):
        row = ds[i]

        repo_owner = row.get("repo_owner")
        repo_name = row.get("repo_name")
        base_sha = row.get("base_sha")

        issue_title = row.get("issue_title")
        issue_body = row.get("issue_body")
        changed_files_raw = row.get("changed_files")

        repo_id = f"{repo_owner}/{repo_name}"
        issue_text = build_issue_text(issue_title, issue_body)

        changed_files = tuple(parse_changed_files(changed_files_raw))

        example_id = str(row.get("text_id") or row.get("id") or i)

        examples.append(
            BugExample(
                example_id=example_id,
                repo_id=repo_id,
                base_sha=safe_str(base_sha),
                issue_text=issue_text,
                changed_files=changed_files,
            )
        )
    
    return examples
