from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

def _run_git(args : list[str], cwd : Path) -> None:

    cmd = ["git"] + args
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Git command failed:\n"
            f"cwd : {cwd}\n"
            f"cmd : {''.join(cmd)}\n"
            f"stdout : {proc.stdout}\n"
            f"stderr : {proc.stderr}\n"
        )

@dataclass
class RepoManager:
    cache_dir : Path

    def __init__(self, cache_dir : str | Path = "cache") -> None:
        self.cache_dir = Path(cache_dir)
        (self.cache_dir / "repos").mkdir(parents=True, exist_ok=True)
    
    def repo_dir(self, repo_id : str) -> Path:
        # owner/name -> owner__name
        safe_folder_name = repo_id.replace("/", "__")
        return self.cache_dir / "repos" / safe_folder_name
    
    def repo_url(self, repo_id : str) -> str:
        return f"https://github.com/{repo_id}.git"
    
    def ensure_cloned(self, repo_id : str) -> Path:
        
        stored_path = self.repo_dir(repo_id)
        if stored_path.exists() and (stored_path / ".git").exists():
            return stored_path
        
        if stored_path.exists():
            shutil.rmtree(stored_path)
        
        stored_path.parent.mkdir(parents=True, exist_ok=True)

        url = self.repo_url(repo_id)
        print(f"[RepoManager] Cloning {url} -> {stored_path}")
        subprocess.run(
            ["git", "clone", url, str(stored_path)],
            check=True,
        )

        return stored_path
    
    def checkout_sha(self, repo_path : Path, sha : str) -> None:
        try:
            _run_git(["cat-file", "-e", f"{sha}^{{commit}}"], cwd=repo_path)
        except RuntimeError:
            print("[RepoManager] SHA not found locally; fetching...")
            _run_git(["fetch", "--all", "--prune"], cwd=repo_path)
        
        _run_git(["reset", "--hard"], cwd=repo_path)
        _run_git(["clean", "-fdx"], cwd=repo_path)

        print(F"[RepoManager] Checking out {sha}")
        _run_git(["checkout", "--detach", sha], cwd=repo_path)
    
    def prepare_repo(self, repo_id : str, base_sha : str) -> Path:
        repo_path = self.ensure_cloned(repo_id)
        self.checkout_sha(repo_path, base_sha)
        return repo_path
