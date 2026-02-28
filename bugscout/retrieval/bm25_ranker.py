from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rank_bm25 import BM25Okapi

from bugscout.data.file_utils import read_text_truncated
from bugscout.retrieval.tokenize import normalize_rel_path, tokenize

Mode = Literal["path", "content", "path+content"]

@dataclass
class BM25RepoIndex:
    repo_root : Path
    file_paths : list[str]
    mode : Mode = "path+content"
    max_chars : int = 8888

    def __post_init__(self) -> None:
        self.repo_root = self.repo_root.resolve()
        self.file_path = [normalize_rel_path(p) for p in self.file_paths]

        corpus_tokens = []
        for rel in self.file_paths:
            tokens = []

            if self.mode in ("path", "path+content"):
                tokens.extend(tokenize(rel.replace("/", " ").replace(".", " ")))
            
            if self.mode in ("content", "path+content"):
                abs_path = self.repo_root / rel
                text = read_text_truncated(abs_path, max_chars=self.max_chars)
                tokens.extend(tokenize(text))
            
            if not tokens:
                tokens = ["__empty__"]
            
            corpus_tokens.append(tokens)
        
        self._bm25 = BM25Okapi(corpus_tokens)
    
    def rank(self, issue_text : str, topk : int = 66) -> list[str]:
        
        q_tokens = tokenize(issue_text)
        if not q_tokens:
            return self.file_path[:topk]
        
        scores = self._bm25.get_scores(q_tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i : scores[i], reverse=True)

        ranked_paths = [self.file_paths[i] for i in ranked_idx[:topk]]
        return ranked_paths
