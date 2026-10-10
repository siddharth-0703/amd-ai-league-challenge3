"""Search and extraction engine for Mini-Challenge 3.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set


STOPWORDS = {
    "what", "is", "the", "of", "for", "in", "a", "an", "at", "to", "on",
    "does", "which", "how", "when", "where", "why", "who", "by", "from",
    "with", "as", "about", "are", "was", "were", "be", "been", "being"
}


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words, symbols, and identifiers."""
    # Find words with hyphens and underscores like tq-40, orr-1847, therm_alert#
    raw_tokens = re.findall(r'[A-Za-z0-9]+(?:[\-_#\.][A-Za-z0-9]+)*', text.lower())
    clean = []
    for t in raw_tokens:
        t = t.strip('.-_')
        if len(t) > 1 and t not in STOPWORDS:
            clean.append(t)
    return clean


class SimpleBM25:
    """Lightweight, zero-dependency BM25 indexer."""
    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        self.doc_count = len(docs)
        self.avg_dl = sum(len(d["tokens"]) for d in docs) / max(1, self.doc_count)
        self.doc_freqs: Dict[str, int] = Counter()
        self.doc_term_freqs: List[Dict[str, int]] = []
        
        for d in docs:
            tf = Counter(d["tokens"])
            self.doc_term_freqs.append(tf)
            for term in tf:
                self.doc_freqs[term] += 1

    def score(self, query_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> List[Tuple[float, int]]:
        scores = []
        for i, d in enumerate(self.docs):
            score = 0.0
            dl = len(d["tokens"])
            tf_map = self.doc_term_freqs[i]
            for q in query_tokens:
                if q not in tf_map:
                    continue
                tf = tf_map[q]
                df = self.doc_freqs.get(q, 0)
                # IDF
                idf = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
                # Term score
                term_score = idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (dl / self.avg_dl)))
                score += term_score
            
            # Penalize withdrawn or superseded documents
            rel_path = d["path"].lower()
            if "withdrawn" in rel_path or "superseded" in rel_path:
                score *= 0.1
            scores.append((score, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores


def build_knowledge_base(corpus_root: Path, parse_fn) -> List[Dict[str, Any]]:
    """Walk corpus and build documents list."""
    documents = []
    for file in sorted(corpus_root.rglob("*")):
        if not file.is_file():
            continue
        try:
            rel = file.relative_to(corpus_root).as_posix()
            text = parse_fn(file)
            if not text.strip():
                continue
            
            # Break large docs into chunks if needed, or keep per-file / per-section
            tokens = tokenize(text)
            documents.append({
                "path": rel,
                "text": text,
                "tokens": tokens,
            })
        except Exception:
            continue
    return documents
