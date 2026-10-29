"""Datasheet store: PDF text extraction and page-cited BM25 search."""

import io
import math
import re
from collections import Counter
from dataclasses import dataclass

from pypdf import PdfReader

CHUNK_CHARS = 1500
CHUNK_OVERLAP = 250
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Words, keeping part numbers and decimals whole. Hyphenated words are indexed both whole
    and as their parts, so "unity gain" matches a datasheet's "Unity-Gain"."""
    tokens = _TOKEN.findall(text.lower())
    return tokens + [part for token in tokens if "-" in token for part in token.split("-") if part]


class BM25:
    """Okapi BM25 with Lucene's IDF, log(1 + (N - n + 0.5) / (n + 0.5)).

    That IDF stays positive, so a term present in half of a small corpus still counts
    (the classic form gives it zero weight, which breaks one- or two-chunk libraries).
    """

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(doc) for doc in documents]
        self.lengths = [len(doc) for doc in documents]
        self.avg_length = (sum(self.lengths) / len(documents)) if documents else 0.0
        doc_freq = Counter(term for tf in self.term_freqs for term in tf)
        n = len(documents)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def scores(self, query: list[str]) -> list[float]:
        results = []
        for tf, length in zip(self.term_freqs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1))
            results.append(
                sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in set(query) if t in tf)
            )
        return results


@dataclass(frozen=True)
class Chunk:
    doc: str
    page: int  # 1-based, as printed in a PDF viewer
    text: str


def extract_pages(data: bytes) -> list[str]:
    """Text of each page of a PDF ("" for pages with no text layer)."""
    return [page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages]


class Library:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.page_counts: dict[str, int] = {}
        self._index: BM25 | None = None

    def docs(self) -> list[str]:
        return list(self.page_counts)

    def add_pdf(self, name: str, data: bytes) -> int:
        """Index a PDF. Returns the number of pages that had extractable text."""
        return self.add_pages(name, extract_pages(data))

    def add_pages(self, name: str, pages: list[str]) -> int:
        self.remove(name)
        self.page_counts[name] = len(pages)
        with_text = 0
        for number, text in enumerate(pages, start=1):
            text = re.sub(r"[ \t]+", " ", text).strip()
            if not text:
                continue
            with_text += 1
            start = 0
            while True:
                self.chunks.append(Chunk(name, number, text[start : start + CHUNK_CHARS]))
                if start + CHUNK_CHARS >= len(text):
                    break
                start += CHUNK_CHARS - CHUNK_OVERLAP
        self._index = None
        return with_text

    def remove(self, name: str) -> None:
        if name in self.page_counts:
            del self.page_counts[name]
            self.chunks = [c for c in self.chunks if c.doc != name]
            self._index = None

    def first_pages(self, name: str, count: int = 3) -> list[tuple[int, str]]:
        """Text of the first few pages of a document, for spec-card extraction."""
        pages: dict[int, str] = {}
        for chunk in self.chunks:
            if chunk.doc == name and chunk.page <= count:
                pages.setdefault(chunk.page, chunk.text)
        return sorted(pages.items())

    def search(self, query: str, k: int = 5, doc: str | None = None) -> list[Chunk]:
        if not self.chunks:
            return []
        if self._index is None:
            self._index = BM25([tokenize(c.text) for c in self.chunks])
        scores = self._index.scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        hits = [
            self.chunks[i]
            for i in ranked
            if scores[i] > 0 and (doc is None or self.chunks[i].doc == doc)
        ]
        return hits[:k]
