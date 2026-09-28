"""Text embedders and a batched, GPU-aware ingestion pipeline.

* :class:`HashingEmbedder`: deterministic signed feature hashing of word
  uni/bi-grams and character tri-grams with sublinear TF. No model, no
  network, stable across processes. A lexical baseline, not a semantic model.
* :class:`SentenceTransformerEmbedder`: any sentence-transformers model on the
  best available device (CUDA -> MPS -> CPU), fp16 on CUDA, batched.
* :func:`embed_stream`: bounded-memory batched embedding for large corpora.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Iterable, Iterator, Protocol, Sequence

import numpy as np

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024, char_ngrams: int = 3, bigrams: bool = True):
        self.dim, self.char_ngrams, self.bigrams = dim, char_ngrams, bigrams
        self.name = f"hashing-{dim}"

    def _h(self, feat: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def _features(self, text: str) -> dict[str, float]:
        toks = [t.lower() for t in _TOKEN.findall(text)]
        feats: dict[str, float] = {}
        for t in toks:
            feats["w:" + t] = feats.get("w:" + t, 0) + 1
            if self.char_ngrams and len(t) > self.char_ngrams:
                padded = f"#{t}#"
                for i in range(len(padded) - self.char_ngrams + 1):
                    g = "c:" + padded[i:i + self.char_ngrams]
                    feats[g] = feats.get(g, 0) + 0.5
        if self.bigrams:
            for a, b in zip(toks, toks[1:]):
                feats[f"b:{a}_{b}"] = feats.get(f"b:{a}_{b}", 0) + 1
        return feats

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for r, text in enumerate(texts):
            for f, c in self._features(text).items():
                i, s = self._h(f)
                out[r, i] += s * (1.0 + np.log(c))
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-12)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None,
                 batch_size: int = 64, fp16: bool | None = None):
        from sentence_transformers import SentenceTransformer

        from ..accel import _torch_device
        self.device = device or _torch_device() or "cpu"
        self.model = SentenceTransformer(model, device=self.device)
        if fp16 if fp16 is not None else self.device == "cuda":
            self.model.half()
        self.batch_size = batch_size
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.name = model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        v = self.model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
        return v.astype(np.float32)


def embed_stream(texts: Iterable[str], embedder: Embedder, batch_size: int = 256,
                 stats: dict | None = None) -> Iterator[tuple[list[str], np.ndarray]]:
    """Yield ``(batch_texts, batch_vectors)``; memory bounded by one batch."""
    batch: list[str] = []
    t0 = time.time()
    n = 0
    for t in texts:
        batch.append(t)
        if len(batch) == batch_size:
            yield batch, embedder.encode(batch)
            n += len(batch)
            batch = []
    if batch:
        yield batch, embedder.encode(batch)
        n += len(batch)
    if stats is not None:
        dt = time.time() - t0
        stats.update(texts=n, seconds=dt, texts_per_second=n / dt if dt > 0 else float("inf"))


def get_embedder(spec: str = "hashing") -> Embedder:
    """``hashing[:dim]`` or ``st:<model name>`` (sentence-transformers)."""
    kind, _, arg = spec.partition(":")
    if kind == "hashing":
        return HashingEmbedder(int(arg) if arg else 1024)
    if kind in ("st", "sentence-transformers"):
        return SentenceTransformerEmbedder(arg or "sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"unknown embedder '{spec}'")
