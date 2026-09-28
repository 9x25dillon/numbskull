"""HolographicStore: a persistent vector store for retrieval-augmented generation.

Retrieval is exact inner-product search over L2-normalised embeddings, run on
the selected array backend (GPU when available) in bounded-memory chunks.

Holographic part: for configured attribute names (e.g. ``source``,
``author``, ``topic``) each record also gets a *composite* vector::

    r = normalize( content + w * sum_a bind(role_a, filler(a, value_a)) )

Because bound pairs are quasi-orthogonal to content and to each other:

* **soft attribute queries in one dot product**: querying with
  ``content_q + w * bind(role_a, filler(a, v))`` adds ~``w^2`` (before
  normalisation) to every record whose attribute ``a`` equals ``v``.
  That is a ranking boost rather than a hard filter, so it composes with any
  ANN index (FAISS, HNSW) that has no metadata filtering;
* **attribute recovery from the vector alone**:
  ``unbind(r, role_a)`` is close to ``filler(a, v)``, so cleanup against
  the known values recovers ``v``.

Hard filters (``mode="hard"``) remain available as an exact metadata mask.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..accel import Backend, get_backend
from .embed import Embedder, embed_stream, get_embedder
from .hrr import Cleanup, bind, normalize, symbol, unbind

FORMAT_VERSION = 1


@dataclass
class Hit:
    id: str
    score: float
    text: str
    metadata: dict = field(default_factory=dict)


class HolographicStore:
    def __init__(self, dim: int | None = None, embedder: Embedder | str | None = None, backend: str = "auto",
                 attributes: Sequence[str] = (), attr_weight: float = 0.5, namespace: str = "hstore"):
        self.embedder = get_embedder(embedder) if isinstance(embedder, str) else embedder
        if dim is None:
            if self.embedder is None:
                raise ValueError("need dim or an embedder")
            dim = self.embedder.dim
        if self.embedder is not None and self.embedder.dim != dim:
            raise ValueError(f"embedder dim {self.embedder.dim} != store dim {dim}")
        self.dim = dim
        self.backend: Backend = get_backend(backend)
        self.attributes = tuple(attributes)
        self.attr_weight = attr_weight
        self.ns = namespace
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._texts: list[str] = []
        self._meta: list[dict] = []
        self._content = np.zeros((0, dim), np.float32)
        self._composite = np.zeros((0, dim), np.float32)
        self._n = 0
        self._dev: dict[str, Any] = {}
        self._fillers = {a: Cleanup(dim) for a in self.attributes}
        self._next_auto = 0

    # -- symbols -------------------------------------------------------------------
    def _role(self, attr: str) -> np.ndarray:
        return symbol(attr, self.dim, self.ns + ":role")

    def _filler(self, attr: str, value: Any) -> np.ndarray:
        return symbol(str(value), self.dim, f"{self.ns}:fill:{attr}")

    def _attr_vector(self, meta: dict) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        for a in self.attributes:
            if a in meta and meta[a] is not None:
                v += bind(self._role(a), self._filler(a, meta[a]))
        return v

    # -- ingestion -----------------------------------------------------------------
    def __len__(self) -> int:
        return self._n

    def _grow(self, extra: int) -> None:
        need = self._n + extra
        cap = len(self._content)
        if need <= cap:
            return
        new = max(need, int(cap * 1.5) + 64)
        for name in ("_content", "_composite"):
            old = getattr(self, name)
            buf = np.zeros((new, self.dim), np.float32)
            buf[: self._n] = old[: self._n]
            setattr(self, name, buf)

    def add(self, texts: Sequence[str] | None = None, embeddings: np.ndarray | None = None,
            ids: Sequence[str] | None = None, metadata: Sequence[dict] | None = None) -> list[str]:
        if embeddings is None:
            if texts is None or self.embedder is None:
                raise ValueError("provide embeddings, or texts with an embedder")
            embeddings = self.embedder.encode(list(texts))
        E = normalize(np.asarray(embeddings, np.float32).reshape(-1, self.dim))
        m = len(E)
        texts = list(texts) if texts is not None else [""] * m
        metadata = [dict(x) for x in metadata] if metadata is not None else [{} for _ in range(m)]
        if ids is None:
            ids = []
            for _ in range(m):
                while str(self._next_auto) in self._pos:
                    self._next_auto += 1
                ids.append(str(self._next_auto))
                self._next_auto += 1
        ids = [str(i) for i in ids]
        if not (len(texts) == len(metadata) == len(ids) == m):
            raise ValueError("texts, embeddings, ids and metadata must have equal length")
        if len(set(ids)) != m or any(i in self._pos for i in ids):
            raise ValueError("duplicate ids (use upsert to replace)")
        self._grow(m)
        s = self._n
        self._content[s:s + m] = E
        if self.attributes:
            A = np.stack([self._attr_vector(md) for md in metadata])
            self._composite[s:s + m] = normalize(E + self.attr_weight * A)
            for md in metadata:
                for a in self.attributes:
                    if a in md and md[a] is not None:
                        self._fillers[a].add(str(md[a]), self._filler(a, md[a]))
        for j, i in enumerate(ids):
            self._pos[i] = s + j
        self._ids += ids
        self._texts += texts
        self._meta += metadata
        self._n += m
        self._dev.clear()
        return ids

    def upsert(self, texts, embeddings=None, ids=None, metadata=None) -> list[str]:
        if ids is not None:
            self.delete([i for i in ids if str(i) in self._pos])
        return self.add(texts, embeddings, ids, metadata)

    def ingest(self, texts: Iterable[str], batch_size: int = 256, metadata: Iterable[dict] | None = None) -> dict:
        """Stream a large corpus through the embedder in batches."""
        if self.embedder is None:
            raise ValueError("ingest needs an embedder")
        stats: dict = {}
        meta_it = iter(metadata) if metadata is not None else None
        for batch, vecs in embed_stream(texts, self.embedder, batch_size, stats):
            md = [next(meta_it) for _ in batch] if meta_it is not None else None
            self.add(batch, vecs, metadata=md)
        return stats

    def delete(self, ids: Iterable[str]) -> int:
        rm = sorted({self._pos[str(i)] for i in ids if str(i) in self._pos})
        if not rm:
            return 0
        keep = np.setdiff1d(np.arange(self._n), rm)
        self._content = self._content[keep].copy()
        self._composite = self._composite[keep].copy() if self.attributes else np.zeros((len(keep), self.dim), np.float32)
        self._ids = [self._ids[i] for i in keep]
        self._texts = [self._texts[i] for i in keep]
        self._meta = [self._meta[i] for i in keep]
        self._n = len(keep)
        self._pos = {i: p for p, i in enumerate(self._ids)}
        self._dev.clear()
        return len(rm)

    def get(self, id: str) -> Hit:
        p = self._pos[str(id)]
        return Hit(self._ids[p], 1.0, self._texts[p], self._meta[p])

    # -- retrieval -------------------------------------------------------------------
    def _device(self, which: str):
        if which not in self._dev:
            src = self._content if which == "content" else self._composite
            self._dev[which] = self.backend.asarray(src[: self._n])
        return self._dev[which]

    def _query_vectors(self, query) -> np.ndarray:
        if isinstance(query, str):
            if self.embedder is None:
                raise ValueError("text query needs an embedder")
            return normalize(self.embedder.encode([query]))
        if isinstance(query, (list, tuple)) and query and isinstance(query[0], str):
            return normalize(self.embedder.encode(list(query)))
        return normalize(np.asarray(query, np.float32).reshape(-1, self.dim))

    def search(self, query, k: int = 5, where: dict | None = None, mode: str = "soft",
               attr_weight: float | None = None) -> list[Hit] | list[list[Hit]]:
        """``query``: text, list of texts, or vector(s). Returns hits (per query
        when a batch is given). ``where`` filters/boosts on metadata."""
        if self._n == 0:
            return []
        if isinstance(query, str):
            batch = False
        elif isinstance(query, (list, tuple)) and query and isinstance(query[0], str):
            batch = True
        else:
            batch = np.asarray(query).ndim == 2
        Q = self._query_vectors(query)
        if where and mode == "hard":
            rows = np.array([p for p in range(self._n)
                             if all(self._meta[p].get(a) == v for a, v in where.items())], dtype=np.int64)
            if len(rows) == 0:
                return [[] for _ in Q] if batch else []
            sub = self.backend.asarray(self._content[rows])
            sc, ix = self.backend.topk(sub, Q, k)
            ix = rows[ix]
        elif where and mode == "soft":
            unknown = [a for a in where if a not in self.attributes]
            if unknown:
                raise ValueError(f"soft queries need configured attributes; not configured: {unknown}")
            w = self.attr_weight if attr_weight is None else attr_weight
            A = np.zeros(self.dim, np.float32)
            for a, v in where.items():
                A += bind(self._role(a), self._filler(a, v))
            Q = normalize(Q + w * A[None, :])
            sc, ix = self.backend.topk(self._device("composite"), Q, k)
        else:
            sc, ix = self.backend.topk(self._device("content"), Q, k)
        out = [[Hit(self._ids[i], float(s), self._texts[i], self._meta[i]) for s, i in zip(srow, irow)]
               for srow, irow in zip(sc, ix)]
        return out if batch else out[0]

    def recover_attribute(self, id: str, attr: str, k: int = 1) -> list[tuple[str, float]]:
        """Decode an attribute value from the record's composite vector alone."""
        if attr not in self.attributes:
            raise ValueError(f"'{attr}' is not a configured attribute")
        r = self._composite[self._pos[str(id)]]
        return self._fillers[attr].query(unbind(r, self._role(attr)), k)

    @staticmethod
    def format_context(hits: Sequence[Hit], max_chars: int = 4000) -> str:
        """Render hits as a citation-tagged context block for an LLM prompt."""
        parts, used = [], 0
        for h in hits:
            block = f"[{h.id}] {h.text.strip()}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block) + 2
        return "\n\n".join(parts)

    # -- persistence -------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Atomic save: write to a temp dir, then swap into place."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".hstore-", dir=path.parent))
        try:
            np.save(tmp / "content.npy", self._content[: self._n])
            if self.attributes:
                np.save(tmp / "composite.npy", self._composite[: self._n])
            with open(tmp / "records.jsonl", "w", encoding="utf-8") as f:
                for i, t, m in zip(self._ids, self._texts, self._meta):
                    f.write(json.dumps({"id": i, "text": t, "metadata": m}, ensure_ascii=False) + "\n")
            manifest = {"format": FORMAT_VERSION, "dim": self.dim, "count": self._n,
                        "attributes": list(self.attributes), "attr_weight": self.attr_weight,
                        "namespace": self.ns, "embedder": getattr(self.embedder, "name", None)}
            (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
            backup = None
            if path.exists():
                backup = path.with_name(path.name + ".old")
                if backup.exists():
                    shutil.rmtree(backup)
                os.replace(path, backup)
            os.replace(tmp, path)
            if backup is not None:
                shutil.rmtree(backup)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | str | None = None, backend: str = "auto") -> "HolographicStore":
        path = Path(path)
        man = json.loads((path / "manifest.json").read_text())
        if man["format"] != FORMAT_VERSION:
            raise ValueError(f"unsupported store format {man['format']}")
        if embedder is None and man.get("embedder", "") and str(man["embedder"]).startswith("hashing-"):
            embedder = "hashing:" + man["embedder"].split("-", 1)[1]
        st = cls(man["dim"], embedder, backend, man["attributes"], man["attr_weight"], man["namespace"])
        content = np.load(path / "content.npy")
        comp = np.load(path / "composite.npy") if man["attributes"] else None
        ids, texts, metas = [], [], []
        with open(path / "records.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                ids.append(r["id"])
                texts.append(r["text"])
                metas.append(r["metadata"])
        if len(ids) != man["count"] or len(content) != man["count"]:
            raise ValueError("store is corrupt: record count mismatch")
        st._grow(len(ids))
        st._content[: len(ids)] = content
        if comp is not None:
            st._composite[: len(ids)] = comp
        st._ids, st._texts, st._meta, st._n = ids, texts, metas, len(ids)
        st._pos = {i: p for p, i in enumerate(ids)}
        for md in metas:
            for a in st.attributes:
                if a in md and md[a] is not None:
                    st._fillers[a].add(str(md[a]), st._filler(a, md[a]))
        return st
