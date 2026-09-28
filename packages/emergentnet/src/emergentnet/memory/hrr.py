"""Holographic Reduced Representations (Plate, IEEE TNN 6(3), 1995).

* ``bind(a, b)``    circular convolution, computed as ``irfft(rfft(a) * rfft(b))``, O(d log d)
* ``unbind(c, a)``  circular correlation with ``a`` (``c`` bound with the involution of ``a``);
                    exact inverse when ``a`` is unitary (|rfft(a)| = 1)
* ``bundle(...)``   superposition (sum), then normalise

Random unitary vectors are quasi-orthogonal: their cosine similarity has
std ~ ``1/sqrt(d)``. A bundle of ``k`` bound pairs, unbound with one key,
returns the matching value plus crosstalk noise of std ~ ``sqrt(k/d)``
relative to the signal. Cleanup against a codebook of ``M`` items succeeds
reliably while ``d >~ 8 k ln(M)`` (see :func:`capacity` for measured rates).
"""

from __future__ import annotations

import hashlib

import numpy as np


def unitary(d: int, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Random real vector with unit-magnitude spectrum (norm 1)."""
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    nf = d // 2 + 1
    ph = rng.uniform(-np.pi, np.pi, nf)
    ph[0] = 0.0 if rng.random() < 0.5 else np.pi
    if d % 2 == 0:
        ph[-1] = 0.0 if rng.random() < 0.5 else np.pi
    v = np.fft.irfft(np.exp(1j * ph), n=d)
    return (v / np.linalg.norm(v)).astype(np.float32)


def symbol(name: str, d: int, namespace: str = "") -> np.ndarray:
    """Deterministic unitary vector for a symbol: identical in every process."""
    seed = int.from_bytes(hashlib.blake2b(f"{namespace}\x1f{name}".encode(), digest_size=8).digest(), "little")
    return unitary(d, seed)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a.shape[-1]
    return np.fft.irfft(np.fft.rfft(a, axis=-1) * np.fft.rfft(b, axis=-1), n=d, axis=-1).astype(np.float32)


def unbind(c: np.ndarray, a: np.ndarray) -> np.ndarray:
    d = c.shape[-1]
    return np.fft.irfft(np.fft.rfft(c, axis=-1) * np.conj(np.fft.rfft(a, axis=-1)), n=d, axis=-1).astype(np.float32)


def bundle(*vs: np.ndarray, normalize: bool = True) -> np.ndarray:
    s = np.sum(vs, axis=0)
    if normalize:
        s = s / max(float(np.linalg.norm(s)), 1e-12)
    return s.astype(np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.maximum(n, 1e-12)).astype(np.float32)


def capacity(d: int, codebook: int, safety: float = 8.0) -> int:
    """Number of bound pairs one trace can hold: ``k = d / (safety * ln M)``.

    Measured recall with a 64-item codebook, d in {512, 1024, 2048}:
    safety 8 -> 100 %, safety 4 -> 94-98 %, safety 2 -> 69-75 %.
    """
    return max(1, int(d / (safety * np.log(max(codebook, 2)))))


class Cleanup:
    """Nearest-neighbour cleanup memory over a named codebook."""

    def __init__(self, d: int):
        self.d = d
        self.names: list[str] = []
        self._index: dict[str, int] = {}
        self._M = np.zeros((0, d), np.float32)

    def add(self, name: str, vec: np.ndarray | None = None, namespace: str = "") -> np.ndarray:
        if name in self._index:
            return self._M[self._index[name]]
        v = symbol(name, self.d, namespace) if vec is None else normalize(np.asarray(vec, np.float32))
        self._index[name] = len(self.names)
        self.names.append(name)
        self._M = np.vstack([self._M, v[None, :]])
        return v

    def __len__(self) -> int:
        return len(self.names)

    def query(self, noisy: np.ndarray, k: int = 1) -> list[tuple[str, float]]:
        if not self.names:
            return []
        s = self._M @ normalize(noisy)
        idx = np.argsort(-s)[:k]
        return [(self.names[i], float(s[i])) for i in idx]


class AssociativeTrace:
    """Fixed-size key -> value memory: ``T = sum_i bind(key_i, value_i)``.

    Storage is O(d) regardless of the number of pairs; recall degrades
    gracefully as the load approaches :func:`capacity`.
    """

    def __init__(self, d: int = 1024, namespace: str = "trace"):
        self.d = d
        self.ns = namespace
        self.trace = np.zeros(d, np.float32)
        self.values = Cleanup(d)
        self.count = 0

    def store(self, key: str, value: str, weight: float = 1.0) -> None:
        k = symbol(key, self.d, self.ns + ":key")
        v = self.values.add(value, namespace=self.ns + ":val")
        self.trace += weight * bind(k, v)
        self.count += 1

    def recall(self, key: str, k: int = 1) -> list[tuple[str, float]]:
        return self.values.query(unbind(self.trace, symbol(key, self.d, self.ns + ":key")), k)

    def decay(self, factor: float) -> None:
        """Exponential forgetting (e.g. per conversation turn)."""
        self.trace *= factor

    @property
    def load(self) -> float:
        return self.count / capacity(self.d, max(len(self.values), 2))
