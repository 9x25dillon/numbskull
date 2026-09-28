"""Problem representations.

Ising (spins ``s in {-1,+1}^n``)::

    E(s) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j + offset = h.s + 1/2 s^T J s + offset

with ``J`` stored full, symmetric, zero diagonal.

QUBO (bits ``x in {0,1}^n``)::

    E(x) = x^T Q x + offset

Both are exactly inter-convertible via ``x = (1 + s) / 2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Ising:
    h: np.ndarray
    J: np.ndarray
    offset: float = 0.0

    def __post_init__(self):
        h = np.asarray(self.h, dtype=np.float64)
        J = np.asarray(self.J, dtype=np.float64)
        n = len(h)
        if J.shape != (n, n):
            raise ValueError("J must be (n, n)")
        J = (J + J.T) / 2 if not np.allclose(J, J.T) else J.copy()
        np.fill_diagonal(J, 0.0)
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "J", J)

    @property
    def n(self) -> int:
        return len(self.h)

    def energy(self, s: np.ndarray) -> np.ndarray:
        """Energy of one state (n,) or a batch (R, n)."""
        s = np.asarray(s, dtype=np.float64)
        return s @ self.h + 0.5 * np.einsum("...i,ij,...j->...", s, self.J, s) + self.offset

    def local_fields(self, s: np.ndarray) -> np.ndarray:
        return self.h + np.asarray(s, dtype=np.float64) @ self.J

    def scale(self) -> float:
        m = max(np.abs(self.h).max(initial=0.0), np.abs(self.J).max(initial=0.0))
        return m if m > 0 else 1.0

    def to_qubo(self) -> "QUBO":
        # s = 2x - 1
        n = self.n
        Q = 2.0 * self.J  # off-diagonal: 1/2 * 4 J_ij x_i x_j summed both ways
        lin = 2.0 * self.h - 2.0 * self.J.sum(axis=1)
        Q[np.diag_indices(n)] = lin
        off = self.offset - self.h.sum() + 0.5 * self.J.sum()
        return QUBO(Q, off)


@dataclass(frozen=True)
class QUBO:
    Q: np.ndarray
    offset: float = 0.0

    def __post_init__(self):
        Q = np.asarray(self.Q, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be square")
        object.__setattr__(self, "Q", (Q + Q.T) / 2)

    @property
    def n(self) -> int:
        return len(self.Q)

    def energy(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.einsum("...i,ij,...j->...", x, self.Q, x) + self.offset

    def to_ising(self) -> Ising:
        Q = self.Q
        h = 0.5 * Q.sum(axis=1)
        J = 0.5 * Q.copy()
        np.fill_diagonal(J, 0.0)
        off = self.offset + 0.25 * Q.sum() + 0.25 * np.trace(Q)
        return Ising(h, J, off)

    @classmethod
    def from_dict(cls, terms: dict, n: int | None = None, offset: float = 0.0) -> "QUBO":
        """``{(i, j): w}`` (``i == j`` for linear terms), as used by dimod/D-Wave."""
        n = n or 1 + max(max(k) for k in terms)
        Q = np.zeros((n, n))
        for (i, j), w in terms.items():
            if i == j:
                Q[i, i] += w
            else:
                Q[i, j] += w / 2
                Q[j, i] += w / 2
        return cls(Q, offset)


@dataclass(frozen=True)
class Continuous:
    """Box-constrained continuous minimisation ``min f(x), lo <= x <= hi``."""

    f: Callable[[np.ndarray], float]
    lower: np.ndarray
    upper: np.ndarray
    vectorized: bool = False   # f accepts (N, d) and returns (N,)

    def __post_init__(self):
        lo = np.asarray(self.lower, dtype=np.float64)
        hi = np.asarray(self.upper, dtype=np.float64)
        if lo.shape != hi.shape or np.any(hi <= lo):
            raise ValueError("bounds must have equal shape and upper > lower")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)

    @property
    def dim(self) -> int:
        return len(self.lower)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if self.vectorized:
            return np.asarray(self.f(X), dtype=np.float64)
        return np.array([self.f(x) for x in X], dtype=np.float64)


# -- problem builders ------------------------------------------------------------

def maxcut(W: np.ndarray) -> Ising:
    """Minimising returns ``-cut``: ``cut(s) = sum_{i<j} W_ij (1 - s_i s_j) / 2``."""
    W = np.asarray(W, dtype=np.float64)
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    return Ising(np.zeros(len(W)), W / 2, -np.triu(W, 1).sum() / 2)


def maxcut_from_edges(edges: Sequence[tuple], n: int | None = None) -> Ising:
    n = n or 1 + max(max(e[0], e[1]) for e in edges)
    W = np.zeros((n, n))
    for e in edges:
        w = e[2] if len(e) > 2 else 1.0
        W[e[0], e[1]] += w
        W[e[1], e[0]] += w
    return maxcut(W)


def number_partitioning(a: Sequence[float]) -> Ising:
    """``E(s) = (sum_i a_i s_i)^2``; zero iff a perfect partition exists."""
    a = np.asarray(a, dtype=np.float64)
    return Ising(np.zeros(len(a)), 2 * np.outer(a, a), float(np.sum(a ** 2)))


def sherrington_kirkpatrick(n: int, seed: int | None = None) -> Ising:
    rng = np.random.default_rng(seed)
    J = np.triu(rng.choice([-1.0, 1.0], size=(n, n)), 1) / np.sqrt(n)
    return Ising(np.zeros(n), J + J.T)


def random_regular_graph(n: int, d: int = 3, seed: int | None = None) -> np.ndarray:
    """Adjacency matrix of a random d-regular graph (configuration model with retries)."""
    rng = np.random.default_rng(seed)
    if (n * d) % 2:
        raise ValueError("n*d must be even")
    for _ in range(1000):
        stubs = rng.permutation(np.repeat(np.arange(n), d))
        pairs = stubs.reshape(-1, 2)
        if np.any(pairs[:, 0] == pairs[:, 1]):
            continue
        A = np.zeros((n, n))
        ok = True
        for i, j in pairs:
            if A[i, j]:
                ok = False
                break
            A[i, j] = A[j, i] = 1
        if ok:
            return A
    raise RuntimeError("failed to sample a simple regular graph")
