"""Terminated feed-forward convolutional code with soft-decision Viterbi.

Default: NASA/CCSDS K=7, rate 1/2, generators (171, 133)_8, free distance 10.
Each block of ``k`` info bits is followed by ``K-1`` zero tail bits.
"""

from __future__ import annotations

import numpy as np

from .base import BlockCodec, DecodeResult
from .trellis import feedforward_trellis


class Convolutional(BlockCodec):
    def __init__(self, k: int = 256, gens: tuple[int, ...] = (171, 133), K: int = 7):
        self._k = k
        self.K = K
        self.tr = feedforward_trellis(gens, K)
        self.r = len(gens)
        self.name = f"conv{K}_{k}"
        S = self.tr.n_states
        prev = [[] for _ in range(S)]
        for s in range(S):
            for u in (0, 1):
                prev[self.tr.next_state[s, u]].append((s, u))
        self._ps = np.array([[p[0][0], p[1][0]] for p in prev])
        self._pu = np.array([[p[0][1], p[1][1]] for p in prev])
        x = 1.0 - 2.0 * self.tr.outputs.astype(np.float64)  # (S,2,r)
        self._x0 = x[self._ps[:, 0], self._pu[:, 0]]  # (S,r)
        self._x1 = x[self._ps[:, 1], self._pu[:, 1]]

    @property
    def k(self) -> int:
        return self._k

    @property
    def n(self) -> int:
        return (self._k + self.K - 1) * self.r

    def encode_block(self, info: np.ndarray) -> np.ndarray:
        u = np.concatenate([np.asarray(info, dtype=np.uint8), np.zeros(self.K - 1, dtype=np.uint8)])
        out = np.empty((len(u), self.r), dtype=np.uint8)
        s = 0
        for t, b in enumerate(u):
            out[t] = self.tr.outputs[s, b]
            s = self.tr.next_state[s, b]
        return out.ravel()

    def decode_block(self, llr: np.ndarray) -> DecodeResult:
        T = self._k + self.K - 1
        L = np.asarray(llr, dtype=np.float64).reshape(T, self.r)
        S = self.tr.n_states
        # correlation metrics: maximise sum(L * x)
        m0 = L @ self._x0.T  # (T,S) metric of branch from first predecessor into s'
        m1 = L @ self._x1.T
        pm = np.full(S, -1e30)
        pm[0] = 0.0
        surv = np.empty((T, S), dtype=np.uint8)
        ps = self._ps
        for t in range(T):
            c0 = pm[ps[:, 0]] + m0[t]
            c1 = pm[ps[:, 1]] + m1[t]
            choose1 = c1 > c0
            surv[t] = choose1
            pm = np.where(choose1, c1, c0)
            pm -= pm.max()
        s = 0  # terminated
        bits = np.empty(T, dtype=np.uint8)
        for t in range(T - 1, -1, -1):
            j = surv[t, s]
            bits[t] = self._pu[s, j]
            s = ps[s, j]
        return DecodeResult(bits=bits[: self._k], ok=True)
