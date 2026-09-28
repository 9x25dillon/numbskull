"""Parallel-concatenated (turbo) code with log-MAP BCJR decoding.

Constituents: 8-state RSC ``(13, 15)_8`` as in 3GPP LTE, both trellises
terminated with 3 tail steps. Interleaver: deterministic S-random
permutation (spread ``S ~ sqrt(K/2)``) seeded by ``K``.

Codeword layout (rate ~1/3)::

    [ u (K) | p1 (K) | p2 (K) | u_t1 p_t1 (2*3) | u_t2 p_t2 (2*3) ]

``puncture=True`` alternates p1/p2 (rate ~1/2); punctured positions are fed
to the decoder as zero LLRs (erasures).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .base import BlockCodec, DecodeResult
from .trellis import Trellis, rsc_trellis


@lru_cache(maxsize=32)
def s_random_interleaver(K: int, seed: int | None = None, S: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(K if seed is None else seed)
    if S is None:
        S = max(1, int(np.sqrt(K / 2)))
    for _ in range(50):
        pool = list(rng.permutation(K))
        perm: list[int] = []
        stuck = 0
        while pool and stuck < len(pool):
            cand = pool.pop(0)
            if all(abs(cand - p) > S for p in perm[-S:]):
                perm.append(cand)
                stuck = 0
            else:
                pool.append(cand)
                stuck += 1
        if not pool:
            out = np.array(perm, dtype=np.int64)
            out.setflags(write=False)
            return out
        S = max(1, S - 1)
    out = rng.permutation(K)
    out.setflags(write=False)
    return out


def _encode_rsc(tr: Trellis, u: np.ndarray, nu: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = 0
    p = np.empty(len(u), dtype=np.uint8)
    for t, b in enumerate(u):
        p[t] = tr.outputs[s, b, 1]
        s = tr.next_state[s, b]
    tu = np.empty(nu, dtype=np.uint8)
    tp = np.empty(nu, dtype=np.uint8)
    for t in range(nu):
        b = tr.term_input[s]
        tu[t], tp[t] = b, tr.outputs[s, b, 1]
        s = tr.next_state[s, b]
    assert s == 0, "trellis termination failed"
    return p, tu, tp


class _BCJR:
    """Log-MAP BCJR over a rate-1/2 systematic trellis (terminated at state 0)."""

    def __init__(self, tr: Trellis):
        self.tr = tr
        S = tr.n_states
        prev = [[] for _ in range(S)]
        for s in range(S):
            for u in (0, 1):
                prev[tr.next_state[s, u]].append((s, u))
        self.p_state = np.array([[p[0][0], p[1][0]] for p in prev])
        self.p_input = np.array([[p[0][1], p[1][1]] for p in prev])
        self.xpar = 1.0 - 2.0 * tr.outputs[:, :, 1].astype(np.float64)  # (S,2)
        self.xu = np.array([1.0, -1.0])

    def llr(self, Lu: np.ndarray, Lp: np.ndarray) -> np.ndarray:
        """Posterior LLRs of the input bits. ``Lu`` includes a-priori info."""
        T = len(Lu)
        S = self.tr.n_states
        g = 0.5 * (Lu[:, None, None] * self.xu[None, None, :] + Lp[:, None, None] * self.xpar[None])  # (T,S,2)
        ps, pi = self.p_state, self.p_input
        G0 = g[:, ps[:, 0], pi[:, 0]]  # (T,S) branch into s' from its first predecessor
        G1 = g[:, ps[:, 1], pi[:, 1]]
        neg = -1e30
        alpha = np.full((T + 1, S), neg)
        alpha[0, 0] = 0.0
        for t in range(T):
            a = alpha[t]
            nxt = np.logaddexp(a[ps[:, 0]] + G0[t], a[ps[:, 1]] + G1[t])
            alpha[t + 1] = nxt - nxt.max()
        beta = np.full((T + 1, S), neg)
        beta[T, 0] = 0.0
        ns = self.tr.next_state
        for t in range(T - 1, -1, -1):
            b = beta[t + 1]
            cur = np.logaddexp(g[t, :, 0] + b[ns[:, 0]], g[t, :, 1] + b[ns[:, 1]])
            beta[t] = cur - cur.max()
        m0 = alpha[:T] + g[:, :, 0] + beta[1:][:, ns[:, 0]]
        m1 = alpha[:T] + g[:, :, 1] + beta[1:][:, ns[:, 1]]
        return _lse(m0) - _lse(m1)


def _lse(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1)
    return m + np.log(np.exp(x - m[:, None]).sum(axis=1))


class Turbo(BlockCodec):
    def __init__(self, K: int = 1024, iterations: int = 8, puncture: bool = False,
                 g_fb: int = 13, g_ff: int = 15, constraint: int = 4, seed: int | None = None):
        self.K = K
        self.iterations = iterations
        self.puncture = puncture
        self.nu = constraint - 1
        self.tr = rsc_trellis(g_fb, g_ff, constraint)
        self.bcjr = _BCJR(self.tr)
        self.perm = s_random_interleaver(K, seed)
        self.name = f"turbo{K}" + ("p" if puncture else "")
        par_idx = np.arange(K)
        self._keep1 = par_idx if not puncture else par_idx[0::2]
        self._keep2 = par_idx if not puncture else par_idx[1::2]

    @property
    def k(self) -> int:
        return self.K

    @property
    def n(self) -> int:
        return self.K + len(self._keep1) + len(self._keep2) + 4 * self.nu

    def encode_block(self, info: np.ndarray) -> np.ndarray:
        u = np.asarray(info, dtype=np.uint8)
        p1, tu1, tp1 = _encode_rsc(self.tr, u, self.nu)
        p2, tu2, tp2 = _encode_rsc(self.tr, u[self.perm], self.nu)
        return np.concatenate([u, p1[self._keep1], p2[self._keep2],
                               np.ravel(np.column_stack([tu1, tp1])),
                               np.ravel(np.column_stack([tu2, tp2]))]).astype(np.uint8)

    def decode_block(self, llr: np.ndarray) -> DecodeResult:
        K, nu, perm = self.K, self.nu, self.perm
        llr = np.clip(np.asarray(llr, dtype=np.float64), -60, 60)
        Ls = llr[:K]
        o = K
        Lp1 = np.zeros(K)
        Lp1[self._keep1] = llr[o:o + len(self._keep1)]
        o += len(self._keep1)
        Lp2 = np.zeros(K)
        Lp2[self._keep2] = llr[o:o + len(self._keep2)]
        o += len(self._keep2)
        t1 = llr[o:o + 2 * nu].reshape(nu, 2)
        t2 = llr[o + 2 * nu:o + 4 * nu].reshape(nu, 2)
        Ls2 = Ls[perm]
        La1 = np.zeros(K)
        prev_hard = None
        it = 0
        converged = False
        for it in range(1, self.iterations + 1):
            post1 = self.bcjr.llr(np.concatenate([Ls + La1, t1[:, 0]]), np.concatenate([Lp1, t1[:, 1]]))[:K]
            Le1 = post1 - Ls - La1
            La2 = Le1[perm]
            post2 = self.bcjr.llr(np.concatenate([Ls2 + La2, t2[:, 0]]), np.concatenate([Lp2, t2[:, 1]]))[:K]
            Le2 = post2 - Ls2 - La2
            La1 = np.empty(K)
            La1[perm] = Le2
            final = np.empty(K)
            final[perm] = post2
            hard = (final < 0).astype(np.uint8)
            hard1 = (post1 < 0).astype(np.uint8)
            if prev_hard is not None and np.array_equal(hard, prev_hard) and np.array_equal(hard, hard1):
                converged = True
                break
            prev_hard = hard
        return DecodeResult(bits=hard, ok=converged, iterations=it)
