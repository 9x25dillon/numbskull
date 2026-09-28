"""Binary LDPC codes.

* Construction: progressive edge-growth (PEG, Hu/Eleftheriou/Arnold 2005),
  which maximises local girth, or import of any parity-check matrix in the
  standard ``alist`` format (e.g. 802.11n / DVB-S2 matrices).
* Encoding: ``H`` is brought to reduced row-echelon form over GF(2); pivot
  columns carry parity, the remaining ``k = n - rank(H)`` columns carry
  information. Handles rank-deficient ``H``.
* Decoding: flooding belief propagation, either exact sum-product in the
  ``phi`` (log-tanh) domain or normalised min-sum; early stop on zero
  syndrome. All message passing is vectorised over a padded check-by-degree
  matrix, so an iteration costs O(E) numpy work.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .base import BlockCodec, DecodeResult


def gf2_rref(H: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row-echelon form over GF(2). Returns (R, pivot_columns)."""
    R = (np.asarray(H) & 1).astype(np.uint8).copy()
    m, n = R.shape
    pivots: list[int] = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        nz = np.nonzero(R[row:, col])[0]
        if len(nz) == 0:
            continue
        p = row + nz[0]
        if p != row:
            R[[row, p]] = R[[p, row]]
        others = np.nonzero(R[:, col])[0]
        others = others[others != row]
        R[others] ^= R[row]
        pivots.append(col)
        row += 1
    return R[:row], pivots


def peg_matrix(n: int, m: int, dv: int, seed: int = 0) -> np.ndarray:
    """Progressive edge-growth parity-check matrix with variable degree ``dv``."""
    if dv > m:
        raise ValueError("dv must not exceed number of checks")
    rng = np.random.default_rng(seed)
    chk_deg = np.zeros(m, dtype=np.int64)
    var_adj: list[list[int]] = [[] for _ in range(n)]
    chk_adj: list[list[int]] = [[] for _ in range(m)]

    def pick(cands: np.ndarray) -> int:
        degs = chk_deg[cands]
        best = cands[degs == degs.min()]
        return int(best[rng.integers(len(best))])

    for j in range(n):
        for e in range(dv):
            if e == 0:
                c = pick(np.arange(m))
            else:
                # BFS layer by layer from variable j; connect to the least
                # loaded check that is unreachable, or else in the deepest layer.
                reached = np.zeros(m, dtype=bool)
                reached[var_adj[j]] = True
                layer = list(var_adj[j])
                while True:
                    vs = {v for c2 in layer for v in chk_adj[c2]}
                    new = {c2 for v in vs for c2 in var_adj[v] if not reached[c2]}
                    if not new or reached.sum() + len(new) == m:
                        cands = np.nonzero(~reached)[0]
                        break
                    reached[list(new)] = True
                    layer = list(new)
                c = pick(cands)
            var_adj[j].append(c)
            chk_adj[c].append(j)
            chk_deg[c] += 1
    H = np.zeros((m, n), dtype=np.uint8)
    for j, cs in enumerate(var_adj):
        H[cs, j] = 1
    return H


def parse_alist(text: str) -> np.ndarray:
    """Parse MacKay's alist format into a dense parity-check matrix."""
    toks = [int(t) for t in text.split()]
    n, m = toks[0], toks[1]
    pos = 4 + n + m  # skip header (n m / max degrees / per-node degrees)
    col_deg = toks[4:4 + n]
    H = np.zeros((m, n), dtype=np.uint8)
    max_col = toks[2]
    for j in range(n):
        entries = toks[pos:pos + max_col]
        pos += max_col
        for r in entries[: col_deg[j]]:
            if r > 0:
                H[r - 1, j] = 1
    return H


def to_alist(H: np.ndarray) -> str:
    m, n = H.shape
    cols = [np.nonzero(H[:, j])[0] + 1 for j in range(n)]
    rows = [np.nonzero(H[i])[0] + 1 for i in range(m)]
    mc, mr = max(map(len, cols)), max(map(len, rows))
    out = [f"{n} {m}", f"{mc} {mr}", " ".join(str(len(c)) for c in cols), " ".join(str(len(r)) for r in rows)]
    out += [" ".join(map(str, list(c) + [0] * (mc - len(c)))) for c in cols]
    out += [" ".join(map(str, list(r) + [0] * (mr - len(r)))) for r in rows]
    return "\n".join(out) + "\n"


class LDPC(BlockCodec):
    def __init__(self, H: np.ndarray, max_iter: int = 50, algorithm: str = "spa", alpha: float = 0.8):
        H = (np.asarray(H) & 1).astype(np.uint8)
        self.H = H
        self.max_iter = max_iter
        if algorithm not in ("spa", "minsum"):
            raise ValueError("algorithm must be 'spa' or 'minsum'")
        self.algorithm = algorithm
        self.alpha = alpha
        m, n = H.shape
        self._n = n
        R, piv = gf2_rref(H)
        self._piv = np.array(piv, dtype=np.int64)
        self._info = np.setdiff1d(np.arange(n), self._piv)
        self._P = R[:, self._info].astype(np.int32)  # parity = P @ info mod 2
        # edge structure (edges sorted by check)
        chk, var = np.nonzero(H)
        self._ec, self._ev = chk, var
        deg = np.bincount(chk, minlength=m)
        dmax = int(deg.max())
        slot = np.arange(len(chk)) - np.repeat(np.cumsum(deg) - deg, deg)
        self._cmat = np.full((m, dmax), -1, dtype=np.int64)
        self._cmat[chk, slot] = np.arange(len(chk))
        self._cmask = self._cmat >= 0
        self._cidx = np.where(self._cmask, self._cmat, 0)
        self._m = m
        self.name = f"ldpc{n}_{len(self._info)}"

    @classmethod
    def peg(cls, n: int = 1024, rate: float = 0.5, dv: int = 3, seed: int = 0, **kw) -> "LDPC":
        m = int(round(n * (1 - rate)))
        return cls(_cached_peg(n, m, dv, seed), **kw)

    @classmethod
    def from_alist(cls, text_or_path: str, **kw) -> "LDPC":
        text = text_or_path
        if "\n" not in text_or_path:
            with open(text_or_path) as f:
                text = f.read()
        return cls(parse_alist(text), **kw)

    @property
    def k(self) -> int:
        return len(self._info)

    @property
    def n(self) -> int:
        return self._n

    def encode_block(self, info: np.ndarray) -> np.ndarray:
        info = np.asarray(info, dtype=np.int32)
        cw = np.zeros(self._n, dtype=np.uint8)
        cw[self._info] = info
        cw[self._piv] = (self._P @ info) % 2
        return cw

    def syndrome(self, cw: np.ndarray) -> np.ndarray:
        return np.bincount(self._ec, weights=cw[self._ev], minlength=self._m).astype(np.int64) % 2

    def decode_block(self, llr: np.ndarray) -> DecodeResult:
        llr = np.clip(np.asarray(llr, dtype=np.float64), -50, 50)
        E = len(self._ev)
        c2v = np.zeros(E)
        hard = (llr < 0).astype(np.uint8)
        if not self.syndrome(hard).any():
            return DecodeResult(bits=hard[self._info], ok=True, iterations=0)
        it = 0
        for it in range(1, self.max_iter + 1):
            total = llr + np.bincount(self._ev, weights=c2v, minlength=self._n)
            v2c = total[self._ev] - c2v
            M = np.where(self._cmask, v2c[self._cidx], np.inf)
            sgn = np.where(M < 0, -1.0, 1.0)
            tsgn = np.prod(sgn, axis=1, keepdims=True)
            A = np.abs(M)
            if self.algorithm == "spa":
                phi = _phi(np.clip(A, 1e-12, 50))
                phi = np.where(self._cmask, phi, 0.0)
                mag = _phi(np.clip(phi.sum(axis=1, keepdims=True) - phi, 1e-12, 50))
            else:
                i1 = np.argmin(A, axis=1)
                rows = np.arange(self._m)
                m1 = A[rows, i1]
                A2 = A.copy()
                A2[rows, i1] = np.inf
                m2 = A2.min(axis=1)
                mag = np.where(np.arange(A.shape[1])[None, :] == i1[:, None], m2[:, None], m1[:, None])
                mag = self.alpha * mag
            out = tsgn * sgn * mag
            c2v[self._cmat[self._cmask]] = out[self._cmask]
            total = llr + np.bincount(self._ev, weights=c2v, minlength=self._n)
            hard = (total < 0).astype(np.uint8)
            if not self.syndrome(hard).any():
                return DecodeResult(bits=hard[self._info], ok=True, iterations=it)
        return DecodeResult(bits=hard[self._info], ok=False, iterations=it)


def _phi(x: np.ndarray) -> np.ndarray:
    return -np.log(np.tanh(x / 2.0))


@lru_cache(maxsize=16)
def _cached_peg(n: int, m: int, dv: int, seed: int) -> np.ndarray:
    H = peg_matrix(n, m, dv, seed)
    H.setflags(write=False)
    return H
