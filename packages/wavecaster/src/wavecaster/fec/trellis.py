"""Trellis tables for binary convolutional encoders.

Generators are given in octal with the **MSB as the D^0 tap** (textbook
convention: LTE RSC ``g0 = 13, g1 = 15``, NASA K=7 ``171, 133``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def octal_taps(g: int, K: int) -> list[int]:
    """Octal generator -> tap list [c_0 (D^0), c_1, ..., c_{K-1}]."""
    bits = bin(int(str(g), 8))[2:].zfill(K)
    return [int(b) for b in bits]


@dataclass(frozen=True)
class Trellis:
    n_states: int
    next_state: np.ndarray   # (S, 2) input u -> next state
    outputs: np.ndarray      # (S, 2, n_out) coded bits for input u
    term_input: np.ndarray   # (S,) input that drives the register towards 0 (RSC)


def rsc_trellis(g_fb: int = 13, g_ff: int = 15, K: int = 4) -> Trellis:
    """Recursive systematic encoder: outputs (u, parity)."""
    fb, ff = octal_taps(g_fb, K), octal_taps(g_ff, K)
    nu = K - 1
    S = 1 << nu
    nxt = np.zeros((S, 2), dtype=np.int64)
    out = np.zeros((S, 2, 2), dtype=np.uint8)
    term = np.zeros(S, dtype=np.uint8)
    for s in range(S):
        reg = [(s >> (nu - 1 - i)) & 1 for i in range(nu)]  # reg[0] = most recent
        fbsum = sum(fb[i + 1] & reg[i] for i in range(nu)) & 1
        term[s] = fbsum
        for u in (0, 1):
            a = u ^ fbsum
            p = (ff[0] & a) ^ (sum(ff[i + 1] & reg[i] for i in range(nu)) & 1)
            new = [a] + reg[:-1]
            nxt[s, u] = sum(b << (nu - 1 - i) for i, b in enumerate(new))
            out[s, u] = (u, p)
    return Trellis(S, nxt, out, term)


def feedforward_trellis(gens: tuple[int, ...] = (171, 133), K: int = 7) -> Trellis:
    """Non-recursive rate 1/len(gens) encoder."""
    taps = [octal_taps(g, K) for g in gens]
    nu = K - 1
    S = 1 << nu
    nxt = np.zeros((S, 2), dtype=np.int64)
    out = np.zeros((S, 2, len(gens)), dtype=np.uint8)
    for s in range(S):
        reg = [(s >> (nu - 1 - i)) & 1 for i in range(nu)]
        for u in (0, 1):
            window = [u] + reg
            out[s, u] = [sum(t & w for t, w in zip(tp, window)) & 1 for tp in taps]
            new = [u] + reg[:-1]
            nxt[s, u] = sum(b << (nu - 1 - i) for i, b in enumerate(new))
    return Trellis(S, nxt, out, np.zeros(S, dtype=np.uint8))
