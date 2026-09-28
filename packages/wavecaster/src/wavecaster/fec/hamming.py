"""Hamming(7,4) with syndrome decoding (bit layout p1 p2 d0 p3 d1 d2 d3)."""

from __future__ import annotations

import numpy as np

from .base import BlockCodec, DecodeResult

# Generator (4x7) and parity-check (3x7) in the classical positional layout,
# where column j of H is the binary representation of j+1.
_G = np.array(
    [
        [1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1],
    ],
    dtype=np.uint8,
)
_H = np.array([[(j + 1) >> b & 1 for j in range(7)] for b in range(3)], dtype=np.uint8)
_DATA_POS = np.array([2, 4, 5, 6])


class Hamming74(BlockCodec):
    name = "hamming74"

    @property
    def k(self) -> int:
        return 4

    @property
    def n(self) -> int:
        return 7

    def encode_block(self, info: np.ndarray) -> np.ndarray:
        return (np.asarray(info, dtype=np.uint8) @ _G) % 2

    # Vectorised stream versions (the per-block API is kept for uniformity).
    def encode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8)
        nb = self.num_blocks(len(bits))
        padded = np.zeros(nb * 4, dtype=np.uint8)
        padded[: len(bits)] = bits
        return ((padded.reshape(nb, 4).astype(np.int32) @ _G) % 2).astype(np.uint8).ravel()

    def decode_block(self, llr: np.ndarray) -> DecodeResult:
        return self.decode(llr)

    def decode(self, llr: np.ndarray, nbits: int | None = None) -> DecodeResult:
        r = (np.asarray(llr) < 0).astype(np.uint8)
        nb = len(r) // 7
        if nb == 0:
            raise ValueError("hamming74: need at least 7 LLRs")
        cw = r[: nb * 7].reshape(nb, 7).copy()
        syn = ((cw.astype(np.int32) @ _H.T) % 2) @ np.array([1, 2, 4])
        err = np.nonzero(syn)[0]
        cw[err, syn[err] - 1] ^= 1
        bits = cw[:, _DATA_POS].ravel()
        if nbits is not None:
            bits = bits[:nbits]
        return DecodeResult(bits=bits, ok=True, corrected=int(len(err)))
