"""FEC codec contract.

Every codec is a *block* code mapping ``k`` information bits to ``n`` coded
bits. Streams are handled by :meth:`BlockCodec.encode` /
:meth:`BlockCodec.decode`, which zero-pad the last block; the caller keeps
track of the true payload length (the PHY header carries it).

Decoders consume LLRs (``ln P(0)/P(1)``) so soft-decision codes (LDPC, turbo,
convolutional) get full channel information while hard-decision codes
(Reed-Solomon, Hamming) simply threshold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecodeResult:
    bits: np.ndarray            # decoded information bits (uint8 0/1)
    ok: bool                    # every block passed the code's own check
    corrected: int = 0          # corrected symbols/bits (when the code knows)
    failed_blocks: int = 0
    iterations: int = 0         # iterative decoders: max iterations used


class BlockCodec(ABC):
    name: str = "abstract"

    @property
    @abstractmethod
    def k(self) -> int: ...

    @property
    @abstractmethod
    def n(self) -> int: ...

    @property
    def rate(self) -> float:
        return self.k / self.n

    @abstractmethod
    def encode_block(self, info: np.ndarray) -> np.ndarray:
        """``k`` info bits -> ``n`` coded bits."""

    @abstractmethod
    def decode_block(self, llr: np.ndarray) -> DecodeResult:
        """``n`` LLRs -> ``k`` info bits."""

    def num_blocks(self, nbits: int) -> int:
        return max(1, -(-nbits // self.k))

    def encoded_length(self, nbits: int) -> int:
        return self.num_blocks(nbits) * self.n

    def encode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8)
        nb = self.num_blocks(len(bits))
        padded = np.zeros(nb * self.k, dtype=np.uint8)
        padded[: len(bits)] = bits
        blocks = padded.reshape(nb, self.k)
        return np.concatenate([self.encode_block(b) for b in blocks])

    def decode(self, llr: np.ndarray, nbits: int | None = None) -> DecodeResult:
        llr = np.asarray(llr, dtype=np.float64)
        nb = len(llr) // self.n
        if nb == 0:
            raise ValueError(f"{self.name}: need at least {self.n} LLRs, got {len(llr)}")
        results = [self.decode_block(llr[i * self.n:(i + 1) * self.n]) for i in range(nb)]
        bits = np.concatenate([r.bits for r in results])
        if nbits is not None:
            bits = bits[:nbits]
        failed = sum(1 for r in results if not r.ok)
        return DecodeResult(
            bits=bits,
            ok=failed == 0,
            corrected=sum(r.corrected for r in results),
            failed_blocks=failed,
            iterations=max(r.iterations for r in results),
        )


class NullCodec(BlockCodec):
    """Uncoded pass-through (k = n = 8 for byte alignment)."""

    name = "none"

    @property
    def k(self) -> int:
        return 8

    @property
    def n(self) -> int:
        return 8

    def encode_block(self, info: np.ndarray) -> np.ndarray:
        return np.asarray(info, dtype=np.uint8).copy()

    def decode_block(self, llr: np.ndarray) -> DecodeResult:
        return DecodeResult(bits=(llr < 0).astype(np.uint8), ok=True)
