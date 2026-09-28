"""Bit/byte primitives, CRCs and scrambling.

Conventions used throughout wavecaster:

* bits are ``np.uint8`` arrays holding 0/1, MSB-first within each byte;
* soft values are log-likelihood ratios ``L = ln P(b=0) / P(b=1)``, so a
  positive LLR favours bit 0 and hard decision is ``bit = L < 0``.
"""

from __future__ import annotations

import binascii

import numpy as np

HARD_LLR = 32.0  # magnitude used when turning hard bits into LLRs


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    return np.packbits(bits).tobytes()


def bits_to_llr(bits: np.ndarray, magnitude: float = HARD_LLR) -> np.ndarray:
    return magnitude * (1.0 - 2.0 * np.asarray(bits, dtype=np.float64))


def hard_decision(llr: np.ndarray) -> np.ndarray:
    return (np.asarray(llr) < 0).astype(np.uint8)


def crc32(data: bytes) -> int:
    return binascii.crc32(data) & 0xFFFFFFFF


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    return binascii.crc_hqx(data, init)


def crc8(data: bytes, poly: int = 0x07) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def lfsr_sequence(length: int, taps: int = 0x48, seed: int = 0x7F, width: int = 7) -> np.ndarray:
    """Additive scrambler sequence (default x^7 + x^4 + 1, as in IEEE 802.11)."""
    out = np.empty(length, dtype=np.uint8)
    state = seed & ((1 << width) - 1)
    for i in range(length):
        fb = bin(state & taps).count("1") & 1
        out[i] = fb
        state = ((state << 1) | fb) & ((1 << width) - 1)
    return out


_SCRAMBLE_SEQ = lfsr_sequence(1 << 14)


def _scramble_seq(n: int) -> np.ndarray:
    global _SCRAMBLE_SEQ
    if len(_SCRAMBLE_SEQ) < n:
        _SCRAMBLE_SEQ = lfsr_sequence(n)
    return _SCRAMBLE_SEQ[:n]


def scramble(bits: np.ndarray) -> np.ndarray:
    """Whitening: XOR with a fixed LFSR sequence (self-inverse)."""
    bits = np.asarray(bits, dtype=np.uint8)
    return np.bitwise_xor(bits, _scramble_seq(len(bits)))


def descramble_llr(llr: np.ndarray) -> np.ndarray:
    """Scrambling in the LLR domain flips the sign where the sequence is 1."""
    llr = np.asarray(llr, dtype=np.float64)
    return np.where(_scramble_seq(len(llr)) == 1, -llr, llr)
