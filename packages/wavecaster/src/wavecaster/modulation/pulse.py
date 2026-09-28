"""Pulse shaping filters."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def rrc_taps(sps: int, rolloff: float = 0.35, span: int = 8) -> np.ndarray:
    """Unit-energy root-raised-cosine taps, length ``span*sps + 1``.

    With unit energy, TX RRC followed by RX matched RRC has unit gain at the
    optimum sampling instant and zero ISI at multiples of ``sps``.
    """
    beta = float(rolloff)
    t = (np.arange(span * sps + 1) - span * sps / 2) / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 + beta * (4 / np.pi - 1)
        elif beta > 0 and abs(abs(4 * beta * ti) - 1.0) < 1e-9:
            h[i] = (beta / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                          + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    h.setflags(write=False)
    return h


def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    out = np.zeros(len(symbols) * sps, dtype=np.complex128)
    out[::sps] = symbols
    return out


def shape(symbols: np.ndarray, sps: int, rolloff: float, span: int) -> np.ndarray:
    """Pulse-shape symbols; output length ``len*sps + len(taps) - 1``.

    Symbol ``k`` is centred at sample ``k*sps + (len(taps)-1)/2``; after the
    matched filter it peaks at ``k*sps + len(taps) - 1``.
    """
    h = rrc_taps(sps, rolloff, span)
    return np.convolve(upsample(symbols, sps), h)


def matched_filter(x: np.ndarray, sps: int, rolloff: float, span: int) -> np.ndarray:
    h = rrc_taps(sps, rolloff, span)
    if len(x) > 4096:
        from scipy.signal import fftconvolve
        return fftconvolve(x, h[::-1].conj())
    return np.convolve(x, h[::-1].conj())
