"""Channel impairment models for simulation and loopback testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Channel:
    """Complex baseband channel: delay, gain, phase, CFO, multipath, AWGN.

    ``esn0_db`` is the *received* Es/N0 for a unit-average-power transmit
    signal with ``sps`` samples per symbol: per-sample noise variance
    ``gain^2 * sps * 10^(-esn0/10)``.
    """

    esn0_db: float = 20.0
    sps: int = 8
    cfo: float = 0.0                 # cycles/sample
    phase: float = 0.0
    gain: float = 1.0
    delay: int = 0                   # leading noise-only samples
    taps: tuple = (1.0,)
    seed: int | None = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self._n = 0                  # running sample counter for continuous CFO phase

    @property
    def noise_var(self) -> float:
        return self.gain ** 2 * self.sps * 10 ** (-self.esn0_db / 10)

    def noise(self, n: int) -> np.ndarray:
        s = np.sqrt(self.noise_var / 2)
        return s * (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex128)
        if len(self.taps) > 1:
            x = np.convolve(x, np.asarray(self.taps, dtype=np.complex128))[: len(x)]
        x = np.concatenate([np.zeros(self.delay, dtype=np.complex128), x])
        n = np.arange(self._n, self._n + len(x))
        self._n += len(x)
        y = self.gain * x * np.exp(1j * (self.phase + 2 * np.pi * self.cfo * n))
        return (y + self.noise(len(y))).astype(np.complex64)
