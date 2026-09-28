"""Radio device contract.

Every device, whether SDR, sound card or file, exposes the same complex-baseband
interface, so the PHY never knows what hardware it is driving:

* ``transmit(iq)`` sends one burst (blocking until handed to hardware);
* ``read(n, timeout)`` returns up to ``n`` complex64 RX samples (possibly 0
  on timeout/overflow; overflows are counted in ``stats``);
* ``sample_rate`` is the complex baseband rate the PHY must use.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class RadioDevice(ABC):
    sample_rate: float = 0.0
    half_duplex: bool = True

    def __init__(self) -> None:
        self.stats = {"rx_samples": 0, "tx_samples": 0, "overflows": 0, "underflows": 0, "bursts": 0}
        self._stop = threading.Event()
        self._opened = False

    # lifecycle ---------------------------------------------------------------
    def open(self) -> "RadioDevice":
        self._opened = True
        return self

    def close(self) -> None:
        self._stop.set()
        self._opened = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # I/O -----------------------------------------------------------------------
    @abstractmethod
    def transmit(self, iq: np.ndarray) -> None: ...

    @abstractmethod
    def read(self, n: int, timeout: float = 1.0) -> np.ndarray: ...

    def rx_stream(self, block: int = 8192, max_samples: int | None = None) -> Iterator[np.ndarray]:
        got = 0
        while not self._stop.is_set():
            x = self.read(block)
            if len(x):
                got += len(x)
                yield x
            elif self.exhausted:
                return
            if max_samples is not None and got >= max_samples:
                return

    @property
    def exhausted(self) -> bool:
        """True when a finite source (file) has no more samples."""
        return False

    def stop(self) -> None:
        self._stop.set()

    def describe(self) -> dict:
        return {"type": type(self).__name__, "sample_rate": self.sample_rate, **self.stats}


def peak_normalize(iq: np.ndarray, amplitude: float) -> np.ndarray:
    """Scale a burst so its peak magnitude equals ``amplitude`` (DAC headroom)."""
    peak = float(np.max(np.abs(iq))) if len(iq) else 0.0
    if peak == 0:
        return iq.astype(np.complex64)
    return (iq * (amplitude / peak)).astype(np.complex64)
