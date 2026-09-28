"""Simulated and file-backed devices."""

from __future__ import annotations

import collections
import threading
import time
import wave
from pathlib import Path

import numpy as np

from ..channel import Channel
from .base import RadioDevice


class LoopbackDevice(RadioDevice):
    """TX bursts pass through a :class:`Channel` into this device's RX queue.

    While idle, ``read`` returns channel noise so the receiver sees a
    continuous stream exactly like live hardware. ``realtime=True`` paces
    reads to ``sample_rate``.
    """

    def __init__(self, sample_rate: float = 1e6, channel: Channel | None = None, realtime: bool = False):
        super().__init__()
        self.sample_rate = sample_rate
        self.channel = channel or Channel(esn0_db=30.0)
        self.realtime = realtime
        self._q: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._t0 = None
        self._read_total = 0
        self.half_duplex = False

    def transmit(self, iq: np.ndarray) -> None:
        y = self.channel(np.asarray(iq))
        with self._lock:
            self._q.append(y)
        self.stats["tx_samples"] += len(iq)
        self.stats["bursts"] += 1

    def read(self, n: int, timeout: float = 1.0) -> np.ndarray:
        if self.realtime:
            if self._t0 is None:
                self._t0 = time.monotonic()
            due = self._t0 + (self._read_total + n) / self.sample_rate
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(min(delay, timeout))
        out = []
        need = n
        with self._lock:
            while need and self._q:
                head = self._q[0]
                take = head[:need]
                out.append(take)
                need -= len(take)
                if len(take) == len(head):
                    self._q.popleft()
                else:
                    self._q[0] = head[len(take):]
        if need:
            out.append(self.channel.noise(need).astype(np.complex64))
        x = np.concatenate(out).astype(np.complex64)
        self._read_total += n
        self.stats["rx_samples"] += n
        return x

    @property
    def pending(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._q)


_FORMATS = {"cf32": np.complex64, "cs16": np.int16, "cu8": np.uint8, "cs8": np.int8}


def load_iq(path: str | Path, fmt: str = "cf32") -> np.ndarray:
    raw = np.fromfile(path, dtype=_FORMATS[fmt])
    if fmt == "cf32":
        return raw
    x = raw.astype(np.float32)
    if fmt == "cs16":
        x /= 32768.0
    elif fmt == "cu8":
        x = (x - 127.5) / 128.0
    elif fmt == "cs8":
        x /= 128.0
    return (x[0::2] + 1j * x[1::2]).astype(np.complex64)


def save_iq(path: str | Path, iq: np.ndarray, fmt: str = "cf32", append: bool = False) -> None:
    iq = np.asarray(iq, dtype=np.complex64)
    if fmt == "cf32":
        data = iq
    else:
        inter = np.empty(2 * len(iq), dtype=np.float32)
        inter[0::2], inter[1::2] = iq.real, iq.imag
        if fmt == "cs16":
            data = np.clip(inter * 32767, -32768, 32767).astype(np.int16)
        elif fmt == "cu8":
            data = np.clip(inter * 128 + 127.5, 0, 255).astype(np.uint8)
        else:
            data = np.clip(inter * 127, -128, 127).astype(np.int8)
    with open(path, "ab" if append else "wb") as f:
        data.tofile(f)


class FileDevice(RadioDevice):
    """Raw IQ files (GNU Radio ``cf32``, RTL-SDR ``cu8``, HackRF ``cs8``, ``cs16``).

    RX reads ``rx_path`` once (``exhausted`` afterwards); TX appends bursts to
    ``tx_path`` separated by ``gap`` zero samples.
    """

    def __init__(self, rx_path: str | None = None, tx_path: str | None = None, sample_rate: float = 1e6,
                 fmt: str = "cf32", gap: int = 1000):
        super().__init__()
        self.sample_rate = sample_rate
        self.fmt, self.gap = fmt, gap
        self.tx_path = tx_path
        self._rx = load_iq(rx_path, fmt) if rx_path else np.zeros(0, np.complex64)
        self._pos = 0
        if tx_path:
            Path(tx_path).write_bytes(b"")

    def transmit(self, iq: np.ndarray) -> None:
        if not self.tx_path:
            raise RuntimeError("FileDevice opened without tx_path")
        save_iq(self.tx_path, np.concatenate([np.asarray(iq, np.complex64), np.zeros(self.gap, np.complex64)]),
                self.fmt, append=True)
        self.stats["tx_samples"] += len(iq)
        self.stats["bursts"] += 1

    def read(self, n: int, timeout: float = 1.0) -> np.ndarray:
        x = self._rx[self._pos:self._pos + n]
        self._pos += len(x)
        self.stats["rx_samples"] += len(x)
        return x

    @property
    def exhausted(self) -> bool:
        return self._pos >= len(self._rx)


class WavDevice(RadioDevice):
    """Real passband audio in WAV files, converted to/from complex baseband.

    Lets audio-profile transmissions be recorded, replayed through any
    transceiver or decoded from recordings made off-air.
    """

    def __init__(self, rx_path: str | None = None, tx_path: str | None = None, sample_rate: float = 48000,
                 center: float = 1700.0, level: float = 0.5, gap_s: float = 0.25):
        super().__init__()
        from .audio import Passband
        self.sample_rate = sample_rate
        self.center, self.level, self.gap_s = center, level, gap_s
        self.tx_path = tx_path
        self._pb_rx = Passband(sample_rate, center)
        self._pb_tx = Passband(sample_rate, center)
        self._tx_chunks: list[np.ndarray] = []
        self._rx = np.zeros(0, np.complex64)
        self._pos = 0
        if rx_path:
            with wave.open(str(rx_path), "rb") as w:
                sr = w.getframerate()
                ch = w.getnchannels()
                width = w.getsampwidth()
                frames = w.readframes(w.getnframes())
            if width != 2:
                raise ValueError("only 16-bit PCM WAV supported")
            pcm = np.frombuffer(frames, dtype=np.int16).reshape(-1, ch)[:, 0].astype(np.float32) / 32768.0
            if sr != sample_rate:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(int(sr), int(sample_rate))
                pcm = resample_poly(pcm, int(sample_rate) // g, int(sr) // g).astype(np.float32)
            self._rx = self._pb_rx.down(pcm)

    def transmit(self, iq: np.ndarray) -> None:
        if not self.tx_path:
            raise RuntimeError("WavDevice opened without tx_path")
        real = self._pb_tx.up(iq, self.level)
        self._tx_chunks += [real, np.zeros(int(self.gap_s * self.sample_rate), np.float32)]
        self.stats["tx_samples"] += len(iq)
        self.stats["bursts"] += 1
        self._flush()

    def _flush(self) -> None:
        pcm = (np.clip(np.concatenate(self._tx_chunks), -1, 1) * 32767).astype(np.int16)
        with wave.open(str(self.tx_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(self.sample_rate))
            w.writeframes(pcm.tobytes())

    def read(self, n: int, timeout: float = 1.0) -> np.ndarray:
        x = self._rx[self._pos:self._pos + n]
        self._pos += len(x)
        self.stats["rx_samples"] += len(x)
        return x

    @property
    def exhausted(self) -> bool:
        return self._pos >= len(self._rx)
