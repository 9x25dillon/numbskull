"""Sound-card modem: real passband audio <-> complex baseband.

Connect the sound card to a transceiver's data port (DigiRig, SignaLink,
AIOC, a direct line-in/out cable, ...). The PHY runs at the audio sample
rate; this module mixes the complex baseband to a real signal centred on
``center`` Hz (default 1700 Hz, inside a 300-3000 Hz SSB/FM voice channel)
and back, keeps oscillator phase continuous across blocks, and keys the
transmitter via a :class:`~wavecaster.radio.ptt.PTT`.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np
from scipy.signal import firwin, lfilter

from .base import RadioDevice
from .ptt import PTT, NullPTT, open_ptt

log = logging.getLogger(__name__)


class Passband:
    """Stateful complex <-> real frequency translation.

    ``up``:   ``sqrt(2) * Re{x e^{j w n}}`` (power preserving)
    ``down``: ``LPF{ sqrt(2) * r e^{-j w n} }`` removes the 2*fc image.
    """

    def __init__(self, sample_rate: float, center: float, ntaps: int = 255, cutoff: float | None = None):
        self.fs, self.fc = float(sample_rate), float(center)
        cutoff = cutoff or min(self.fc, self.fs / 2 - self.fc) * 0.999
        self.taps = firwin(ntaps, cutoff, fs=self.fs)
        self._zi = np.zeros(ntaps - 1, dtype=np.complex128)
        self._n_up = 0
        self._n_down = 0

    def _osc(self, start: int, n: int) -> np.ndarray:
        return np.exp(2j * np.pi * self.fc * (np.arange(start, start + n) / self.fs))

    def up(self, iq: np.ndarray, level: float = 1.0) -> np.ndarray:
        iq = np.asarray(iq, dtype=np.complex128)
        y = np.sqrt(2) * np.real(iq * self._osc(self._n_up, len(iq)))
        self._n_up += len(iq)
        return (level * y).astype(np.float32)

    def down(self, real: np.ndarray) -> np.ndarray:
        real = np.asarray(real, dtype=np.float64)
        mixed = np.sqrt(2) * real * np.conj(self._osc(self._n_down, len(real)))
        self._n_down += len(real)
        out, self._zi = lfilter(self.taps, 1.0, mixed, zi=self._zi)
        return out.astype(np.complex64)


class AudioDevice(RadioDevice):
    """Full-duplex capable sound card via ``sounddevice`` (PortAudio).

    TX: ``transmit`` keys PTT, waits ``ptt_lead`` s, plays the burst
    (blocking), waits ``ptt_tail`` s, unkeys. When ``mute_during_tx`` the RX
    path discards captured audio while keyed (avoids decoding our own echo).
    RX: a PortAudio callback pushes blocks into a queue; ``read`` drains and
    downconverts them.
    """

    def __init__(self, sample_rate: float = 48000, center: float = 1700.0, input=None, output=None,
                 ptt: PTT | str | None = None, level: float = 0.5, ptt_lead: float = 0.08,
                 ptt_tail: float = 0.05, blocksize: int = 2048, mute_during_tx: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.center = center
        self.input_dev, self.output_dev = input, output
        self.ptt = open_ptt(ptt) if isinstance(ptt, str) or ptt is None else ptt
        self.level = level
        self.ptt_lead, self.ptt_tail = ptt_lead, ptt_tail
        self.blocksize = blocksize
        self.mute_during_tx = mute_during_tx
        self._pb_tx = Passband(sample_rate, center)
        self._pb_rx = Passband(sample_rate, center)
        self._q: queue.Queue = queue.Queue(maxsize=512)
        self._pending = np.zeros(0, np.complex64)
        self._keyed = threading.Event()
        self._stream = None
        self._sd = None

    def open(self) -> "AudioDevice":
        import sounddevice as sd
        self._sd = sd
        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                                      device=self.input_dev, blocksize=self.blocksize, callback=self._callback)
        self._stream.start()
        return super().open()

    def _callback(self, indata, frames, time_info, status):  # PortAudio thread
        if status:
            self.stats["overflows"] += 1
        block = indata[:, 0].copy()
        if self.mute_during_tx and self._keyed.is_set():
            block[:] = 0.0
        try:
            self._q.put_nowait(block)
        except queue.Full:
            self.stats["overflows"] += 1

    def transmit(self, iq: np.ndarray) -> None:
        if self._sd is None:
            raise RuntimeError("AudioDevice not opened")
        real = np.clip(self._pb_tx.up(iq, self.level), -1.0, 1.0)
        self._keyed.set()
        try:
            self.ptt.key()
            time.sleep(self.ptt_lead)
            self._sd.play(real, samplerate=self.sample_rate, device=self.output_dev, blocking=True)
            time.sleep(self.ptt_tail)
        finally:
            self.ptt.unkey()
            self._keyed.clear()
        self.stats["tx_samples"] += len(iq)
        self.stats["bursts"] += 1

    def read(self, n: int, timeout: float = 1.0) -> np.ndarray:
        deadline = time.monotonic() + timeout
        parts = [self._pending]
        have = len(self._pending)
        while have < n:
            try:
                blk = self._q.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                break
            bb = self._pb_rx.down(blk)
            parts.append(bb)
            have += len(bb)
        x = np.concatenate(parts)
        self._pending = x[n:]
        out = x[:n]
        self.stats["rx_samples"] += len(out)
        return out

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.ptt.close()
        super().close()


def list_audio_devices() -> list[dict]:
    import sounddevice as sd
    return [dict(index=i, name=d["name"], inputs=d["max_input_channels"], outputs=d["max_output_channels"],
                 default_rate=d["default_samplerate"]) for i, d in enumerate(sd.query_devices())]
