"""Real-time link: PHY + radio device + background receive thread.

::

    with Link(open_device("soapy:driver=uhd", sample_rate=1e6, center_freq=915e6),
              get_profile("sdr-narrow")) as link:
        link.send(b"hello")
        for frame in link.frames(timeout=5):
            print(frame.payload)

The RX thread pulls fixed-size blocks from the device and feeds the
streaming :class:`~wavecaster.phy.Receiver`; completed frames go to a
queue (or a callback). TX runs in the caller's thread. On half-duplex
devices the RX thread keeps running during TX; the audio device mutes its
own capture while keyed.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Iterator

from .phy import Frame, PHYConfig, Receiver, Transmitter
from .radio.base import RadioDevice
from .transport import MessageReassembler, Segmenter, StreamReassembler

log = logging.getLogger(__name__)


class Link:
    def __init__(self, device: RadioDevice, cfg: PHYConfig, block: int = 8192, max_payload: int = 4096,
                 on_frame: Callable[[Frame], None] | None = None, deliver_bad: bool = False):
        if abs(device.sample_rate - cfg.sample_rate) > 1e-6 * cfg.sample_rate:
            raise ValueError(f"device rate {device.sample_rate} != PHY rate {cfg.sample_rate}")
        self.device, self.cfg = device, cfg
        self.tx = Transmitter(cfg)
        self.rx = Receiver(cfg, max_payload=max_payload)
        self.block = block
        self.on_frame = on_frame
        self.deliver_bad = deliver_bad
        self._q: queue.Queue[Frame] = queue.Queue()
        self._seq = 0
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self.error: BaseException | None = None

    # lifecycle ----------------------------------------------------------------
    def __enter__(self) -> "Link":
        if not self.device._opened:
            self.device.open()
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        self.device.close()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._rx_loop, name="wavecaster-rx", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout)

    def _rx_loop(self) -> None:
        try:
            while self._running.is_set():
                x = self.device.read(self.block, timeout=0.5)
                if not len(x):
                    if self.device.exhausted:
                        for fr in self.rx.flush():
                            self._deliver(fr)
                        break
                    continue
                for fr in self.rx.feed(x):
                    self._deliver(fr)
        except BaseException as exc:  # surface driver errors to the caller
            log.exception("RX thread failed")
            self.error = exc
        finally:
            self._running.clear()

    def _deliver(self, fr: Frame) -> None:
        if not fr.ok and not self.deliver_bad:
            return
        if self.on_frame:
            self.on_frame(fr)
        else:
            self._q.put(fr)

    # data path ------------------------------------------------------------------
    def send(self, payload: bytes, seq: int | None = None) -> int:
        s = self._seq if seq is None else seq
        self._seq = (s + 1) & 0xFFFF
        self.device.transmit(self.tx.modulate(payload, seq=s, pad=self.cfg.sps * 16))
        return s

    def frames(self, timeout: float | None = None, max_frames: int | None = None) -> Iterator[Frame]:
        deadline = None if timeout is None else time.monotonic() + timeout
        n = 0
        while max_frames is None or n < max_frames:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return
            try:
                fr = self._q.get(timeout=0.1 if remaining is None else min(0.1, remaining))
            except queue.Empty:
                if self.error:
                    raise RuntimeError("receiver thread failed") from self.error
                if not self._running.is_set() and self._q.empty():
                    return
                continue
            n += 1
            yield fr

    # transport helpers -----------------------------------------------------------
    def send_message(self, data: bytes, mtu: int = 1024, stream_id: int = 0) -> int:
        segs = Segmenter(mtu, stream_id).message(data)
        for s in segs:
            self.send(s)
        return len(segs)

    def receive_message(self, timeout: float = 10.0) -> bytes | None:
        r = MessageReassembler()
        for fr in self.frames(timeout=timeout):
            msg = r.push(fr.payload)
            if msg is not None:
                return msg
        return None

    def stream_out(self, chunks: Iterator[bytes], mtu: int = 256, stream_id: int = 0) -> int:
        """Transmit a live byte stream (e.g. Codec2/Opus audio from stdin)."""
        seg = Segmenter(mtu, stream_id)
        n = 0
        for chunk in chunks:
            for s in seg.stream(chunk):
                self.send(s)
                n += 1
        for s in seg.stream(b"", final=True):
            self.send(s)
            n += 1
        return n

    def stream_in(self, timeout: float | None = None, window: int = 4) -> Iterator[bytes]:
        r = StreamReassembler(window)
        for fr in self.frames(timeout=timeout):
            data = r.push(fr.payload)
            if data:
                yield data
            if r.ended:
                return
