"""Segmentation and reassembly of byte streams over PHY frames.

Transport header (4 bytes, inside the PHY payload)::

    stream_id(1) flags(1) index(2)      flags: bit0 = first, bit1 = last

* :class:`Segmenter` cuts a message (or a live stream) into MTU-sized
  segments.
* :class:`MessageReassembler` rebuilds whole messages (needs every segment).
* :class:`StreamReassembler` emits in-order bytes for real-time streams such
  as compressed audio: missing segments are skipped once ``window``
  later segments have arrived, trading completeness for bounded latency.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

FIRST, LAST = 1, 2
HDR = struct.Struct(">BBH")


@dataclass
class Segment:
    stream_id: int
    index: int
    first: bool
    last: bool
    data: bytes

    def pack(self) -> bytes:
        return HDR.pack(self.stream_id, (FIRST if self.first else 0) | (LAST if self.last else 0),
                        self.index & 0xFFFF) + self.data

    @classmethod
    def unpack(cls, raw: bytes) -> "Segment":
        if len(raw) < HDR.size:
            raise ValueError("segment too short")
        sid, flags, idx = HDR.unpack(raw[:HDR.size])
        return cls(sid, idx, bool(flags & FIRST), bool(flags & LAST), raw[HDR.size:])


class Segmenter:
    def __init__(self, mtu: int = 255, stream_id: int = 0):
        if mtu <= HDR.size:
            raise ValueError("mtu too small")
        self.mtu, self.stream_id = mtu, stream_id & 0xFF
        self._index = 0
        self._started = False

    def message(self, data: bytes) -> list[bytes]:
        """Whole message -> packed segments (first..last)."""
        room = self.mtu - HDR.size
        chunks = [data[i:i + room] for i in range(0, len(data), room)] or [b""]
        out = []
        for i, c in enumerate(chunks):
            out.append(Segment(self.stream_id, self._index, i == 0, i == len(chunks) - 1, c).pack())
            self._index = (self._index + 1) & 0xFFFF
        return out

    def stream(self, data: bytes, final: bool = False) -> list[bytes]:
        """Continuous stream chunk -> segments; ``final`` marks end of stream."""
        room = self.mtu - HDR.size
        chunks = [data[i:i + room] for i in range(0, len(data), room)] or ([b""] if final else [])
        out = []
        for i, c in enumerate(chunks):
            out.append(Segment(self.stream_id, self._index, not self._started,
                               final and i == len(chunks) - 1, c).pack())
            self._started = True
            self._index = (self._index + 1) & 0xFFFF
        return out


@dataclass
class MessageReassembler:
    max_pending: int = 64
    _parts: dict = field(default_factory=dict)

    def push(self, raw: bytes) -> bytes | None:
        seg = Segment.unpack(raw)
        key = seg.stream_id
        st = self._parts.setdefault(key, {"segs": {}, "first": None, "last": None})
        st["segs"][seg.index] = seg.data
        if seg.first:
            st["first"] = seg.index
        if seg.last:
            st["last"] = seg.index
        if st["first"] is not None and st["last"] is not None:
            n = (st["last"] - st["first"]) & 0xFFFF
            idxs = [(st["first"] + i) & 0xFFFF for i in range(n + 1)]
            if all(i in st["segs"] for i in idxs):
                del self._parts[key]
                return b"".join(st["segs"][i] for i in idxs)
        if len(st["segs"]) > self.max_pending:
            del self._parts[key]
        return None


class StreamReassembler:
    """In-order delivery with bounded reordering window (lossy, low latency)."""

    def __init__(self, window: int = 4):
        self.window = window
        self._next: int | None = None
        self._buf: dict[int, bytes] = {}
        self.lost = 0
        self.ended = False

    def push(self, raw: bytes) -> bytes:
        seg = Segment.unpack(raw)
        if seg.first and self._next is None:
            self._next = seg.index
        if self._next is None:
            self._next = seg.index  # joined mid-stream
        ahead = (seg.index - self._next) & 0xFFFF
        if ahead >= 0x8000:
            return b""  # late duplicate
        self._buf[seg.index] = seg.data
        if seg.last:
            self.ended = True
        out = bytearray()
        while True:
            if self._next in self._buf:
                out += self._buf.pop(self._next)
                self._next = (self._next + 1) & 0xFFFF
            elif self._buf and max((i - self._next) & 0xFFFF for i in self._buf) >= self.window:
                self.lost += 1
                self._next = (self._next + 1) & 0xFFFF
            else:
                break
        return bytes(out)
