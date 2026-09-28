"""KISS TNC client and AX.25 UI framing.

Integrates with external modems speaking KISS: Direwolf (TCP 8001), hardware
TNCs (Mobilinkd, Kantronics, TNC-Pi over serial), soundmodem, or UZ7HO. The
modem does the waveform; wavecaster supplies payloads and framing.
"""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass

FEND, FESC, TFEND, TFESC = 0xC0, 0xDB, 0xDC, 0xDD


def kiss_encode(data: bytes, port: int = 0, cmd: int = 0) -> bytes:
    body = bytearray([((port & 0x0F) << 4) | (cmd & 0x0F)])
    for b in data:
        if b == FEND:
            body += bytes([FESC, TFEND])
        elif b == FESC:
            body += bytes([FESC, TFESC])
        else:
            body.append(b)
    return bytes([FEND]) + bytes(body) + bytes([FEND])


class KissDecoder:
    """Incremental decoder: ``feed(bytes) -> [(port, cmd, payload), ...]``."""

    def __init__(self):
        self._buf = bytearray()
        self._in = False
        self._esc = False

    def feed(self, data: bytes) -> list[tuple[int, int, bytes]]:
        out = []
        for b in data:
            if b == FEND:
                if self._in and self._buf:
                    t = self._buf[0]
                    out.append((t >> 4, t & 0x0F, bytes(self._buf[1:])))
                self._buf.clear()
                self._in = True
                self._esc = False
            elif not self._in:
                continue
            elif self._esc:
                self._buf.append(FEND if b == TFEND else FESC if b == TFESC else b)
                self._esc = False
            elif b == FESC:
                self._esc = True
            else:
                self._buf.append(b)
        return out


class KissTNC:
    """KISS over TCP (``host:port``) or serial (``/dev/tty...``)."""

    def __init__(self, address: str = "localhost:8001", baud: int = 9600, timeout: float = 1.0):
        self.dec = KissDecoder()
        self._lock = threading.Lock()
        if address.startswith("/dev/") or address.upper().startswith("COM"):
            import serial
            self._ser = serial.Serial(address, baud, timeout=timeout)
            self._sock = None
        else:
            host, _, port = address.rpartition(":")
            self._sock = socket.create_connection((host or "localhost", int(port)), timeout=timeout)
            self._sock.settimeout(timeout)
            self._ser = None

    @classmethod
    def from_socket(cls, sock: socket.socket) -> "KissTNC":
        obj = cls.__new__(cls)
        obj.dec, obj._lock, obj._sock, obj._ser = KissDecoder(), threading.Lock(), sock, None
        return obj

    def send(self, data: bytes, port: int = 0) -> None:
        frame = kiss_encode(data, port)
        with self._lock:
            if self._sock is not None:
                self._sock.sendall(frame)
            else:
                self._ser.write(frame)

    def recv(self) -> list[tuple[int, bytes]]:
        """Blocking read of whatever arrives within the timeout; data frames only."""
        try:
            raw = self._sock.recv(4096) if self._sock is not None else self._ser.read(4096)
        except socket.timeout:
            return []
        return [(p, d) for p, c, d in self.dec.feed(raw) if c == 0]

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
        if self._ser is not None:
            self._ser.close()


# -- AX.25 UI frames (so payloads interoperate with packet-radio networks) ------

@dataclass(frozen=True)
class AX25Frame:
    dst: str
    src: str
    path: tuple[str, ...]
    info: bytes
    pid: int = 0xF0


def _addr(call: str, last: bool, cmd_bit: int = 0) -> bytes:
    base, _, ssid = call.upper().partition("-")
    if not base or len(base) > 6 or not base.isalnum():
        raise ValueError(f"bad callsign '{call}'")
    s = int(ssid) if ssid else 0
    if not 0 <= s <= 15:
        raise ValueError("SSID must be 0..15")
    return bytes(ord(c) << 1 for c in base.ljust(6)) + bytes([(cmd_bit << 7) | 0x60 | (s << 1) | int(last)])


def ax25_encode(frame: AX25Frame) -> bytes:
    """AX.25 UI frame without flags/FCS (KISS TNCs add those)."""
    out = _addr(frame.dst, False, 1) + _addr(frame.src, not frame.path, 0)
    for i, digi in enumerate(frame.path):
        out += _addr(digi, i == len(frame.path) - 1)
    return out + bytes([0x03, frame.pid]) + frame.info


def ax25_decode(raw: bytes) -> AX25Frame:
    calls = []
    i = 0
    while True:
        if i + 7 > len(raw):
            raise ValueError("truncated AX.25 address field")
        chunk = raw[i:i + 7]
        base = "".join(chr(b >> 1) for b in chunk[:6]).strip()
        ssid = (chunk[6] >> 1) & 0x0F
        calls.append(f"{base}-{ssid}" if ssid else base)
        i += 7
        if chunk[6] & 1:
            break
    if len(calls) < 2 or i + 2 > len(raw) or raw[i] != 0x03:
        raise ValueError("not an AX.25 UI frame")
    return AX25Frame(calls[0], calls[1], tuple(calls[2:]), raw[i + 2:], raw[i + 1])
