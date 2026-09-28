"""Push-to-talk and rig control for sound-card-connected transceivers.

URIs accepted by :func:`open_ptt`::

    none | vox                       no keying (VOX or external)
    serial:/dev/ttyUSB0[:rts|dtr][:invert]   RTS/DTR line keying (pyserial)
    rigctld[:host[:port]]            Hamlib rigctld TCP (default localhost:4532),
                                     which also covers CAT and CM108 GPIO PTT
"""

from __future__ import annotations

import socket
import threading


class PTT:
    def key(self) -> None: ...
    def unkey(self) -> None: ...
    def close(self) -> None: ...
    def set_frequency(self, hz: float) -> None:
        raise NotImplementedError(f"{type(self).__name__} has no CAT control")

    def __enter__(self):
        self.key()
        return self

    def __exit__(self, *exc):
        self.unkey()


class NullPTT(PTT):
    pass


class SerialPTT(PTT):
    def __init__(self, port: str, line: str = "rts", invert: bool = False, baud: int = 9600):
        import serial  # pyserial
        if line not in ("rts", "dtr"):
            raise ValueError("line must be rts or dtr")
        self.line, self.invert = line, invert
        self.ser = serial.Serial(port, baud)
        self._set(False)

    def _set(self, on: bool) -> None:
        setattr(self.ser, self.line, on != self.invert)

    def key(self) -> None:
        self._set(True)

    def unkey(self) -> None:
        self._set(False)

    def close(self) -> None:
        self.unkey()
        self.ser.close()


class RigctldPTT(PTT):
    """Hamlib ``rigctld`` network protocol (``T 1`` / ``T 0`` / ``F <hz>``)."""

    def __init__(self, host: str = "localhost", port: int = 4532, timeout: float = 2.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self._f = self.sock.makefile("rwb", buffering=0)
        self._lock = threading.Lock()

    def _cmd(self, cmd: str) -> str:
        with self._lock:
            self._f.write((cmd + "\n").encode())
            reply = self._f.readline().decode().strip()
        if reply.startswith("RPRT") and reply != "RPRT 0":
            raise RuntimeError(f"rigctld '{cmd}' failed: {reply}")
        return reply

    def key(self) -> None:
        self._cmd("T 1")

    def unkey(self) -> None:
        self._cmd("T 0")

    def set_frequency(self, hz: float) -> None:
        self._cmd(f"F {int(round(hz))}")

    def close(self) -> None:
        try:
            self.unkey()
        finally:
            self.sock.close()


def open_ptt(uri: str | None) -> PTT:
    if not uri or uri in ("none", "vox"):
        return NullPTT()
    kind, _, rest = uri.partition(":")
    if kind == "serial":
        parts = rest.split(":")
        line = next((p for p in parts[1:] if p in ("rts", "dtr")), "rts")
        return SerialPTT(parts[0], line, invert="invert" in parts[1:])
    if kind == "rigctld":
        parts = [p for p in rest.split(":") if p]
        host = parts[0] if parts else "localhost"
        port = int(parts[1]) if len(parts) > 1 else 4532
        return RigctldPTT(host, port)
    raise ValueError(f"unknown PTT '{uri}'")
