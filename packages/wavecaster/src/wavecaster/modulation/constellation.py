"""Generic constellations: bit labelling, mapping and soft demapping.

A :class:`Constellation` is fully described by ``M`` complex points and an
integer bit label per point, so any memoryless linear modulation (PSK, QAM,
APSK, PAM, cross-QAM, hand-designed geometric shaping, ...) is a data
definition, not code. Soft demapping returns per-bit LLRs
``ln P(b=0|y) / P(b=1|y)`` for AWGN with complex noise variance ``N0``,
either exact (log-sum-exp) or max-log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


def gray(i: np.ndarray | int):
    return np.bitwise_xor(i, np.right_shift(i, 1))


@dataclass(frozen=True)
class Constellation:
    points: np.ndarray                     # (M,) complex
    labels: np.ndarray                     # (M,) int, bit label of each point
    name: str = "custom"
    _lut: np.ndarray = field(init=False, repr=False, compare=False)
    _bitmask: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        pts = np.asarray(self.points, dtype=np.complex128)
        lab = np.asarray(self.labels, dtype=np.int64)
        M = len(pts)
        if M < 2 or M & (M - 1):
            raise ValueError("constellation size must be a power of two >= 2")
        if sorted(lab.tolist()) != list(range(M)):
            raise ValueError("labels must be a permutation of 0..M-1")
        lut = np.empty(M, dtype=np.int64)
        lut[lab] = np.arange(M)
        m = M.bit_length() - 1
        bitmask = ((lab[:, None] >> np.arange(m - 1, -1, -1)[None, :]) & 1).astype(bool)  # (M, m) MSB first
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "labels", lab)
        object.__setattr__(self, "_lut", lut)
        object.__setattr__(self, "_bitmask", bitmask)

    # -- properties -----------------------------------------------------------
    @property
    def M(self) -> int:
        return len(self.points)

    @property
    def bits_per_symbol(self) -> int:
        return self.M.bit_length() - 1

    @property
    def energy(self) -> float:
        return float(np.mean(np.abs(self.points) ** 2))

    def normalized(self) -> "Constellation":
        return Constellation(self.points / np.sqrt(self.energy), self.labels, self.name)

    def min_distance(self) -> float:
        d = np.abs(self.points[:, None] - self.points[None, :])
        return float(d[~np.eye(self.M, dtype=bool)].min())

    # -- mapping --------------------------------------------------------------
    def num_symbols(self, nbits: int) -> int:
        return -(-nbits // self.bits_per_symbol)

    def map(self, bits: np.ndarray) -> np.ndarray:
        m = self.bits_per_symbol
        bits = np.asarray(bits, dtype=np.int64)
        ns = self.num_symbols(len(bits))
        padded = np.zeros(ns * m, dtype=np.int64)
        padded[: len(bits)] = bits
        weights = 1 << np.arange(m - 1, -1, -1)
        idx = padded.reshape(ns, m) @ weights
        return self.points[self._lut[idx]]

    def demap(self, y: np.ndarray, noise_var: float | np.ndarray, exact: bool = True,
              chunk: int = 1 << 16) -> np.ndarray:
        """Per-bit LLRs for received symbols ``y`` (length N) -> (N*m,)."""
        y = np.asarray(y, dtype=np.complex128)
        nv = np.broadcast_to(np.maximum(np.asarray(noise_var, dtype=np.float64), 1e-12), y.shape)
        m = self.bits_per_symbol
        out = np.empty((len(y), m))
        step = max(1, chunk // self.M)
        for s in range(0, len(y), step):
            yy, vv = y[s:s + step], nv[s:s + step]
            metric = -np.abs(yy[:, None] - self.points[None, :]) ** 2 / vv[:, None]  # (n, M)
            for b in range(m):
                one = self._bitmask[:, b]
                if exact:
                    out[s:s + step, b] = _lse(metric[:, ~one]) - _lse(metric[:, one])
                else:
                    out[s:s + step, b] = metric[:, ~one].max(axis=1) - metric[:, one].max(axis=1)
        return out.ravel()

    def hard(self, y: np.ndarray) -> np.ndarray:
        idx = np.argmin(np.abs(np.asarray(y)[:, None] - self.points[None, :]), axis=1)
        m = self.bits_per_symbol
        return self._bitmask[idx].astype(np.uint8).reshape(-1)[: len(y) * m]

    # -- constructors ---------------------------------------------------------
    @classmethod
    def psk(cls, M: int, phase_offset: float | None = None, gray_coded: bool = True) -> "Constellation":
        if phase_offset is None:
            phase_offset = np.pi / 4 if M == 4 else 0.0
        k = np.arange(M)
        pts = np.exp(1j * (2 * np.pi * k / M + phase_offset))
        return cls(pts, gray(k) if gray_coded else k, f"{M}psk" if M > 4 else ("bpsk" if M == 2 else "qpsk"))

    @classmethod
    def pam(cls, M: int) -> "Constellation":
        a = np.arange(M)
        return cls((2 * a - M + 1).astype(complex), gray(a), f"{M}pam").normalized()

    @classmethod
    def qam(cls, M: int) -> "Constellation":
        m = M.bit_length() - 1
        if m % 2:
            raise ValueError("square QAM needs an even number of bits; use cross/custom points")
        L = 1 << (m // 2)
        a = np.arange(L)
        lev = 2 * a - L + 1
        I, Q = np.meshgrid(a, a, indexing="ij")
        pts = (lev[I] + 1j * lev[Q]).ravel()
        labels = ((gray(I) << (m // 2)) | gray(Q)).ravel()
        return cls(pts, labels, f"{M}qam").normalized()

    @classmethod
    def apsk(cls, rings: list[tuple[int, float, float]], name: str = "apsk") -> "Constellation":
        """``rings = [(n_points, radius, phase_offset), ...]``; ring-wise Gray labels."""
        pts, labels = [], []
        base = 0
        total = sum(r[0] for r in rings)
        if total & (total - 1):
            raise ValueError("total points must be a power of two")
        for n_pts, radius, phase in rings:
            k = np.arange(n_pts)
            pts.extend(radius * np.exp(1j * (2 * np.pi * k / n_pts + phase)))
            # Gray within power-of-two rings; sequential otherwise
            labels.extend(base + (gray(k) if n_pts & (n_pts - 1) == 0 else k))
            base += n_pts
        return cls(np.array(pts), np.array(labels), name).normalized()

    @classmethod
    def from_spec(cls, spec: dict | str) -> "Constellation":
        """Build from ``{"points": [[re, im], ...], "labels": [...], "name": ...}``
        or ``{"type": "psk"|"qam"|"pam"|"apsk", ...}`` (JSON text or dict)."""
        if isinstance(spec, str):
            spec = json.loads(spec)
        kind = spec.get("type")
        if kind == "psk":
            return cls.psk(int(spec["M"]), spec.get("phase_offset"))
        if kind == "qam":
            return cls.qam(int(spec["M"]))
        if kind == "pam":
            return cls.pam(int(spec["M"]))
        if kind == "apsk":
            return cls.apsk([tuple(r) for r in spec["rings"]], spec.get("name", "apsk"))
        pts = np.array([complex(p[0], p[1]) for p in spec["points"]])
        labels = np.array(spec.get("labels", range(len(pts))))
        c = cls(pts, labels, spec.get("name", "custom"))
        return c.normalized() if spec.get("normalize", True) else c

    def to_spec(self) -> dict:
        return {"name": self.name, "points": [[float(p.real), float(p.imag)] for p in self.points],
                "labels": [int(x) for x in self.labels], "normalize": False}


def _lse(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1)
    return m + np.log(np.exp(x - m[:, None]).sum(axis=1))


# DVB-S2-style ring geometries (ring ratios for code rate ~3/4). Labels are
# ring-major Gray, not the DVB-S2 bit mapping tables.
APSK16 = [(4, 1.0, np.pi / 4), (12, 2.85, np.pi / 12)]
APSK32 = [(4, 1.0, np.pi / 4), (12, 2.84, np.pi / 12), (16, 5.27, 0.0)]
