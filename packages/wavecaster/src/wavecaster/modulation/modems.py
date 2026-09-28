"""Waveform modems: bits <-> complex baseband samples.

Contract (see :class:`Modem`):

* ``modulate(bits)`` returns a self-contained complex baseband *segment*
  with unit average power per sample (so segments can be concatenated
  without level jumps).
* ``num_samples(nbits)`` is the exact segment length for ``nbits``.
* ``demodulate(seg, noise_var)`` receives the segment already aligned to its
  first sample, with coarse CFO removed and gain/phase normalised by the PHY
  (so a transmitted unit-energy symbol arrives with unit gain). It returns
  per-bit LLRs (``ln P(0)/P(1)``) for ``nbits`` bits.

``noise_var`` is the complex noise variance per sample after the PHY's gain
normalisation; for a single-carrier modem at the PHY's ``sps`` this equals
the matched-filter noise variance per unit-energy symbol, ``1/(Es/N0)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .constellation import Constellation, gray
from .pulse import matched_filter, rrc_taps, shape


class Modem(ABC):
    name: str = "abstract"

    @property
    @abstractmethod
    def bits_per_symbol(self) -> int: ...

    @abstractmethod
    def num_samples(self, nbits: int) -> int: ...

    @abstractmethod
    def modulate(self, bits: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def demodulate(self, seg: np.ndarray, noise_var: float, nbits: int) -> np.ndarray: ...

    def describe(self) -> dict:
        return {"name": self.name, "bits_per_symbol": self.bits_per_symbol}


def m_sequence(degree: int = 5, taps: tuple[int, ...] | None = None) -> np.ndarray:
    """Maximal-length +-1 sequence of length 2^degree - 1 (Fibonacci LFSR)."""
    default_taps = {3: (3, 2), 4: (4, 3), 5: (5, 3), 6: (6, 5), 7: (7, 6), 8: (8, 6, 5, 4), 9: (9, 5), 10: (10, 7)}
    taps = taps or default_taps[degree]
    state = [1] * degree
    out = []
    for _ in range((1 << degree) - 1):
        out.append(state[-1])
        fb = 0
        for t in taps:
            fb ^= state[t - 1]
        state = [fb] + state[:-1]
    return 1.0 - 2.0 * np.array(out, dtype=np.float64)


class SingleCarrierModem(Modem):
    """Linear modulation with RRC pulses, pilot-aided phase tracking and
    optional direct-sequence spreading.

    Every ``pilot_interval`` data symbols a known pilot ``(1+0j)`` is
    inserted; the receiver interpolates the complex gain between pilots,
    which removes residual CFO and slow phase noise left after the PHY's
    preamble-based correction.
    """

    def __init__(self, constellation: Constellation, sps: int = 8, rolloff: float = 0.35,
                 span: int = 8, pilot_interval: int = 32, spreading: np.ndarray | None = None,
                 name: str | None = None, pilot_window: int = 9):
        self.const = constellation.normalized()
        self.sps, self.rolloff, self.span = sps, rolloff, span
        self.pilot_interval = pilot_interval
        self.pilot_window = pilot_window
        self.pn = None if spreading is None else np.asarray(spreading, dtype=np.float64)
        self.name = name or self.const.name + ("-dsss" if self.pn is not None else "")
        self._ntaps = len(rrc_taps(sps, rolloff, span))

    @property
    def bits_per_symbol(self) -> int:
        return self.const.bits_per_symbol

    @property
    def chips(self) -> int:
        return 1 if self.pn is None else len(self.pn)

    def _layout(self, nsym: int) -> tuple[int, np.ndarray]:
        """Total symbol count and indices of data symbols in the stream."""
        P = self.pilot_interval
        if not P:
            return nsym, np.arange(nsym)
        n_pil = nsym // P + 1  # pilot at start and after every P data symbols
        total = nsym + n_pil
        is_pilot = np.zeros(total, dtype=bool)
        is_pilot[np.arange(n_pil) * (P + 1)] = True
        return total, np.nonzero(~is_pilot)[0]

    def num_samples(self, nbits: int) -> int:
        total, _ = self._layout(self.const.num_symbols(nbits))
        return total * self.chips * self.sps + self._ntaps - 1

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        data = self.const.map(bits)
        total, didx = self._layout(len(data))
        stream = np.ones(total, dtype=np.complex128)  # pilots = 1
        stream[didx] = data
        if self.pn is not None:
            stream = (stream[:, None] * self.pn[None, :]).ravel()
        return shape(stream, self.sps, self.rolloff, self.span) * np.sqrt(self.sps)

    def demodulate(self, seg: np.ndarray, noise_var: float, nbits: int) -> np.ndarray:
        nsym = self.const.num_symbols(nbits)
        total, didx = self._layout(nsym)
        nchip = total * self.chips
        z = matched_filter(seg[: self.num_samples(nbits)], self.sps, self.rolloff, self.span)
        idx = np.arange(nchip) * self.sps + self._ntaps - 1
        idx = idx[idx < len(z)]
        chips = np.zeros(nchip, dtype=np.complex128)
        chips[: len(idx)] = z[idx]
        nv = noise_var
        if self.pn is not None:
            L = len(self.pn)
            # despreading averages L chips: signal unchanged, noise variance / L
            sym = chips.reshape(total, L) @ self.pn / L
            nv = noise_var / L
        else:
            sym = chips
        if self.pilot_interval:
            pil = np.setdiff1d(np.arange(total), didx)
            gain, amp2 = self._track(sym[pil], pil, total)
            sym = sym / gain
            data = sym[didx]
            nv = nv / amp2[didx]
        else:
            data = sym[didx]
        llr = self.const.demap(data, nv)
        return llr[:nbits]

    def _track(self, g: np.ndarray, pil: np.ndarray, total: int) -> tuple[np.ndarray, np.ndarray]:
        """Complex gain per symbol from pilots.

        1. residual CFO from the lag-1 pilot autocorrelation (averaged over all
           pilots), removed as a linear phase ramp;
        2. remaining phase/amplitude smoothed with a sliding window of
           ``pilot_window`` pilots, then linearly interpolated.
        Averaging W pilots reduces estimation noise by W, so the SNR loss is
        ~10*log10(1 + 1/W) dB instead of 3 dB for single-pilot estimates.
        """
        t = np.arange(total)
        if len(g) > 1:
            w = np.angle(np.sum(g[1:] * np.conj(g[:-1]))) / (pil[1] - pil[0])
        else:
            w = 0.0
        g0 = g * np.exp(-1j * w * pil)
        W = min(self.pilot_window, len(g0))
        if W > 1:
            ker = np.ones(W)
            num = np.convolve(g0, ker, mode="same")
            den = np.convolve(np.ones(len(g0)), ker, mode="same")
            g0 = num / den
        amp = np.maximum(np.interp(t, pil, np.abs(g0)), 1e-9)
        ph = np.interp(t, pil, np.unwrap(np.angle(g0))) + w * t
        return amp * np.exp(1j * ph), amp ** 2

    def describe(self) -> dict:
        d = super().describe()
        d.update(sps=self.sps, rolloff=self.rolloff, pilot_interval=self.pilot_interval,
                 spreading=None if self.pn is None else len(self.pn), M=self.const.M)
        return d


class OFDMModem(Modem):
    """CP-OFDM with one training symbol for per-subcarrier channel estimation
    and continual pilots for common-phase-error tracking.

    Subcarrier spacing is ``fs/nfft``; ``n_used`` carriers are placed
    symmetrically around (and excluding) DC.
    """

    def __init__(self, constellation: Constellation, nfft: int = 64, cp: int = 16, n_used: int = 48,
                 n_pilots: int = 4, name: str | None = None, smooth: int = 3):
        if n_used % 2 or n_used >= nfft:
            raise ValueError("n_used must be even and < nfft")
        self.const = constellation.normalized()
        self.nfft, self.cp = nfft, cp
        self.smooth = smooth
        half = n_used // 2
        used = np.concatenate([np.arange(-half, 0), np.arange(1, half + 1)])
        self.used = np.mod(used, nfft)
        pil_pos = np.linspace(0, n_used - 1, n_pilots + 2)[1:-1].round().astype(int) if n_pilots else np.array([], int)
        self.pilots = self.used[pil_pos]
        self.data = np.setdiff1d(self.used, self.pilots, assume_unique=False)
        self.data = self.used[np.isin(self.used, self.data)]
        rng = np.random.default_rng(0x0FD)
        self.train = 1.0 - 2.0 * rng.integers(0, 2, len(self.used))
        self.pilot_vals = 1.0 - 2.0 * rng.integers(0, 2, len(self.pilots))
        self.name = name or f"ofdm-{self.const.name}"
        self._scale = nfft / np.sqrt(n_used)  # unit average sample power

    @property
    def bits_per_symbol(self) -> int:
        return self.const.bits_per_symbol

    def _n_ofdm(self, nbits: int) -> int:
        per = len(self.data) * self.bits_per_symbol
        return max(1, -(-nbits // per))

    def num_samples(self, nbits: int) -> int:
        return (self._n_ofdm(nbits) + 1) * (self.nfft + self.cp)

    def _symbol(self, grid: np.ndarray) -> np.ndarray:
        t = np.fft.ifft(grid) * self._scale
        return np.concatenate([t[-self.cp:], t])

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        nsym = self._n_ofdm(len(bits))
        cap = nsym * len(self.data) * self.bits_per_symbol
        padded = np.zeros(cap, dtype=np.uint8)
        padded[: len(bits)] = bits
        syms = self.const.map(padded).reshape(nsym, len(self.data))
        out = []
        g = np.zeros(self.nfft, dtype=np.complex128)
        g[self.used] = self.train
        out.append(self._symbol(g))
        for s in syms:
            g = np.zeros(self.nfft, dtype=np.complex128)
            g[self.data] = s
            g[self.pilots] = self.pilot_vals
            out.append(self._symbol(g))
        return np.concatenate(out)

    def demodulate(self, seg: np.ndarray, noise_var: float, nbits: int) -> np.ndarray:
        nsym = self._n_ofdm(nbits)
        L = self.nfft + self.cp
        need = (nsym + 1) * L
        x = np.zeros(need, dtype=np.complex128)
        x[: min(need, len(seg))] = seg[:need]
        blocks = x.reshape(nsym + 1, L)
        # sample a quarter CP early: tolerant to timing error in both directions
        off = self.cp - self.cp // 4
        F = np.fft.fft(blocks[:, off:off + self.nfft], axis=1) / self._scale
        # undo the linear phase introduced by the early FFT window
        k = np.fft.fftfreq(self.nfft) * self.nfft
        F = F * np.exp(2j * np.pi * k * (self.cp - off) / self.nfft)[None, :]
        H = np.ones(self.nfft, dtype=np.complex128)
        Hu = F[0, self.used] / self.train
        if self.smooth > 1:
            # moving average over adjacent used carriers (valid while the
            # coherence bandwidth exceeds `smooth` carriers)
            ker = np.ones(self.smooth)
            Hu = np.convolve(Hu, ker, "same") / np.convolve(np.ones(len(Hu)), ker, "same")
        H[self.used] = Hu
        Y = F[1:]
        Heq = H[None, :]
        if len(self.pilots):
            cpe = np.sum(Y[:, self.pilots] * np.conj(Heq[:, self.pilots] * self.pilot_vals), axis=1)
            Heq = Heq * np.exp(1j * np.angle(cpe))[:, None]
        Yd = Y[:, self.data] / Heq[:, self.data]
        # per-carrier noise: time-domain variance nv maps to nv*n_used/nfft after
        # the scaled FFT; dividing by |H|^2 accounts for frequency selectivity.
        nv_carrier = noise_var * len(self.used) / self.nfft
        nv = nv_carrier / np.maximum(np.abs(Heq[:, self.data]) ** 2, 1e-9)
        llr = self.const.demap(Yd.ravel(), nv.ravel())
        return llr[:nbits]

    def describe(self) -> dict:
        d = super().describe()
        d.update(nfft=self.nfft, cp=self.cp, n_used=len(self.used), n_pilots=len(self.pilots))
        return d


class FSKModem(Modem):
    """Continuous-phase M-FSK with noncoherent detection.

    Tones sit at ``(m - (M-1)/2) * spacing`` Hz around the centre frequency;
    ``M=2, spacing=1000, fs=48000, sps=40`` is Bell-202-like 1200 Bd AFSK
    once the PHY upconverts to an audio carrier of 1700 Hz.
    LLRs use the exact noncoherent metric ``ln I0(2 A |c_m| / sigma^2)``.
    """

    def __init__(self, M: int = 2, sps: int = 40, spacing: float = 1000.0, sample_rate: float = 48000.0,
                 name: str | None = None):
        if M < 2 or M & (M - 1):
            raise ValueError("M must be a power of two")
        self.M, self.sps, self.spacing, self.fs = M, sps, spacing, sample_rate
        self.freqs = (np.arange(M) - (M - 1) / 2) * spacing
        self.labels = gray(np.arange(M))
        self._lut = np.empty(M, dtype=int)
        self._lut[self.labels] = np.arange(M)
        self.name = name or f"{M}fsk"

    @property
    def bits_per_symbol(self) -> int:
        return self.M.bit_length() - 1

    def num_samples(self, nbits: int) -> int:
        return -(-nbits // self.bits_per_symbol) * self.sps

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        m = self.bits_per_symbol
        ns = -(-len(bits) // m)
        b = np.zeros(ns * m, dtype=np.int64)
        b[: len(bits)] = bits
        idx = self._lut[b.reshape(ns, m) @ (1 << np.arange(m - 1, -1, -1))]
        f = np.repeat(self.freqs[idx], self.sps)
        phase = 2 * np.pi * np.cumsum(f) / self.fs
        return np.exp(1j * phase)

    def demodulate(self, seg: np.ndarray, noise_var: float, nbits: int) -> np.ndarray:
        from scipy.special import i0e
        m = self.bits_per_symbol
        ns = -(-nbits // m)
        x = np.zeros(ns * self.sps, dtype=np.complex128)
        x[: min(len(x), len(seg))] = seg[: len(x)]
        blocks = x.reshape(ns, self.sps)
        n = np.arange(self.sps)
        tones = np.exp(-2j * np.pi * self.freqs[:, None] * n[None, :] / self.fs)  # (M, sps)
        c = blocks @ tones.T / self.sps  # (ns, M) ~ unit amplitude for the sent tone
        mag = np.abs(c)
        # averaging sps samples of per-sample variance nv gives nv/sps
        sigma2 = max(noise_var, 1e-6) / self.sps
        A = np.median(mag.max(axis=1))
        x_arg = 2 * A * mag / sigma2
        metric = np.log(i0e(x_arg)) + x_arg  # ln I0
        bm = ((self.labels[:, None] >> np.arange(m - 1, -1, -1)[None, :]) & 1).astype(bool)  # (M, m)
        out = np.empty((ns, m))
        for b in range(m):
            one = bm[:, b]
            out[:, b] = _lse(metric[:, ~one]) - _lse(metric[:, one])
        return out.ravel()[:nbits]

    def describe(self) -> dict:
        d = super().describe()
        d.update(sps=self.sps, spacing=self.spacing, M=self.M)
        return d


def _lse(x: np.ndarray) -> np.ndarray:
    mx = x.max(axis=1)
    return mx + np.log(np.exp(x - mx[:, None]).sum(axis=1))
