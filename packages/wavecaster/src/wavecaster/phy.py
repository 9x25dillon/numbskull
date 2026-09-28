"""Burst PHY: framing, synchronisation and the streaming receiver.

Frame on air (complex baseband)::

    | SC segment: RRC-shaped BPSK            | payload segment (any Modem) |
    | preamble (m-seq, 127) | header (140)   | FEC(payload+CRC32), scrambled|

Header (8 bytes, K=7 r=1/2 convolutional coded, BPSK)::

    version(1) profile(1) seq(2) length(2) crc16(2)

``profile = crc8("<modem>|<fec>")`` rejects frames from a differently
configured transmitter before any payload decoding is attempted.

Receiver pipeline per detected burst::

    matched filter -> segmented noncoherent preamble correlation (CFO tolerant)
    -> 2-stage CFO estimate (lag 1, lag D) -> derotate -> complex gain + noise
    -> header Viterbi + CRC -> payload segment / gain -> Modem.demodulate
    -> descramble -> FEC decode -> CRC32

Every segment is normalised to unit average power per sample, so the gain
measured on the preamble applies to the payload regardless of waveform.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.signal import fftconvolve

from .bits import bits_to_bytes, bytes_to_bits, crc8, crc16_ccitt, crc32, descramble_llr, scramble
from .fec import BlockCodec, Convolutional, get_codec
from .modulation import Constellation, Modem, get_modem, m_sequence
from .modulation.pulse import matched_filter, rrc_taps, shape

log = logging.getLogger(__name__)

VERSION = 1
HEADER_BYTES = 8
MAX_PAYLOAD = 0xFFFF


@dataclass(frozen=True)
class PHYConfig:
    sample_rate: float = 1_000_000.0
    sps: int = 8
    rolloff: float = 0.35
    span: int = 8
    modem: str = "qpsk"
    fec: str = "ldpc:1024,0.5"
    pilot_interval: int = 32
    preamble_degree: int = 7          # m-sequence length 2^d - 1
    preamble_segments: int = 8        # noncoherent combining -> CFO tolerance
    detect_threshold: float = 0.45    # segmented (verification) metric, in [0, 1]
    diff_threshold: float | None = None   # differential (candidate) metric; None = auto from preamble length
    cfo_lag: int = 16
    header_repeat: int = 1            # repetition of the coded header (soft-combined)

    @property
    def symbol_rate(self) -> float:
        return self.sample_rate / self.sps

    def with_(self, **kw) -> "PHYConfig":
        return replace(self, **kw)


# Named operating points. Audio profiles assume a real passband conversion
# (see radio.audio) around ``AUDIO_CENTER_HZ``.
PROFILES: dict[str, PHYConfig] = {
    "audio-1200": PHYConfig(sample_rate=48_000, sps=40, modem="qpsk", fec="conv:256", pilot_interval=16),
    "audio-2400": PHYConfig(sample_rate=48_000, sps=20, modem="8psk", fec="ldpc:512,0.5", pilot_interval=16),
    "audio-afsk": PHYConfig(sample_rate=48_000, sps=40, modem="afsk1200", fec="conv:256"),
    "audio-ofdm": PHYConfig(sample_rate=48_000, sps=40, modem="ofdm:const=qpsk,nfft=1024,cp=256,used=48,pilots=4",
                            fec="ldpc:1024,0.5"),
    "sdr-narrow": PHYConfig(sample_rate=250_000, sps=8, modem="qpsk", fec="ldpc:1024,0.5"),
    "sdr-wide": PHYConfig(sample_rate=2_000_000, sps=4, modem="16qam", fec="turbo:1024,6,p"),
    "sdr-robust": PHYConfig(sample_rate=1_000_000, sps=8, modem="bpsk", fec="turbo:512,8", header_repeat=4,
                            preamble_degree=9, preamble_segments=32, detect_threshold=0.3),
    "sdr-dsss": PHYConfig(sample_rate=1_000_000, sps=8, modem="dsss:const=bpsk,degree=5", fec="conv:256"),
    "sdr-ofdm": PHYConfig(sample_rate=2_000_000, sps=4, modem="ofdm:const=16qam", fec="ldpc:1024,0.5"),
}


def get_profile(name: str, **overrides) -> PHYConfig:
    if name not in PROFILES:
        raise KeyError(f"unknown profile '{name}'; known: {sorted(PROFILES)}")
    return PROFILES[name].with_(**overrides) if overrides else PROFILES[name]


@dataclass
class Frame:
    payload: bytes
    seq: int
    ok: bool                 # CRC32 passed
    sample_index: int        # absolute index of burst start in the RX stream
    snr_db: float
    cfo_hz: float
    fec_iterations: int = 0
    info: dict = field(default_factory=dict)


class _Common:
    def __init__(self, cfg: PHYConfig):
        self.cfg = cfg
        phy = {"sps": cfg.sps, "rolloff": cfg.rolloff, "span": cfg.span, "sample_rate": cfg.sample_rate,
               "pilot_interval": cfg.pilot_interval}
        self.modem: Modem = get_modem(cfg.modem, phy)
        self.codec: BlockCodec = get_codec(cfg.fec)
        self.hcodec = Convolutional(k=HEADER_BYTES * 8)
        self.pre = m_sequence(cfg.preamble_degree)
        self.Lp = len(self.pre)
        self.Lh = self.hcodec.n * cfg.header_repeat
        self.ntaps = len(rrc_taps(cfg.sps, cfg.rolloff, cfg.span))
        self.sc_len = (self.Lp + self.Lh) * cfg.sps + self.ntaps - 1
        self.profile = crc8(f"{self.modem.name}|{self.codec.name}".encode())
        self.bpsk = Constellation.psk(2)

    def payload_coded_bits(self, nbytes: int) -> int:
        return self.codec.encoded_length((nbytes + 4) * 8)

    def payload_samples(self, nbytes: int) -> int:
        return self.modem.num_samples(self.payload_coded_bits(nbytes))

    def frame_samples(self, nbytes: int) -> int:
        return self.sc_len + self.payload_samples(nbytes)


class Transmitter(_Common):
    def build_header(self, seq: int, length: int) -> bytes:
        core = struct.pack(">BBHH", VERSION, self.profile, seq & 0xFFFF, length)
        return core + struct.pack(">H", crc16_ccitt(core))

    def modulate(self, payload: bytes, seq: int = 0, pad: int = 0) -> np.ndarray:
        """Payload bytes -> complex64 burst (unit average power)."""
        if len(payload) > MAX_PAYLOAD:
            raise ValueError(f"payload too long ({len(payload)} > {MAX_PAYLOAD}); use transport.Segmenter")
        cfg = self.cfg
        hdr_bits = np.tile(self.hcodec.encode(bytes_to_bits(self.build_header(seq, len(payload)))),
                           cfg.header_repeat)
        sc_syms = np.concatenate([self.pre, 1.0 - 2.0 * hdr_bits]).astype(np.complex128)
        sc = shape(sc_syms, cfg.sps, cfg.rolloff, cfg.span) * np.sqrt(cfg.sps)
        data = payload + struct.pack(">I", crc32(payload))
        coded = scramble(self.codec.encode(bytes_to_bits(data)))
        body = self.modem.modulate(coded)
        z = np.zeros(pad, dtype=np.complex128)
        return np.concatenate([z, sc, body, z]).astype(np.complex64)


class Receiver(_Common):
    """Streaming burst receiver. Push samples with :meth:`feed`; it returns the
    frames completed by those samples. Memory is bounded by one frame."""

    def __init__(self, cfg: PHYConfig, max_payload: int = 4096):
        super().__init__(cfg)
        self.max_payload = max_payload
        self._buf = np.zeros(0, dtype=np.complex128)
        self._base = 0          # absolute index of _buf[0]
        self._pending = None    # decoded header awaiting payload samples
        sps = cfg.sps
        span = (self.Lp - 1) * sps + 1
        segs = np.array_split(np.arange(self.Lp), cfg.preamble_segments)
        self._templates = []
        for seg in segs:
            t = np.zeros(span)
            t[seg * sps] = self.pre[seg]
            self._templates.append(t[::-1])
        ones = np.zeros(span)
        ones[np.arange(self.Lp) * sps] = 1.0
        self._ones = ones[::-1]
        self._corr_span = span
        # differential template: d[m] = z[m+sps] z*[m] turns CFO into a constant phase
        dspan = (self.Lp - 2) * sps + 1
        dt = np.zeros(dspan)
        dt[np.arange(self.Lp - 1) * sps] = self.pre[1:] * self.pre[:-1]
        self._dtemp = dt[::-1].copy()
        do = np.zeros(dspan)
        do[np.arange(self.Lp - 1) * sps] = 1.0
        self._dones = do[::-1].copy()
        # noise-only metric ~ 1.13/sqrt(L); 4/sqrt(L) keeps candidate rate low
        self._thr_d = cfg.diff_threshold if cfg.diff_threshold is not None else min(0.45, 4.0 / np.sqrt(self.Lp - 1))
        self.stats = {"detections": 0, "header_fail": 0, "profile_mismatch": 0, "frames": 0, "crc_fail": 0,
                      "no_signal": 0, "rejected": 0}

    # -- public API -------------------------------------------------------------
    def feed(self, samples: np.ndarray) -> list[Frame]:
        self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.complex128)])
        out: list[Frame] = []
        while True:
            if self._pending is not None:
                fr = self._try_payload()
                if fr is None:
                    break
                out.append(fr)
                continue
            if not self._search():
                break
        return out

    def flush(self) -> list[Frame]:
        """End of stream: zero-pad just enough to finish a pending frame and
        to let the search reach the end of the buffer."""
        out: list[Frame] = []
        for _ in range(4):
            if self._pending is not None:
                missing = self.sc_len + self.payload_samples(self._pending["length"]) - len(self._buf)
            else:
                missing = self.sc_len + self.ntaps
            out += self.feed(np.zeros(max(missing, 0) + self.ntaps))
            if self._pending is None:
                break
        return out

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.complex128)
        self._pending = None

    # -- internals ----------------------------------------------------------------
    def _consume(self, n: int) -> None:
        n = max(0, min(n, len(self._buf)))
        self._buf = self._buf[n:]
        self._base += n

    def _metric(self, z: np.ndarray) -> np.ndarray:
        num = np.zeros(len(z) - self._corr_span + 1)
        for t in self._templates:
            num += np.abs(fftconvolve(z, t, mode="valid"))
        energy = fftconvolve(np.abs(z) ** 2, self._ones, mode="valid").real
        # absolute floor (1e-20 * Lp) keeps digitally silent input (exact zeros in
        # files, muted sound cards) from producing 0/0 "detections"
        return num / np.sqrt(self.Lp * np.maximum(energy, 1e-20 * self.Lp))

    def _diff_metric(self, z: np.ndarray) -> np.ndarray:
        """Differential preamble correlation, bounded in [0, 1] (triangle
        inequality). Insensitive to CFO up to +-Rs/2, one FFT correlation."""
        sps = self.cfg.sps
        d = z[sps:] * np.conj(z[:-sps])
        num = np.abs(fftconvolve(d, self._dtemp, mode="valid"))
        den = fftconvolve(np.abs(d), self._dones, mode="valid").real
        return num / np.maximum(den, 1e-20 * self.Lp)

    def _search(self) -> bool:
        """Two-stage acquisition.

        1. candidates: differential metric over every sample (2 FFT correlations);
        2. verification: segmented noncoherent metric only around the
           strongest candidate within one preamble length (cost independent
           of buffer size), thresholded with ``detect_threshold``.
        """
        cfg = self.cfg
        sps = cfg.sps
        need = self.sc_len + self.ntaps
        if len(self._buf) < need:
            return False
        z = matched_filter(self._buf, sps, cfg.rolloff, cfg.span)[: len(self._buf)]
        rd = self._diff_metric(z)
        # n0 is the MF index of preamble symbol 0; burst start d = n0 - (ntaps-1)
        first = self.ntaps - 1
        cand = np.nonzero(rd[first:] > self._thr_d)[0]
        if len(cand) == 0:
            self._consume(len(self._buf) - need)
            return False
        c0 = first + cand[0]
        # strongest candidate within one preamble length of the first crossing,
        # so a noise spike just before a real preamble cannot win
        if c0 + self.Lp * sps > len(rd) and len(self._buf) < self.sc_len * 4:
            self._consume(max(0, c0 - self.ntaps))
            return False
        m = c0 + int(np.argmax(rd[c0:c0 + self.Lp * sps]))
        lo = max(first, m - 2 * sps)
        hi = m + 2 * sps + 1
        if hi + self.sc_len + self.ntaps > len(self._buf):
            self._consume(max(0, lo - self.ntaps))
            return False  # wait for the full header
        rho_loc = self._metric(z[lo:hi - 1 + self._corr_span])
        n0 = lo + int(np.argmax(rho_loc))
        rho = float(rho_loc.max())
        if rho < cfg.detect_threshold:
            self.stats["rejected"] += 1
            self._consume(m + sps - first)
            return True
        d = n0 - first
        self.stats["detections"] += 1
        hdr = self._decode_header(z, n0, d, rho)
        if hdr is None:
            self._consume(d + sps)  # skip this false alarm
            return True
        self._pending = hdr
        self._consume(d)
        self._pending["d"] = 0
        return True

    def _estimate_cfo(self, w: np.ndarray) -> float:
        """CFO in cycles/sample from symbol-spaced, modulation-stripped samples.

        Lag cascade 1, 4, 16, ...: each stage removes the previous estimate and
        measures the residual with a 4x longer lag, keeping the residual well
        inside the next stage's +-1/(2*lag*sps) ambiguity while its variance
        shrinks ~lag^-2 (approaches the Cramer-Rao bound for long sequences).
        """
        sps = self.cfg.sps
        f = 0.0
        k = np.arange(len(w))
        lag = 1
        while lag <= len(w) // 2:
            ww = w * np.exp(-2j * np.pi * f * sps * k)
            f += np.angle(np.sum(ww[lag:] * np.conj(ww[:-lag]))) / (2 * np.pi * sps * lag)
            lag *= 4
        return f

    def _channel(self, s: np.ndarray, known: np.ndarray, f_res: float):
        """Gain/noise from known symbols after removing a residual CFO ramp."""
        t = np.arange(len(s)) * self.cfg.sps + (self.ntaps - 1) / 2
        s = s * np.exp(-2j * np.pi * f_res * t)
        h = np.mean(s * known)
        nv_abs = float(np.mean(np.abs(s - h * known) ** 2))
        return s, h, nv_abs

    def _decode_header(self, z, n0, d, rho) -> dict | None:
        cfg = self.cfg
        k = np.arange(self.Lp) * cfg.sps
        w = z[n0 + k] * self.pre
        f = self._estimate_cfo(w)
        seg = self._buf[d:d + self.sc_len + self.ntaps]
        y = seg * np.exp(-2j * np.pi * f * np.arange(len(seg)))
        zc = matched_filter(y, cfg.sps, cfg.rolloff, cfg.span)
        s = zc[np.arange(self.Lp + self.Lh) * cfg.sps + self.ntaps - 1]
        _, h, nv_abs = self._channel(s[: self.Lp], self.pre, 0.0)
        if abs(h) < 1e-12:
            self.stats["no_signal"] += 1
            return None
        nv = max(nv_abs / abs(h) ** 2, 1e-6)
        hs = s[self.Lp:] / h
        llr = self.bpsk.demap(hs, nv).reshape(cfg.header_repeat, self.hcodec.n).sum(axis=0)
        hb = self.hcodec.decode(llr, HEADER_BYTES * 8).bits
        raw = bits_to_bytes(hb)
        ver, prof, seq, length = struct.unpack(">BBHH", raw[:6])
        (crc,) = struct.unpack(">H", raw[6:8])
        if crc != crc16_ccitt(raw[:6]) or ver != VERSION:
            self.stats["header_fail"] += 1
            return None
        if prof != self.profile:
            self.stats["profile_mismatch"] += 1
            log.debug("profile mismatch: got %02x expected %02x", prof, self.profile)
            return None
        if length > self.max_payload:
            self.stats["header_fail"] += 1
            return None
        # data-aided refinement over preamble + re-encoded header
        coded = np.tile(self.hcodec.encode(hb), cfg.header_repeat)
        known = np.concatenate([self.pre, 1.0 - 2.0 * coded])
        f_res = self._estimate_cfo(s * known)
        _, h, nv_abs = self._channel(s, known, f_res)
        f += f_res
        nv = max(nv_abs / abs(h) ** 2, 1e-6)
        return {"seq": seq, "length": length, "f": f, "h": h, "nv": nv, "abs": self._base + d, "rho": rho}

    def _try_payload(self) -> Frame | None:
        p = self._pending
        n_coded = self.payload_coded_bits(p["length"])
        n_samp = self.modem.num_samples(n_coded)
        end = self.sc_len + n_samp
        if len(self._buf) < end:
            return None
        idx = np.arange(self.sc_len, end)
        seg = self._buf[self.sc_len:end] * np.exp(-2j * np.pi * p["f"] * idx) / p["h"]
        llr = descramble_llr(self.modem.demodulate(seg, p["nv"], n_coded))
        nbits = (p["length"] + 4) * 8
        res = self.codec.decode(llr, nbits)
        data = bits_to_bytes(res.bits)
        payload, (rx_crc,) = data[:-4], struct.unpack(">I", data[-4:])
        ok = rx_crc == crc32(payload)
        self.stats["frames" if ok else "crc_fail"] += 1
        fr = Frame(payload=payload, seq=p["seq"], ok=ok, sample_index=p["abs"],
                   snr_db=float(10 * np.log10(1 / p["nv"])), cfo_hz=float(p["f"] * self.cfg.sample_rate),
                   fec_iterations=res.iterations,
                   info={"fec_ok": res.ok, "rho": p["rho"], "failed_blocks": res.failed_blocks})
        self._pending = None
        # the header CRC already validated this burst, so skip it whole either way
        self._consume(end)
        return fr
