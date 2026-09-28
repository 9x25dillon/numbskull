"""Radio front-ends behind one complex-baseband contract.

``open_device(uri, **settings)`` URIs::

    loopback[:esn0=20,cfo=0.001,delay=0,seed=1]   simulated channel
    file:rx=in.cf32,tx=out.cf32[,fmt=cf32|cs16|cu8|cs8]
    wav:rx=in.wav,tx=out.wav[,center=1700]
    audio[:in=<dev>,out=<dev>,ptt=<ptt-uri>,center=1700,level=0.5]
    soapy:<SoapySDR args>        e.g. soapy:driver=uhd,type=b200 | soapy:driver=hackrf
    uhd:<UHD args>               e.g. uhd:type=b200 | uhd:addr=192.168.10.2

``settings`` (keyword args) are hardware parameters shared across SDRs:
``sample_rate, center_freq, rx_gain, tx_gain, rx_antenna, tx_antenna,
bandwidth, tx_amplitude``.
"""

from __future__ import annotations

from ..channel import Channel
from .audio import AudioDevice, Passband
from .base import RadioDevice
from .kiss import AX25Frame, KissDecoder, KissTNC, ax25_decode, ax25_encode, kiss_encode
from .ptt import PTT, NullPTT, RigctldPTT, SerialPTT, open_ptt
from .sdr import SoapyDevice, UHDDevice, enumerate_sdrs
from .sim import FileDevice, LoopbackDevice, WavDevice, load_iq, save_iq


def _kv(rest: str) -> dict:
    out = {}
    for item in filter(None, (s.strip() for s in rest.split(","))):
        k, _, v = item.partition("=")
        out[k.strip()] = v.strip()
    return out


def _num(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v


def open_device(uri: str, sample_rate: float = 1e6, center_freq: float = 915e6, **settings) -> RadioDevice:
    kind, _, rest = uri.partition(":")
    kind = kind.lower()
    if kind == "loopback":
        p = {k: _num(v) for k, v in _kv(rest).items()}
        ch = Channel(esn0_db=float(p.get("esn0", 30)), sps=int(p.get("sps", settings.get("sps", 8))),
                     cfo=float(p.get("cfo", 0.0)), phase=float(p.get("phase", 0.0)),
                     gain=float(p.get("gain", 1.0)), delay=int(p.get("delay", 0)),
                     seed=p.get("seed"))
        return LoopbackDevice(sample_rate, ch, realtime=bool(p.get("realtime", 0)))
    if kind == "file":
        p = _kv(rest)
        return FileDevice(p.get("rx"), p.get("tx"), sample_rate, p.get("fmt", "cf32"))
    if kind == "wav":
        p = _kv(rest)
        return WavDevice(p.get("rx"), p.get("tx"), sample_rate, float(p.get("center", 1700)),
                         float(p.get("level", 0.5)))
    if kind == "audio":
        # ptt URIs contain ':' so split on ',' only
        p = _kv(rest)
        return AudioDevice(sample_rate, float(p.get("center", 1700)), _num(p.get("in")), _num(p.get("out")),
                           p.get("ptt"), float(p.get("level", 0.5)))
    hw = {k: settings[k] for k in ("rx_gain", "tx_gain", "rx_antenna", "tx_antenna", "bandwidth", "tx_amplitude")
          if k in settings and settings[k] is not None}
    if kind == "soapy":
        return SoapyDevice(rest, sample_rate, center_freq, **hw)
    if kind == "uhd":
        return UHDDevice(rest, sample_rate, center_freq, **hw)
    raise ValueError(f"unknown device '{uri}'")


__all__ = [
    "RadioDevice", "LoopbackDevice", "FileDevice", "WavDevice", "AudioDevice", "SoapyDevice", "UHDDevice",
    "Passband", "PTT", "NullPTT", "SerialPTT", "RigctldPTT", "open_ptt", "KissTNC", "KissDecoder",
    "kiss_encode", "AX25Frame", "ax25_encode", "ax25_decode", "open_device", "enumerate_sdrs",
    "load_iq", "save_iq",
]
