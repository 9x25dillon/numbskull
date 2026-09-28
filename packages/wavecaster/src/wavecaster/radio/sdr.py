"""SDR front-ends: SoapySDR (USRP via SoapyUHD, HackRF, LimeSDR, PlutoSDR,
bladeRF, RTL-SDR RX-only, ...) and native UHD for Ettus USRPs.

Both keep a continuously running RX stream and send TX as bursts with
start/end-of-burst framing, so half-duplex TDD links work without the
transmitter idling on the air. Bursts are peak-normalised to
``tx_amplitude`` of DAC full scale; the receiver re-estimates gain per burst.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import RadioDevice, peak_normalize

log = logging.getLogger(__name__)


class SoapyDevice(RadioDevice):
    def __init__(self, args: str | dict = "", sample_rate: float = 1e6, center_freq: float = 915e6,
                 rx_gain: float | None = 30.0, tx_gain: float | None = 0.0, rx_antenna: str | None = None,
                 tx_antenna: str | None = None, bandwidth: float | None = None, channel: int = 0,
                 tx_amplitude: float = 0.7, rx_enabled: bool = True, tx_enabled: bool = True,
                 freq_correction_ppm: float | None = None):
        super().__init__()
        self.args = args
        self.sample_rate = float(sample_rate)
        self.center_freq = float(center_freq)
        self.rx_gain, self.tx_gain = rx_gain, tx_gain
        self.rx_antenna, self.tx_antenna = rx_antenna, tx_antenna
        self.bandwidth = bandwidth
        self.ch = channel
        self.tx_amplitude = tx_amplitude
        self.rx_enabled, self.tx_enabled = rx_enabled, tx_enabled
        self.ppm = freq_correction_ppm
        self.sdr = None
        self._rx = self._tx = None

    def open(self) -> "SoapyDevice":
        import SoapySDR
        from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX, SOAPY_SDR_TX
        self._S = SoapySDR
        self.sdr = SoapySDR.Device(self.args)
        dirs = []
        if self.rx_enabled:
            dirs.append((SOAPY_SDR_RX, self.rx_gain, self.rx_antenna))
        if self.tx_enabled:
            dirs.append((SOAPY_SDR_TX, self.tx_gain, self.tx_antenna))
        for d, gain, ant in dirs:
            self.sdr.setSampleRate(d, self.ch, self.sample_rate)
            self.sdr.setFrequency(d, self.ch, self.center_freq)
            if self.bandwidth:
                self.sdr.setBandwidth(d, self.ch, self.bandwidth)
            if gain is not None:
                self.sdr.setGain(d, self.ch, gain)
            if ant:
                self.sdr.setAntenna(d, self.ch, ant)
            if self.ppm is not None and hasattr(self.sdr, "setFrequencyCorrection"):
                self.sdr.setFrequencyCorrection(d, self.ch, self.ppm)
        if self.rx_enabled:
            self._rx = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [self.ch])
            self.sdr.activateStream(self._rx)
        if self.tx_enabled:
            self._tx = self.sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [self.ch])
            self.sdr.activateStream(self._tx)
            self._mtu = int(self.sdr.getStreamMTU(self._tx)) or 4096
        log.info("SoapySDR %s: %.0f S/s @ %.3f MHz", self.args, self.sample_rate, self.center_freq / 1e6)
        return super().open()

    def read(self, n: int, timeout: float = 1.0) -> np.ndarray:
        if self._rx is None:
            raise RuntimeError("RX not enabled/opened")
        S = self._S
        buf = np.empty(n, dtype=np.complex64)
        got = 0
        while got < n:
            sr = self.sdr.readStream(self._rx, [buf[got:]], n - got, timeoutUs=int(timeout * 1e6))
            if sr.ret > 0:
                got += sr.ret
            elif sr.ret == S.SOAPY_SDR_OVERFLOW:
                self.stats["overflows"] += 1
            elif sr.ret == S.SOAPY_SDR_TIMEOUT:
                break
            else:
                raise RuntimeError(f"readStream error {sr.ret} ({S.errToStr(sr.ret)})")
        self.stats["rx_samples"] += got
        return buf[:got].copy()

    def transmit(self, iq: np.ndarray) -> None:
        if self._tx is None:
            raise RuntimeError("TX not enabled/opened")
        S = self._S
        x = peak_normalize(np.asarray(iq), self.tx_amplitude)
        pos = 0
        while pos < len(x):
            n = min(self._mtu, len(x) - pos)
            flags = S.SOAPY_SDR_END_BURST if pos + n >= len(x) else 0
            sr = self.sdr.writeStream(self._tx, [x[pos:pos + n]], n, flags, timeoutUs=1_000_000)
            if sr.ret == S.SOAPY_SDR_UNDERFLOW:
                self.stats["underflows"] += 1
                continue
            if sr.ret < 0:
                raise RuntimeError(f"writeStream error {sr.ret} ({S.errToStr(sr.ret)})")
            pos += sr.ret
        self.stats["tx_samples"] += len(x)
        self.stats["bursts"] += 1

    def close(self) -> None:
        if self.sdr is not None:
            for st in (self._rx, self._tx):
                if st is not None:
                    self.sdr.deactivateStream(st)
                    self.sdr.closeStream(st)
            self._rx = self._tx = None
            self.sdr = None
        super().close()


class UHDDevice(RadioDevice):
    """Native UHD (``import uhd``) for Ettus/NI USRPs.

    ``tx_delay`` > 0 schedules each burst at ``device_time + tx_delay`` using
    timed commands, giving deterministic TX latency (useful for TDD slots).
    """

    def __init__(self, args: str = "", sample_rate: float = 1e6, center_freq: float = 915e6,
                 rx_gain: float = 30.0, tx_gain: float = 0.0, rx_antenna: str | None = "RX2",
                 tx_antenna: str | None = "TX/RX", bandwidth: float | None = None, channel: int = 0,
                 tx_amplitude: float = 0.7, tx_delay: float = 0.0, otw: str = "sc16"):
        super().__init__()
        self.args = args
        self.sample_rate, self.center_freq = float(sample_rate), float(center_freq)
        self.rx_gain, self.tx_gain = rx_gain, tx_gain
        self.rx_antenna, self.tx_antenna = rx_antenna, tx_antenna
        self.bandwidth, self.ch = bandwidth, channel
        self.tx_amplitude, self.tx_delay, self.otw = tx_amplitude, tx_delay, otw
        self.usrp = None

    def open(self) -> "UHDDevice":
        import uhd
        self._uhd = uhd
        u = self.usrp = uhd.usrp.MultiUSRP(self.args)
        ch = self.ch
        u.set_rx_rate(self.sample_rate, ch)
        u.set_tx_rate(self.sample_rate, ch)
        u.set_rx_freq(uhd.types.TuneRequest(self.center_freq), ch)
        u.set_tx_freq(uhd.types.TuneRequest(self.center_freq), ch)
        u.set_rx_gain(self.rx_gain, ch)
        u.set_tx_gain(self.tx_gain, ch)
        if self.rx_antenna:
            u.set_rx_antenna(self.rx_antenna, ch)
        if self.tx_antenna:
            u.set_tx_antenna(self.tx_antenna, ch)
        if self.bandwidth:
            u.set_rx_bandwidth(self.bandwidth, ch)
            u.set_tx_bandwidth(self.bandwidth, ch)
        actual = u.get_rx_rate(ch)
        if abs(actual - self.sample_rate) > 1e-6 * self.sample_rate:
            log.warning("USRP rate %.1f differs from requested %.1f", actual, self.sample_rate)
            self.sample_rate = actual
        st = uhd.usrp.StreamArgs("fc32", self.otw)
        st.channels = [ch]
        self._rx = u.get_rx_stream(st)
        self._tx = u.get_tx_stream(st)
        self._tx_max = int(self._tx.get_max_num_samps())
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd.stream_now = True
        self._rx.issue_stream_cmd(cmd)
        self._md = uhd.types.RXMetadata()
        return super().open()

    def read(self, n: int, timeout: float = 1.0) -> np.ndarray:
        uhd = self._uhd
        buf = np.zeros((1, n), dtype=np.complex64)
        got = int(self._rx.recv(buf, self._md, timeout))
        err = self._md.error_code
        if err == uhd.types.RXMetadataErrorCode.overflow:
            self.stats["overflows"] += 1
        elif err not in (uhd.types.RXMetadataErrorCode.none, uhd.types.RXMetadataErrorCode.timeout):
            raise RuntimeError(f"UHD RX error: {self._md.strerror()}")
        self.stats["rx_samples"] += got
        return buf[0, :got].copy()

    def transmit(self, iq: np.ndarray) -> None:
        uhd = self._uhd
        x = peak_normalize(np.asarray(iq), self.tx_amplitude)
        md = uhd.types.TXMetadata()
        md.start_of_burst = True
        md.end_of_burst = False
        if self.tx_delay > 0:
            md.has_time_spec = True
            md.time_spec = self.usrp.get_time_now() + uhd.types.TimeSpec(self.tx_delay)
        pos = 0
        while pos < len(x):
            n = min(self._tx_max, len(x) - pos)
            md.end_of_burst = pos + n >= len(x)
            sent = int(self._tx.send(x[pos:pos + n].reshape(1, -1), md, 1.0 + self.tx_delay))
            if sent == 0:
                self.stats["underflows"] += 1
            pos += sent
            md.start_of_burst = False
            md.has_time_spec = False
        self.stats["tx_samples"] += len(x)
        self.stats["bursts"] += 1

    def close(self) -> None:
        if self.usrp is not None:
            cmd = self._uhd.types.StreamCMD(self._uhd.types.StreamMode.stop_cont)
            self._rx.issue_stream_cmd(cmd)
            self.usrp = None
        super().close()


def enumerate_sdrs() -> list[dict]:
    found: list[dict] = []
    try:
        import SoapySDR
        found += [dict(backend="soapy", **{k: str(v[k]) for k in v.keys()}) for v in SoapySDR.Device.enumerate()]
    except ImportError:
        pass
    try:
        import uhd
        finder = getattr(uhd, "find", None)
        if finder:
            found += [dict(backend="uhd", args=str(a.to_string()) if hasattr(a, "to_string") else str(a))
                      for a in finder("")]
    except ImportError:
        pass
    return found
