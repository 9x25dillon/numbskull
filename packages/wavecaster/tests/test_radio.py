import socket
import sys
import types

import numpy as np
import pytest

from wavecaster.link import Link
from wavecaster.phy import PHYConfig, Transmitter, Receiver, get_profile
from wavecaster.radio import (AX25Frame, KissDecoder, KissTNC, Passband, ax25_decode, ax25_encode, kiss_encode,
                              load_iq, open_device, save_iq)
from wavecaster.transport import MessageReassembler, Segmenter, StreamReassembler


def test_loopback_link_messages_and_streams():
    cfg = get_profile("sdr-narrow")
    dev = open_device("loopback:esn0=15,cfo=0.0003,seed=3", sample_rate=cfg.sample_rate, sps=cfg.sps)
    with Link(dev, cfg) as link:
        link.send(b"ping")
        assert next(link.frames(timeout=5)).payload == b"ping"
        blob = bytes(range(256)) * 20
        link.send_message(blob, mtu=900)
        assert link.receive_message(timeout=10) == blob
        chunks = [bytes([i]) * 100 for i in range(8)]
        link.stream_out(iter(chunks), mtu=200)
        assert b"".join(link.stream_in(timeout=10)) == b"".join(chunks)


@pytest.mark.parametrize("fmt", ["cf32", "cs16", "cu8", "cs8"])
def test_iq_file_formats_roundtrip(tmp_path, fmt):
    cfg = PHYConfig(modem="qpsk", fec="conv:128")
    path = tmp_path / f"burst.{fmt}"
    x = Transmitter(cfg).modulate(b"file io", pad=300)
    save_iq(path, 0.3 * x, fmt)
    y = load_iq(path, fmt)
    rx = Receiver(cfg)
    fr = rx.feed(y) + rx.flush()
    assert [f.payload for f in fr if f.ok] == [b"file io"]


def test_wav_passband_roundtrip(tmp_path):
    cfg = get_profile("audio-1200")
    wav = tmp_path / "tx.wav"
    dev = open_device(f"wav:tx={wav}", sample_rate=48000)
    tx = Transmitter(cfg)
    for i in range(2):
        dev.transmit(tx.modulate(f"audio {i}".encode(), seq=i))
    rxd = open_device(f"wav:rx={wav}", sample_rate=48000)
    rx = Receiver(cfg)
    frames = []
    for blk in rxd.rx_stream(4096):
        frames += rx.feed(blk)
    frames += rx.flush()
    assert [f.payload for f in frames if f.ok] == [b"audio 0", b"audio 1"]


def test_passband_is_phase_continuous_across_blocks():
    pb1, pb2 = Passband(48000, 1700), Passband(48000, 1700)
    x = np.exp(2j * np.pi * 0.001 * np.arange(4000))
    whole = pb1.up(x)
    parts = np.concatenate([pb2.up(x[:1234]), pb2.up(x[1234:])])
    assert np.allclose(whole, parts, atol=1e-6)


def test_kiss_escaping_and_ax25():
    data = bytes([0xC0, 0xDB, 1, 2, 0xC0])
    dec = KissDecoder()
    enc = kiss_encode(data, port=2)
    frames = dec.feed(enc[:3]) + dec.feed(enc[3:])
    assert frames == [(2, 0, data)]
    f = AX25Frame("APRS", "N0CALL-7", ("WIDE1-1",), b"hello")
    assert ax25_decode(ax25_encode(f)) == f


def test_kiss_tnc_over_socket():
    a, b = socket.socketpair()
    a.settimeout(1)
    tnc = KissTNC.from_socket(a)
    tnc.send(b"abc", port=1)
    assert KissDecoder().feed(b.recv(100)) == [(1, 0, b"abc")]
    b.sendall(kiss_encode(b"reply"))
    assert tnc.recv() == [(0, b"reply")]
    tnc.close()
    b.close()


def test_transport_reassembly_out_of_order_and_loss():
    seg = Segmenter(mtu=10)
    parts = seg.message(b"0123456789abcdefghij")
    r = MessageReassembler()
    assert [r.push(p) for p in reversed(parts)][-1] == b"0123456789abcdefghij"
    s = Segmenter(mtu=8)
    pieces = s.stream(b"AAAABBBBCCCCDDDD") + s.stream(b"", final=True)
    sr = StreamReassembler(window=2)
    out = b"".join(sr.push(p) for i, p in enumerate(pieces) if i != 1)  # lose segment 1
    assert out == b"AAAACCCCDDDD" and sr.lost == 1 and sr.ended


# -- hardware drivers against fake SDKs --------------------------------------------

class _SR:
    def __init__(self, ret):
        self.ret, self.flags, self.timeNs = ret, 0, 0


def _fake_soapy(loop):
    mod = types.ModuleType("SoapySDR")
    mod.SOAPY_SDR_RX, mod.SOAPY_SDR_TX, mod.SOAPY_SDR_CF32 = 1, 0, "CF32"
    mod.SOAPY_SDR_END_BURST, mod.SOAPY_SDR_TIMEOUT, mod.SOAPY_SDR_OVERFLOW, mod.SOAPY_SDR_UNDERFLOW = 2, -1, -4, -7
    mod.errToStr = str
    calls = []

    class Device:
        def __init__(self, args):
            calls.append(("open", args))

        def __getattr__(self, name):  # setSampleRate, setFrequency, ... recorded
            return lambda *a: calls.append((name, a))

        def setupStream(self, d, fmt, chans):
            return ("rx" if d == mod.SOAPY_SDR_RX else "tx")

        def getStreamMTU(self, st):
            return 1000

        def writeStream(self, st, buffs, n, flags=0, timeoutUs=0):
            loop.append((buffs[0][:n].copy(), flags))
            return _SR(n)

        def readStream(self, st, buffs, n, timeoutUs=0):
            buffs[0][:n] = 0
            return _SR(n)

    mod.Device = Device
    return mod, calls


def test_soapy_driver_bursts(monkeypatch):
    sent = []
    mod, calls = _fake_soapy(sent)
    monkeypatch.setitem(sys.modules, "SoapySDR", mod)
    dev = open_device("soapy:driver=uhd,type=b200", sample_rate=1e6, center_freq=433.92e6, tx_gain=10, rx_gain=20)
    with dev:
        names = [c[0] for c in calls]
        assert ("open", "driver=uhd,type=b200") in calls
        assert ("setFrequency", (1, 0, 433.92e6)) in calls and ("setGain", (0, 0, 10)) in calls
        assert "activateStream" in names
        burst = Transmitter(PHYConfig()).modulate(b"x" * 50)
        dev.transmit(burst)
        assert len(dev.read(500)) == 500
    total = np.concatenate([b for b, _ in sent])
    assert len(total) == len(burst)
    assert [f for _, f in sent].count(mod.SOAPY_SDR_END_BURST) == 1 and sent[-1][1] == mod.SOAPY_SDR_END_BURST
    assert abs(np.max(np.abs(total)) - 0.7) < 1e-5  # peak-normalised
    assert "closeStream" in [c[0] for c in calls]


def test_uhd_driver_bursts(monkeypatch):
    sent, log = [], []
    uhd = types.ModuleType("uhd")
    uhd.types = types.SimpleNamespace()
    uhd.usrp = types.SimpleNamespace()

    class Meta:
        def __init__(self):
            self.start_of_burst = self.end_of_burst = self.has_time_spec = False
            self.error_code = "none"

    class StreamCMD:
        def __init__(self, mode):
            self.mode, self.stream_now = mode, False

    class Streamer:
        def get_max_num_samps(self):
            return 700

        def send(self, buf, md, timeout):
            sent.append((buf.shape, md.start_of_burst, md.end_of_burst))
            return buf.shape[1]

        def recv(self, buf, md, timeout):
            md.error_code = "none"
            return buf.shape[1]

        def issue_stream_cmd(self, cmd):
            log.append(cmd.mode)

    class MultiUSRP:
        def __init__(self, args):
            log.append(("args", args))

        def __getattr__(self, name):
            if name.startswith("get_rx_rate"):
                return lambda ch: 1e6
            if name in ("get_rx_stream", "get_tx_stream"):
                return lambda st: Streamer()
            return lambda *a: log.append((name, a))

    uhd.usrp.MultiUSRP = MultiUSRP
    uhd.usrp.StreamArgs = lambda cpu, otw: types.SimpleNamespace(channels=None)
    uhd.types.TuneRequest = lambda f: ("tune", f)
    uhd.types.StreamCMD = StreamCMD
    uhd.types.StreamMode = types.SimpleNamespace(start_cont="start", stop_cont="stop")
    uhd.types.RXMetadata = uhd.types.TXMetadata = Meta
    uhd.types.RXMetadataErrorCode = types.SimpleNamespace(none="none", timeout="timeout", overflow="overflow")
    monkeypatch.setitem(sys.modules, "uhd", uhd)
    dev = open_device("uhd:type=b200", sample_rate=1e6, center_freq=2.4e9)
    with dev:
        dev.transmit(np.ones(2000, np.complex64))
        assert len(dev.read(100)) == 100
    assert ("args", "type=b200") in log and "start" in log and "stop" in log
    assert sent[0][1] is True and sent[-1][2] is True and sum(s[0][1] for s in sent) == 2000
    assert ("set_tx_freq", (("tune", 2.4e9), 0)) in log
