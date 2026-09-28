import numpy as np
import pytest

from wavecaster.channel import Channel
from wavecaster.phy import PROFILES, PHYConfig, Receiver, Transmitter, get_profile

RNG = np.random.default_rng(99)


def run(cfg, esn0, msgs, cfo=0.0, chunk=4096, taps=(1.0,)):
    tx, rx = Transmitter(cfg), Receiver(cfg)
    sig = np.concatenate([tx.modulate(m, seq=i, pad=int(RNG.integers(10, 2000))) for i, m in enumerate(msgs)])
    ch = Channel(esn0, cfg.sps, cfo=cfo, phase=RNG.uniform(0, 6.28), gain=0.05, delay=int(RNG.integers(0, 3000)),
                 taps=taps, seed=int(RNG.integers(1 << 30)))
    y = np.concatenate([ch(sig), ch.noise(5000)])
    frames = []
    for i in range(0, len(y), chunk):
        frames += rx.feed(y[i:i + chunk])
    frames += rx.flush()
    return frames, rx


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_every_profile_links_with_impairments(name):
    cfg = get_profile(name)
    msgs = [RNG.bytes(int(RNG.integers(1, 200))) for _ in range(3)]
    frames, _ = run(cfg, 18, msgs, cfo=1e-4, taps=(1, 0, 0.15j))
    good = {f.seq: f.payload for f in frames if f.ok}
    assert [good.get(i) for i in range(3)] == msgs


def test_sync_estimates_snr_and_cfo():
    cfg = PHYConfig(modem="qpsk", fec="ldpc:512,0.5")
    frames, _ = run(cfg, 8.0, [b"x" * 100] * 4, cfo=2.5e-3)
    assert len(frames) == 4 and all(f.ok for f in frames)
    assert abs(np.mean([f.snr_db for f in frames]) - 8.0) < 1.0
    assert abs(np.mean([f.cfo_hz for f in frames]) / cfg.sample_rate - 2.5e-3) < 2e-5


def test_chunking_does_not_matter():
    cfg = PHYConfig(modem="16qam", fec="conv:128")
    msgs = [RNG.bytes(50) for _ in range(3)]
    for chunk in (97, 1000, 10 ** 7):
        frames, _ = run(cfg, 20, msgs, chunk=chunk)
        assert [f.payload for f in frames if f.ok] == msgs


def test_profile_mismatch_is_rejected():
    a = PHYConfig(modem="qpsk", fec="conv:128")
    b = PHYConfig(modem="qpsk", fec="ldpc:512,0.5")
    y = Channel(20, 8, seed=1)(Transmitter(a).modulate(b"hello", pad=500))
    rx = Receiver(b)
    assert rx.feed(y) + rx.flush() == []
    assert rx.stats["profile_mismatch"] == 1


def test_noise_only_produces_no_frames():
    cfg = PHYConfig()
    rx = Receiver(cfg)
    ch = Channel(0, cfg.sps, seed=5)
    assert rx.feed(ch.noise(300_000)) + rx.flush() == []


def test_robust_profile_below_zero_db():
    frames, _ = run(get_profile("sdr-robust"), -1.0, [b"low snr"] * 3, cfo=2e-3, chunk=8192)
    assert sum(f.ok for f in frames) >= 2


def test_empty_and_max_size_payloads():
    cfg = PHYConfig(modem="qpsk", fec="conv:256")
    big = RNG.bytes(3000)
    frames, _ = run(cfg, 25, [b"", big])
    assert [f.payload for f in frames if f.ok] == [b"", big]
