import json

import numpy as np
import pytest

from wavecaster.modulation import (Constellation, Modem, SingleCarrierModem, available_modems, get_constellation,
                                   get_modem, register_constellation, register_modem)

RNG = np.random.default_rng(7)


@pytest.mark.parametrize("name", ["bpsk", "qpsk", "8psk", "16psk", "4pam", "16qam", "64qam", "256qam",
                                  "16apsk", "32apsk"])
def test_constellations_are_valid(name):
    c = get_constellation(name)
    assert abs(c.energy - 1) < 1e-9
    bits = RNG.integers(0, 2, 60 * c.bits_per_symbol).astype(np.uint8)
    assert np.array_equal(c.hard(c.map(bits)), bits)
    llr = c.demap(c.map(bits), 1e-3)
    assert np.array_equal((llr < 0).astype(np.uint8), bits)


def test_gray_labelling_neighbours_differ_by_one_bit():
    for c in (Constellation.qam(16), Constellation.psk(8)):
        for i, p in enumerate(c.points):
            d = np.abs(c.points - p)
            d[i] = np.inf
            j = int(np.argmin(d))
            assert bin(int(c.labels[i] ^ c.labels[j])).count("1") == 1


def test_exact_and_maxlog_llr_agree_in_sign():
    c = Constellation.qam(16)
    y = c.map(RNG.integers(0, 2, 400)) + 0.2 * (RNG.standard_normal(100) + 1j * RNG.standard_normal(100))
    a, b = c.demap(y, 0.08, exact=True), c.demap(y, 0.08, exact=False)
    assert np.mean(np.sign(a) == np.sign(b)) > 0.98


def test_custom_constellation_from_json(tmp_path):
    # hexagonal-ish 8-point constellation defined purely as data
    pts = [[1, 0], [0.5, 0.87], [-0.5, 0.87], [-1, 0], [-0.5, -0.87], [0.5, -0.87], [0, 0], [2, 0]]
    spec = {"name": "hex8", "points": pts, "labels": [0, 1, 3, 2, 6, 7, 5, 4]}
    f = tmp_path / "hex8.json"
    f.write_text(json.dumps(spec))
    m = get_modem(f"custom:file={f}")
    bits = RNG.integers(0, 2, 300).astype(np.uint8)
    x = m.modulate(bits)
    llr = m.demodulate(x / np.sqrt(8), 1e-3, len(bits))
    assert np.array_equal((llr < 0).astype(np.uint8), bits)
    assert Constellation.from_spec(m.const.to_spec()).M == 8


def test_register_constellation_extends_all_waveforms():
    register_constellation("star8", lambda: Constellation.apsk([(4, 1.0, 0.0), (4, 2.5, np.pi / 4)], "star8"))
    assert "star8" in available_modems()
    for spec in ("star8", "ofdm:const=star8", "dsss:const=star8,degree=3"):
        m = get_modem(spec)
        bits = RNG.integers(0, 2, 240).astype(np.uint8)
        y = m.modulate(bits) / (1 if spec.startswith("ofdm") else np.sqrt(8))
        assert np.array_equal((m.demodulate(y, 1e-3, len(bits)) < 0).astype(np.uint8), bits)


def test_register_new_waveform():
    class OOK(Modem):
        name = "ook"
        bits_per_symbol = 1

        def __init__(self, sps):
            self.sps = sps

        def num_samples(self, nbits):
            return nbits * self.sps

        def modulate(self, bits):
            return np.repeat(np.asarray(bits, float) * np.sqrt(2), self.sps).astype(complex)

        def demodulate(self, seg, noise_var, nbits):
            e = np.abs(seg[: nbits * self.sps].reshape(nbits, self.sps)).mean(axis=1)
            return (e.max() / 2 - e) / max(noise_var, 1e-6)

    register_modem("ook", lambda phy, **kw: OOK(phy["sps"]))
    m = get_modem("ook")
    bits = RNG.integers(0, 2, 64).astype(np.uint8)
    assert np.array_equal((m.demodulate(m.modulate(bits), 0.1, 64) < 0).astype(np.uint8), bits)


@pytest.mark.parametrize("spec", ["qpsk", "16qam", "8psk", "dsss:degree=4", "ofdm", "ofdm:const=16qam",
                                  "fsk", "fsk:M=4", "afsk1200"])
def test_modems_unit_power_and_length(spec):
    phy = {"sample_rate": 48000, "sps": 40} if "afsk" in spec else None
    m = get_modem(spec, phy)
    x = m.modulate(RNG.integers(0, 2, 3000).astype(np.uint8))
    assert len(x) == m.num_samples(3000)
    assert 0.85 < np.mean(np.abs(x) ** 2) < 1.15


def test_pilot_tracking_removes_residual_cfo():
    m = SingleCarrierModem(Constellation.qam(16), sps=4, pilot_interval=16)
    bits = RNG.integers(0, 2, 8000).astype(np.uint8)
    x = m.modulate(bits) / 2.0
    n = np.arange(len(x))
    y = x * np.exp(1j * (1.0 + 2 * np.pi * 2e-4 * n)) + 0.03 * (RNG.standard_normal(len(x)) + 1j * RNG.standard_normal(len(x)))
    llr = m.demodulate(y, 2 * 0.03 ** 2, len(bits))
    assert np.mean((llr < 0) != bits) < 1e-3
