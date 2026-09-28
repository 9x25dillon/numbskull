"""Three ways to add modulation schemes, each run through the full PHY.

1. a constellation defined as data (JSON-compatible dict);
2. a registered constellation, reused automatically by OFDM and DSSS;
3. a brand-new waveform (on-off keying) implementing the Modem contract.
"""

import numpy as np

from wavecaster.channel import Channel
from wavecaster.modulation import Constellation, Modem, register_constellation, register_modem
from wavecaster.phy import PHYConfig, Receiver, Transmitter

# 1. geometric 8-point constellation: 7 on a hexagonal lattice + 1 outer point
HEX8 = {"name": "hex8",
        "points": [[1, 0], [0.5, 0.87], [-0.5, 0.87], [-1, 0], [-0.5, -0.87], [0.5, -0.87], [0, 0], [2, 0]],
        "labels": [0, 1, 3, 2, 6, 7, 5, 4]}
register_constellation("hex8", lambda: Constellation.from_spec(HEX8))


# 3. a non-linear waveform: OOK with energy detection
class OOK(Modem):
    name = "ook"
    bits_per_symbol = 1

    def __init__(self, sps: int):
        self.sps = sps

    def num_samples(self, nbits):
        return nbits * self.sps

    def modulate(self, bits):
        # "on" amplitude sqrt(2) keeps unit average power for balanced (scrambled) bits
        return np.repeat(np.asarray(bits, float) * np.sqrt(2), self.sps).astype(complex)

    def demodulate(self, seg, noise_var, nbits):
        e = np.abs(seg[: nbits * self.sps].reshape(nbits, self.sps)).mean(axis=1)
        return (e.max() / 2 - e) * 4 / max(noise_var, 1e-6)  # bit 1 = energy present -> negative LLR


register_modem("ook", lambda phy, **kw: OOK(phy["sps"]))


def link(modem: str, esn0: float = 18.0) -> bool:
    cfg = PHYConfig(modem=modem, fec="ldpc:512,0.5")
    msg = b"custom modulation over the full PHY"
    y = Channel(esn0, cfg.sps, cfo=2e-4, phase=1.0, delay=700, seed=1)(Transmitter(cfg).modulate(msg, pad=400))
    rx = Receiver(cfg)
    frames = rx.feed(y) + rx.flush()
    return any(f.ok and f.payload == msg for f in frames)


if __name__ == "__main__":
    for m in ("hex8", "ofdm:const=hex8", "dsss:const=hex8,degree=4", "ook"):
        print(f"{m:28s} link ok: {link(m)}")
