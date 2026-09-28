"""wavecaster: software-defined modem.

Layers (each usable on its own)::

    fec/          block codecs: Hamming, Reed-Solomon, LDPC, turbo, convolutional
    modulation/   constellations + waveforms (single-carrier, DSSS, OFDM, FSK), registry
    phy           burst framing, synchronisation, streaming receiver
    radio/        SDR (SoapySDR, UHD), sound card + PTT, KISS TNC, files, loopback
    link          real-time TX/RX over a device; transport segmentation
"""

__version__ = "0.2.0"

from .phy import PROFILES, Frame, PHYConfig, Receiver, Transmitter, get_profile  # noqa: E402

__all__ = ["PHYConfig", "Transmitter", "Receiver", "Frame", "PROFILES", "get_profile", "__version__"]
