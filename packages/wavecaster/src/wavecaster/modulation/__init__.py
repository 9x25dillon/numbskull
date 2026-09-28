"""Modulation: constellations, pulse shaping and waveform modems.

Modems are built from spec strings ``name[:key=value,...]`` so TX/RX and the
CLI share one vocabulary::

    bpsk qpsk 8psk 16psk 16qam 64qam 256qam 16apsk 32apsk 4pam
    dsss[:const=bpsk,degree=5]        direct-sequence spread (m-sequence 2^deg-1)
    ofdm[:const=qpsk,nfft=64,cp=16,used=48,pilots=4]
    fsk[:M=2,spacing=1000]            noncoherent CP-FSK (afsk1200 = Bell-202-like)
    custom:file=my_constellation.json custom constellation (see Constellation.from_spec)
    sc:const=<name>                   single-carrier with any registered constellation

Extending
---------
* New constellation: ``register_constellation("my16", lambda: Constellation(points, labels))``
  -> immediately usable as ``my16``, ``ofdm:const=my16``, ``dsss:const=my16``.
* New waveform: subclass :class:`Modem` and ``register_modem("name", factory)``;
  ``factory(phy: dict, **params) -> Modem`` where ``phy`` holds
  ``sps, rolloff, span, sample_rate, pilot_interval``.
* Out-of-tree plugins: expose a factory under the ``wavecaster.modems`` or
  ``wavecaster.constellations`` entry-point group; loaded on first lookup.
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np

from .constellation import APSK16, APSK32, Constellation
from .modems import FSKModem, Modem, OFDMModem, SingleCarrierModem, m_sequence
from .pulse import matched_filter, rrc_taps, shape

ModemFactory = Callable[..., Modem]

_CONSTELLATIONS: dict[str, Callable[[], Constellation]] = {
    "bpsk": lambda: Constellation.psk(2),
    "qpsk": lambda: Constellation.psk(4),
    "8psk": lambda: Constellation.psk(8),
    "16psk": lambda: Constellation.psk(16),
    "4pam": lambda: Constellation.pam(4),
    "16qam": lambda: Constellation.qam(16),
    "64qam": lambda: Constellation.qam(64),
    "256qam": lambda: Constellation.qam(256),
    "16apsk": lambda: Constellation.apsk(APSK16, "16apsk"),
    "32apsk": lambda: Constellation.apsk(APSK32, "32apsk"),
}
_MODEMS: dict[str, ModemFactory] = {}
_PLUGINS_LOADED = False


def register_constellation(name: str, factory: Callable[[], Constellation]) -> None:
    _CONSTELLATIONS[name.lower()] = factory


def register_modem(name: str, factory: ModemFactory) -> None:
    _MODEMS[name.lower()] = factory


def _load_plugins() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return
    eps = entry_points()
    for group, reg in (("wavecaster.constellations", register_constellation), ("wavecaster.modems", register_modem)):
        sel = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
        for ep in sel:
            try:
                reg(ep.name, ep.load())
            except Exception as exc:  # a broken plugin must not break the core
                import logging
                logging.getLogger(__name__).warning("plugin %s failed to load: %s", ep.name, exc)


def get_constellation(name: str | dict | Constellation) -> Constellation:
    if isinstance(name, Constellation):
        return name
    if isinstance(name, dict):
        return Constellation.from_spec(name)
    _load_plugins()
    key = name.lower()
    if key in _CONSTELLATIONS:
        return _CONSTELLATIONS[key]()
    if key.endswith(".json"):
        with open(name) as f:
            return Constellation.from_spec(json.load(f))
    raise KeyError(f"unknown constellation '{name}'; known: {sorted(_CONSTELLATIONS)}")


def parse_spec(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition(":")
    params: dict = {}
    for item in filter(None, (s.strip() for s in rest.split(","))):
        k, _, v = item.partition("=")
        params[k.strip()] = _coerce(v.strip())
    return name.strip().lower(), params


def _coerce(v: str):
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


DEFAULT_PHY = {"sps": 8, "rolloff": 0.35, "span": 8, "sample_rate": 1.0, "pilot_interval": 32}


def get_modem(spec: str | Modem, phy: dict | None = None) -> Modem:
    if isinstance(spec, Modem):
        return spec
    _load_plugins()
    phy = {**DEFAULT_PHY, **(phy or {})}
    name, params = parse_spec(spec)
    if name in _MODEMS:
        return _MODEMS[name](phy, **params)
    if name in _CONSTELLATIONS:
        return _sc(phy, const=name, **params)
    raise KeyError(f"unknown modem '{name}'; known: {available_modems()}")


def available_modems() -> list[str]:
    _load_plugins()
    return sorted(set(_MODEMS) | set(_CONSTELLATIONS))


def _sc(phy, const="qpsk", pilots=None, **_):
    return SingleCarrierModem(get_constellation(const), phy["sps"], phy["rolloff"], phy["span"],
                              phy["pilot_interval"] if pilots is None else int(pilots))


def _dsss(phy, const="bpsk", degree=5, pilots=None, **_):
    c = get_constellation(const)
    L = (1 << int(degree)) - 1
    # keep the pilot spacing (in chips) close to the unspread modem's so the
    # residual-CFO phase step between pilots stays well inside +-pi
    default = max(2, phy["pilot_interval"] // max(1, L // 4)) if phy["pilot_interval"] else 0
    return SingleCarrierModem(c, phy["sps"], phy["rolloff"], phy["span"],
                              default if pilots is None else int(pilots),
                              spreading=m_sequence(int(degree)), name=f"dsss-{c.name}-{(1 << int(degree)) - 1}")


def _ofdm(phy, const="qpsk", nfft=64, cp=16, used=48, pilots=4, **_):
    return OFDMModem(get_constellation(const), int(nfft), int(cp), int(used), int(pilots))


def _fsk(phy, M=2, spacing=None, sps=None, **_):
    sps = int(sps or phy["sps"])
    fs = float(phy["sample_rate"])
    if spacing is None:
        spacing = fs / sps  # orthogonal for noncoherent detection (h = 1)
    return FSKModem(int(M), sps, float(spacing), fs)


def _afsk1200(phy, **_):
    # Bell-202-like: 1200 Bd, 1200/2200 Hz once centred on a 1700 Hz carrier
    fs = float(phy["sample_rate"])
    return FSKModem(2, int(round(fs / 1200)), 1000.0, fs, name="afsk1200")


def _custom(phy, file=None, **params):
    if file is None:
        raise ValueError("custom modem needs file=<constellation.json>")
    with open(file) as f:
        c = Constellation.from_spec(json.load(f))
    return SingleCarrierModem(c, phy["sps"], phy["rolloff"], phy["span"], phy["pilot_interval"])


register_modem("sc", _sc)
register_modem("dsss", _dsss)
register_modem("ofdm", _ofdm)
register_modem("fsk", _fsk)
register_modem("afsk1200", _afsk1200)
register_modem("custom", _custom)

__all__ = [
    "Constellation", "Modem", "SingleCarrierModem", "OFDMModem", "FSKModem", "m_sequence",
    "get_modem", "get_constellation", "register_modem", "register_constellation",
    "available_modems", "parse_spec", "rrc_taps", "shape", "matched_filter", "np",
]
