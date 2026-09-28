"""Forward error correction.

``get_codec(spec)`` builds a codec from a compact string so TX and RX can
agree on a code by name::

    none | hamming74 | rs[:n,k] | ldpc[:n,rate[,dv[,seed]]] | turbo[:K[,iters[,p]]] | conv[:k]

Examples: ``rs:255,223``, ``ldpc:1024,0.5``, ``turbo:512,8,p``, ``conv:256``.
Additional codecs can be registered with :func:`register_codec`.
"""

from __future__ import annotations

from typing import Callable

from .base import BlockCodec, DecodeResult, NullCodec
from .convolutional import Convolutional
from .hamming import Hamming74
from .ldpc import LDPC
from .reed_solomon import ReedSolomon, RSCode, ReedSolomonError
from .turbo import Turbo

_REGISTRY: dict[str, Callable[..., BlockCodec]] = {}


def register_codec(name: str, factory: Callable[..., BlockCodec]) -> None:
    _REGISTRY[name.lower()] = factory


def _num(x: str):
    try:
        return int(x)
    except ValueError:
        return float(x)


register_codec("none", lambda: NullCodec())
register_codec("hamming74", lambda: Hamming74())
register_codec("rs", lambda n=255, k=223: ReedSolomon(int(n), int(k)))
register_codec("ldpc", lambda n=1024, rate=0.5, dv=3, seed=0: LDPC.peg(int(n), float(rate), int(dv), int(seed)))
register_codec("turbo", lambda K=1024, iters=8, p=None: Turbo(int(K), int(iters), puncture=p == "p"))
register_codec("conv", lambda k=256: Convolutional(int(k)))


def get_codec(spec: str | BlockCodec) -> BlockCodec:
    if isinstance(spec, BlockCodec):
        return spec
    name, _, args = spec.partition(":")
    name = name.strip().lower()
    if name not in _REGISTRY:
        raise KeyError(f"unknown codec '{name}'; known: {sorted(_REGISTRY)}")
    params = [a if a == "p" else _num(a) for a in args.split(",") if a.strip()] if args else []
    return _REGISTRY[name](*params)


def available_codecs() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "BlockCodec", "DecodeResult", "NullCodec", "Hamming74", "ReedSolomon", "RSCode",
    "ReedSolomonError", "LDPC", "Turbo", "Convolutional", "get_codec", "register_codec",
    "available_codecs",
]
