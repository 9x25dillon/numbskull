"""GF(2^8) arithmetic with log/antilog tables.

Polynomials are Python lists / numpy arrays of field elements, **highest
degree first** (``p[0]`` multiplies ``x^(len(p)-1)``).
"""

from __future__ import annotations

import numpy as np

PRIM_POLY = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1, generator alpha = 2

EXP = np.zeros(512, dtype=np.int32)
LOG = np.zeros(256, dtype=np.int32)
_x = 1
for _i in range(255):
    EXP[_i] = _x
    LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= PRIM_POLY
EXP[255:510] = EXP[:255]
del _x, _i


def mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return int(EXP[LOG[a] + LOG[b]])


def div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("GF(256) division by zero")
    if a == 0:
        return 0
    return int(EXP[(LOG[a] - LOG[b]) % 255])


def pow_(a: int, p: int) -> int:
    if a == 0:
        return 0 if p else 1
    return int(EXP[(LOG[a] * p) % 255])


def inv(a: int) -> int:
    return int(EXP[255 - LOG[a]])


def vmul(a: np.ndarray, b: int) -> np.ndarray:
    """Vector * scalar."""
    a = np.asarray(a, dtype=np.int32)
    if b == 0:
        return np.zeros_like(a)
    out = EXP[LOG[a] + LOG[b]]
    return np.where(a == 0, 0, out)


def poly_scale(p, x: int) -> list:
    return [mul(c, x) for c in p]


def poly_add(p, q) -> list:
    r = [0] * max(len(p), len(q))
    for i, c in enumerate(p):
        r[i + len(r) - len(p)] = c
    for i, c in enumerate(q):
        r[i + len(r) - len(q)] ^= c
    return r


def poly_mul(p, q) -> list:
    r = [0] * (len(p) + len(q) - 1)
    for j, qj in enumerate(q):
        if qj == 0:
            continue
        lq = LOG[qj]
        for i, pi in enumerate(p):
            if pi:
                r[i + j] ^= int(EXP[LOG[pi] + lq])
    return r


def poly_eval(p, x: int) -> int:
    y = p[0]
    for c in p[1:]:
        y = mul(y, x) ^ c
    return y


def poly_eval_many(p, xs: np.ndarray) -> np.ndarray:
    """Horner evaluation of ``p`` at every element of ``xs`` (vectorised)."""
    xs = np.asarray(xs, dtype=np.int32)
    y = np.full(xs.shape, int(p[0]), dtype=np.int32)
    logx = LOG[xs]
    zx = xs == 0
    for c in p[1:]:
        prod = np.where((y == 0) | zx, 0, EXP[LOG[y] + logx])
        y = prod ^ int(c)
    return y
