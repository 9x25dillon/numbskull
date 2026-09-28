"""Reed-Solomon codes over GF(2^8).

Errors-and-erasures decoder: syndromes -> Forney syndromes (erasures removed)
-> Berlekamp-Massey error locator -> Chien search -> Forney magnitudes.

``RS(n, k)`` corrects ``e`` errors and ``f`` erasures while ``2e + f <= n - k``.
Shortened codes (``n < 255``) are supported natively. The bit-level codec adds
generalised-minimum-distance (GMD) decoding: when errors-only decoding fails it
retries while erasing the least reliable bytes as judged from channel LLRs.
"""

from __future__ import annotations

import numpy as np

from . import gf256 as gf
from .base import BlockCodec, DecodeResult


class ReedSolomonError(Exception):
    pass


class RSCode:
    """Byte-level RS(n, k) with first consecutive root ``alpha^fcr``."""

    def __init__(self, n: int = 255, k: int = 223, fcr: int = 0):
        if not 0 < k < n <= 255:
            raise ValueError("require 0 < k < n <= 255")
        self.n, self.k, self.fcr = n, k, fcr
        self.nsym = n - k
        g = [1]
        for i in range(self.nsym):
            g = gf.poly_mul(g, [1, gf.pow_(2, i + fcr)])
        self.gen = g
        self._roots = np.array([gf.pow_(2, i + fcr) for i in range(self.nsym)], dtype=np.int32)

    # -- encoding ----------------------------------------------------------------
    def encode(self, msg: bytes | np.ndarray) -> np.ndarray:
        msg = np.frombuffer(bytes(msg), dtype=np.uint8) if isinstance(msg, (bytes, bytearray)) else np.asarray(msg)
        if len(msg) != self.k:
            raise ValueError(f"message must be {self.k} bytes")
        out = [int(x) for x in msg] + [0] * self.nsym
        gen = self.gen
        for i in range(self.k):
            coef = out[i]
            if coef:
                lc = gf.LOG[coef]
                for j in range(1, len(gen)):
                    if gen[j]:
                        out[i + j] ^= int(gf.EXP[gf.LOG[gen[j]] + lc])
        out[: self.k] = [int(x) for x in msg]
        return np.array(out, dtype=np.uint8)

    # -- decoding ----------------------------------------------------------------
    def syndromes(self, cw) -> list:
        """Syndromes with a leading 0 pad (index shift used by BM/Forney)."""
        return [0] + [int(s) for s in gf.poly_eval_many(list(cw), self._roots)]

    def decode(self, cw, erase_pos=()) -> tuple[np.ndarray, int]:
        """Return ``(message, n_corrected)`` or raise :class:`ReedSolomonError`."""
        cw = [int(x) for x in cw]
        if len(cw) != self.n:
            raise ValueError(f"codeword must be {self.n} bytes")
        erase_pos = sorted(set(int(p) for p in erase_pos))
        if len(erase_pos) > self.nsym:
            raise ReedSolomonError("too many erasures")
        for p in erase_pos:
            cw[p] = 0
        synd = self.syndromes(cw)
        if max(synd) == 0:
            return np.array(cw[: self.k], dtype=np.uint8), 0
        fsynd = self._forney_syndromes(synd, erase_pos, len(cw))
        err_loc = self._error_locator(fsynd, len(erase_pos))
        err_pos = self._chien(err_loc[::-1], len(cw))
        errata = erase_pos + err_pos
        cw = self._correct_errata(cw, synd, errata)
        if max(self.syndromes(cw)) != 0:
            raise ReedSolomonError("could not correct codeword")
        return np.array(cw[: self.k], dtype=np.uint8), len(errata)

    def _forney_syndromes(self, synd, pos, nmess):
        fsynd = list(synd[1:])
        for p in pos:
            x = gf.pow_(2, nmess - 1 - p)
            for j in range(len(fsynd) - 1):
                fsynd[j] = gf.mul(fsynd[j], x) ^ fsynd[j + 1]
        return fsynd

    def _error_locator(self, synd, erase_count):
        err_loc, old_loc = [1], [1]
        for i in range(self.nsym - erase_count):
            k = i
            delta = synd[k]
            for j in range(1, len(err_loc)):
                delta ^= gf.mul(err_loc[-(j + 1)], synd[k - j])
            old_loc = old_loc + [0]
            if delta != 0:
                if len(old_loc) > len(err_loc):
                    new_loc = gf.poly_scale(old_loc, delta)
                    old_loc = gf.poly_scale(err_loc, gf.inv(delta))
                    err_loc = new_loc
                err_loc = gf.poly_add(err_loc, gf.poly_scale(old_loc, delta))
        while err_loc and err_loc[0] == 0:
            err_loc = err_loc[1:]
        errs = len(err_loc) - 1
        if 2 * errs + erase_count > self.nsym:
            raise ReedSolomonError("too many errors")
        return err_loc

    def _chien(self, err_loc_rev, nmess):
        errs = len(err_loc_rev) - 1
        xs = np.array([gf.pow_(2, i) for i in range(nmess)], dtype=np.int32)
        vals = gf.poly_eval_many(err_loc_rev, xs)
        pos = [nmess - 1 - int(i) for i in np.nonzero(vals == 0)[0]]
        if len(pos) != errs:
            raise ReedSolomonError("error locator degree does not match root count")
        return pos

    def _correct_errata(self, cw, synd, err_pos):
        nm = len(cw)
        coef_pos = [nm - 1 - p for p in err_pos]
        loc = [1]
        for c in coef_pos:
            loc = gf.poly_mul(loc, gf.poly_add([1], [gf.pow_(2, c), 0]))
        # evaluator Omega = (S * Lambda) mod x^(nu+1)
        nu = len(loc) - 1
        prod = gf.poly_mul(synd[::-1], loc)
        err_eval = prod[-(nu + 1):][::-1]
        X = [gf.pow_(2, -(255 - c)) for c in coef_pos]
        for i, Xi in enumerate(X):
            Xi_inv = gf.inv(Xi)
            denom = 1
            for j, Xj in enumerate(X):
                if j != i:
                    denom = gf.mul(denom, 1 ^ gf.mul(Xi_inv, Xj))
            if denom == 0:
                raise ReedSolomonError("Forney denominator vanished")
            y = gf.poly_eval(err_eval[::-1], Xi_inv)
            y = gf.mul(gf.pow_(Xi, 1 - self.fcr), y)
            cw[err_pos[i]] ^= gf.div(y, denom)
        return cw


class ReedSolomon(BlockCodec):
    """Bit-level RS codec with GMD (erasure) retries driven by LLR reliability."""

    def __init__(self, n: int = 255, k: int = 223, fcr: int = 0, gmd: bool = True):
        self.code = RSCode(n, k, fcr)
        self.gmd = gmd
        self.name = f"rs{n}_{k}"

    @property
    def k(self) -> int:
        return 8 * self.code.k

    @property
    def n(self) -> int:
        return 8 * self.code.n

    def encode_block(self, info: np.ndarray) -> np.ndarray:
        msg = np.packbits(np.asarray(info, dtype=np.uint8))
        return np.unpackbits(self.code.encode(msg))

    def decode_block(self, llr: np.ndarray) -> DecodeResult:
        hard = (llr < 0).astype(np.uint8)
        cw = np.packbits(hard)
        try:
            msg, nc = self.code.decode(cw)
            return DecodeResult(bits=np.unpackbits(msg), ok=True, corrected=nc)
        except ReedSolomonError:
            pass
        if self.gmd:
            reliability = np.abs(llr).reshape(self.code.n, 8).min(axis=1)
            order = np.argsort(reliability, kind="stable")
            for n_erase in range(2, self.code.nsym + 1, 2):
                try:
                    msg, nc = self.code.decode(cw, order[:n_erase])
                    return DecodeResult(bits=np.unpackbits(msg), ok=True, corrected=nc)
                except ReedSolomonError:
                    continue
        return DecodeResult(bits=hard.reshape(self.code.n, 8)[: self.code.k].ravel(), ok=False)
