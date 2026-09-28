"""Array backend selection: NumPy, CuPy (CUDA) or PyTorch (CUDA / MPS / CPU).

Only the handful of kernels the memory and embedding layers need are
abstracted: batched matmul + top-k, FFTs for HRR binding, row normalisation.
Results always come back as NumPy so callers never touch device arrays.

``get_backend("auto")`` prefers torch-CUDA, then CuPy, then torch-MPS, then
NumPy. Override globally with ``EMERGENTNET_BACKEND=numpy|cupy|torch[:device]``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class Backend:
    name: str
    device: str = "cpu"

    # -- conversion ------------------------------------------------------------
    @property
    def xp(self):
        if self.name == "cupy":
            import cupy
            return cupy
        if self.name == "torch":
            import torch
            return torch
        return np

    def asarray(self, x, dtype="float32"):
        if self.name == "numpy":
            return np.asarray(x, dtype=dtype)
        if self.name == "cupy":
            import cupy
            return cupy.asarray(x, dtype=dtype)
        import torch
        t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
        return t.to(device=self.device, dtype=getattr(torch, dtype))

    def to_numpy(self, x) -> np.ndarray:
        if self.name == "numpy":
            return np.asarray(x)
        if self.name == "cupy":
            return x.get()
        return x.detach().cpu().numpy()

    # -- kernels -----------------------------------------------------------------
    def topk(self, matrix, queries, k: int, chunk: int = 65536) -> tuple[np.ndarray, np.ndarray]:
        """Inner-product top-k of each query row against ``matrix`` rows.

        ``matrix`` (N, d) is a backend array; ``queries`` (Q, d) anything.
        Scans in chunks of ``chunk`` rows so memory is O(Q * chunk).
        Returns NumPy (scores, indices) of shape (Q, k), sorted descending.
        """
        q = self.asarray(queries)
        N = matrix.shape[0]
        k = min(k, N)
        best_s = best_i = None
        for s0 in range(0, N, chunk):
            block = matrix[s0:s0 + chunk]
            sc = q @ block.T
            kk = min(k, sc.shape[1])
            if self.name == "torch":
                import torch
                v, i = torch.topk(sc, kk, dim=1)
                v, i = self.to_numpy(v), self.to_numpy(i)
            else:
                xp = self.xp
                part = xp.argpartition(-sc, kk - 1, axis=1)[:, :kk]
                v = xp.take_along_axis(sc, part, axis=1)
                v, i = self.to_numpy(v), self.to_numpy(part)
            i = i + s0
            if best_s is None:
                best_s, best_i = v, i
            else:
                cs, ci = np.concatenate([best_s, v], 1), np.concatenate([best_i, i], 1)
                sel = np.argpartition(-cs, k - 1, axis=1)[:, :k]
                best_s, best_i = np.take_along_axis(cs, sel, 1), np.take_along_axis(ci, sel, 1)
        order = np.argsort(-best_s, axis=1)
        return np.take_along_axis(best_s, order, 1), np.take_along_axis(best_i, order, 1)

    def normalize_rows(self, x):
        xp = self.xp
        if self.name == "torch":
            n = xp.linalg.norm(x, dim=-1, keepdim=True)
            return x / xp.clamp(n, min=1e-12)
        n = xp.linalg.norm(x, axis=-1, keepdims=True)
        return x / xp.maximum(n, 1e-12)

    def rfft(self, x, n: int):
        return self.xp.fft.rfft(x, n=n)

    def irfft(self, X, n: int):
        return self.xp.fft.irfft(X, n=n)


def _torch_device() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=8)
def get_backend(pref: str = "auto") -> Backend:
    pref = os.environ.get("EMERGENTNET_BACKEND", pref) if pref == "auto" else pref
    name, _, dev = pref.partition(":")
    if name == "numpy":
        return Backend("numpy")
    if name == "cupy":
        import cupy  # noqa: F401  (raise if requested but missing)
        return Backend("cupy", dev or "cuda")
    if name == "torch":
        return Backend("torch", dev or _torch_device() or "cpu")
    if name != "auto":
        raise ValueError(f"unknown backend '{pref}'")
    tdev = _torch_device()
    if tdev == "cuda":
        return Backend("torch", "cuda")
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return Backend("cupy", "cuda")
    except Exception:
        pass
    if tdev == "mps":
        return Backend("torch", "mps")
    return Backend("numpy")


def describe() -> dict:
    b = get_backend()
    info = {"backend": b.name, "device": b.device}
    if b.name == "torch" and b.device == "cuda":
        import torch
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
