"""Classical and quantum-inspired Monte-Carlo solvers for Ising problems.

All solvers are vectorised across independent reads/replicas: a sweep loops
over the ``n`` spins in Python and updates every chain at once with numpy,
so one sweep costs ``O(R * n^2)`` flops for dense couplings.

* :func:`exact`               brute force (n <= 26), ground truth for testing
* :func:`simulated_annealing` Metropolis with a geometric beta schedule
* :func:`parallel_tempering`  replica-exchange Monte Carlo
* :func:`simulated_quantum_annealing`
      path-integral Monte Carlo of the transverse-field Ising model
      (Suzuki-Trotter, Santoro et al., Science 295, 2427, 2002)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .problems import Ising


@dataclass
class Result:
    x: np.ndarray                    # best state (spins, bits or reals depending on problem)
    energy: float
    method: str
    history: list = field(default_factory=list)   # best energy after each outer step
    evaluations: int = 0
    wall_time: float = 0.0
    info: dict = field(default_factory=dict)


def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def default_beta_range(p: Ising) -> tuple[float, float]:
    """Hot start accepts the largest move with p=1/2, cold end rejects the
    smallest non-zero move with p=0.99 (the heuristic used by D-Wave's neal)."""
    field_max = np.abs(p.h) + np.abs(p.J).sum(axis=1)
    dmax = 2 * field_max.max() if field_max.max() > 0 else 1.0
    nz = np.concatenate([np.abs(p.h[p.h != 0]), np.abs(p.J[p.J != 0])])
    dmin = 2 * nz.min() if len(nz) else 1.0
    return np.log(2) / dmax, np.log(100) / dmin


# -- exact -------------------------------------------------------------------------

def exact(p: Ising, chunk_bits: int = 16, keep: int = 1) -> Result:
    n = p.n
    if n > 26:
        raise ValueError("exact enumeration limited to n <= 26")
    t0 = time.time()
    best_e, best_s = np.inf, None
    chunk = 1 << min(chunk_bits, n)
    shifts = np.arange(n - 1, -1, -1)
    degenerate = 0
    for start in range(0, 1 << n, chunk):
        idx = np.arange(start, min(start + chunk, 1 << n), dtype=np.int64)
        s = 1.0 - 2.0 * ((idx[:, None] >> shifts) & 1)
        e = p.energy(s)
        k = int(np.argmin(e))
        if e[k] < best_e - 1e-9:
            best_e, best_s = float(e[k]), s[k].astype(np.int8)
            degenerate = int(np.sum(np.abs(e - e[k]) < 1e-9))
        elif abs(e[k] - best_e) <= 1e-9:
            degenerate += int(np.sum(np.abs(e - best_e) < 1e-9))
    return Result(best_s, best_e, "exact", evaluations=1 << n, wall_time=time.time() - t0,
                  info={"ground_state_degeneracy": degenerate})


# -- Metropolis core -------------------------------------------------------------------

def _sweep(s: np.ndarray, f: np.ndarray, J: np.ndarray, beta: np.ndarray, rng) -> None:
    """One sequential Metropolis sweep, in place. s,f: (R,n); beta: (R,)."""
    R, n = s.shape
    for i in range(n):
        # s_i contributes s_i * f_i to E, so flipping it changes E by -2 s_i f_i
        dE = -2.0 * s[:, i] * f[:, i]
        acc = (dE <= 0) | (rng.random(R) < np.exp(-np.clip(beta * dE, 0, 700)))
        if acc.any():
            rows = np.nonzero(acc)[0]
            delta = -2.0 * s[rows, i]
            s[rows, i] = -s[rows, i]
            f[rows] += delta[:, None] * J[i][None, :]


def simulated_annealing(p: Ising, sweeps: int = 1000, reads: int = 16, beta_range=None,
                        seed=None, init: np.ndarray | None = None) -> Result:
    rng = _rng(seed)
    t0 = time.time()
    b0, b1 = beta_range or default_beta_range(p)
    betas = np.geomspace(b0, b1, sweeps)
    s = rng.choice([-1.0, 1.0], size=(reads, p.n)) if init is None else np.array(init, dtype=np.float64).reshape(reads, p.n)
    f = p.local_fields(s)
    best_e = p.energy(s)
    best_s = s.copy()
    hist = []
    for b in betas:
        _sweep(s, f, p.J, np.full(reads, b), rng)
        e = p.energy(s)
        better = e < best_e
        best_e = np.where(better, e, best_e)
        best_s[better] = s[better]
        hist.append(float(best_e.min()))
    k = int(np.argmin(best_e))
    return Result(best_s[k].astype(np.int8), float(best_e[k]), "sa", hist, sweeps * reads * p.n,
                  time.time() - t0, {"beta_range": (b0, b1), "read_energies": best_e})


def parallel_tempering(p: Ising, sweeps: int = 1000, replicas: int = 16, beta_range=None,
                       seed=None) -> Result:
    rng = _rng(seed)
    t0 = time.time()
    b0, b1 = beta_range or default_beta_range(p)
    betas = np.geomspace(b0, b1, replicas)
    s = rng.choice([-1.0, 1.0], size=(replicas, p.n))
    f = p.local_fields(s)
    e = p.energy(s)
    best_e, best_s = float(e.min()), s[np.argmin(e)].copy()
    hist, swaps, tries = [], 0, 0
    for sweep in range(sweeps):
        _sweep(s, f, p.J, betas, rng)
        e = p.energy(s)
        k = int(np.argmin(e))
        if e[k] < best_e:
            best_e, best_s = float(e[k]), s[k].copy()
        # replica exchange between neighbours, alternating even/odd pairs
        for a in range(sweep % 2, replicas - 1, 2):
            tries += 1
            d = (betas[a] - betas[a + 1]) * (e[a] - e[a + 1])
            if d >= 0 or rng.random() < np.exp(d):
                swaps += 1
                s[[a, a + 1]] = s[[a + 1, a]]
                f[[a, a + 1]] = f[[a + 1, a]]
                e[[a, a + 1]] = e[[a + 1, a]]
        hist.append(best_e)
    return Result(best_s.astype(np.int8), best_e, "pt", hist, sweeps * replicas * p.n, time.time() - t0,
                  {"betas": betas, "swap_rate": swaps / max(tries, 1)})


def simulated_quantum_annealing(p: Ising, sweeps: int = 500, trotter: int = 16, reads: int = 4,
                                gamma_range: tuple[float, float] | None = None, temperature: float | None = None,
                                seed=None) -> Result:
    """Path-integral Monte Carlo of ``H(t) = H_p - Gamma(t) sum_i X_i``.

    The quantum partition function at temperature T maps onto ``P`` coupled
    classical replicas (Trotter slices) with inter-slice ferromagnetic
    coupling ``J_perp = -(P T / 2) ln tanh(Gamma / (P T))``. Gamma is lowered
    linearly; as ``J_perp`` grows the slices are forced into agreement.
    """
    rng = _rng(seed)
    t0 = time.time()
    P, n, R = trotter, p.n, reads
    scale = p.scale()
    T = temperature if temperature is not None else 0.05 * scale * P / 16
    g0, g1 = gamma_range or (3.0 * scale, 1e-3 * scale)
    gammas = np.linspace(g0, g1, sweeps)
    s = rng.choice([-1.0, 1.0], size=(R, P, n))
    fc = p.local_fields(s.reshape(R * P, n)).reshape(R, P, n)
    beta = 1.0 / T
    best_e, best_s = np.inf, None
    hist = []
    J = p.J
    for g in gammas:
        PT = P * T
        jperp = -0.5 * PT * np.log(np.tanh(g / PT))
        for i in range(n):
            si = s[:, :, i]
            nb = np.roll(si, 1, axis=1) + np.roll(si, -1, axis=1)
            # slice-k contribution: s (f/P - J_perp (s_prev + s_next)); flip negates it
            dE = -2.0 * si * (fc[:, :, i] / P - jperp * nb)
            acc = (dE <= 0) | (rng.random((R, P)) < np.exp(-np.clip(beta * dE, 0, 700)))
            if acc.any():
                r, k = np.nonzero(acc)
                delta = -2.0 * s[r, k, i]
                s[r, k, i] = -s[r, k, i]
                fc[r, k] += delta[:, None] * J[i][None, :]
        e = p.energy(s.reshape(R * P, n))
        j = int(np.argmin(e))
        if e[j] < best_e:
            best_e, best_s = float(e[j]), s.reshape(R * P, n)[j].copy()
        hist.append(best_e)
    # final classical polish: zero-temperature sweeps on the best slice
    polished = best_s[None, :].copy()
    fpol = p.local_fields(polished)
    for _ in range(3):
        _sweep(polished, fpol, J, np.array([1e12]), rng)
    pe = float(p.energy(polished)[0])
    if pe < best_e:
        best_e, best_s = pe, polished[0]
    return Result(best_s.astype(np.int8), best_e, "sqa", hist, sweeps * R * P * n, time.time() - t0,
                  {"trotter": P, "temperature": T, "gamma_range": (g0, g1)})
