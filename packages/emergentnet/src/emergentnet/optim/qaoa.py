"""QAOA on an exact statevector simulator (Farhi, Goldstone, Gutmann 2014).

``|psi(gamma, beta)> = prod_l e^{-i beta_l B} e^{-i gamma_l C} |+>^n`` with
``C`` the Ising cost (diagonal in the computational basis) and
``B = sum_i X_i``. Qubit ``q`` in basis state ``z`` maps to spin
``s_q = 1 - 2 z_q``.

Everything is computed exactly: the cost diagonal once (``2^n`` energies),
cost layers as elementwise phases, mixer layers as ``n`` 2x2 rotations applied
by reshaping the state tensor, so a layer costs ``O(n 2^n)``. Practical up
to n ~ 22 qubits in memory (2^22 complex128 = 64 MB).

Angles are optimised with SciPy (``L-BFGS-B`` with analytic-free finite
differences, or ``COBYLA``); depth ``p`` is grown layer by layer using the
INTERP initialisation of Zhou et al., PRX 10, 021067 (2020).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from .annealing import Result, _rng
from .problems import Ising


class QAOASimulator:
    def __init__(self, p: Ising, max_qubits: int = 22):
        if p.n > max_qubits:
            raise ValueError(f"{p.n} qubits exceeds statevector limit {max_qubits}")
        self.problem = p
        self.n = p.n
        idx = np.arange(1 << self.n, dtype=np.int64)
        shifts = np.arange(self.n - 1, -1, -1)
        spins = 1.0 - 2.0 * ((idx[:, None] >> shifts) & 1)
        self.cost = p.energy(spins)          # (2^n,)
        self.spins = spins.astype(np.int8)
        self.evals = 0

    def state(self, gammas, betas) -> np.ndarray:
        n = self.n
        psi = np.full(1 << n, 1 / np.sqrt(1 << n), dtype=np.complex128)
        for g, b in zip(gammas, betas):
            psi *= np.exp(-1j * g * self.cost)
            c, s = np.cos(b), -1j * np.sin(b)
            for q in range(n):
                v = psi.reshape(1 << q, 2, 1 << (n - q - 1))
                a0, a1 = v[:, 0, :].copy(), v[:, 1, :]
                v[:, 0, :] = c * a0 + s * a1
                v[:, 1, :] = s * a0 + c * a1
        return psi

    def expectation(self, params: np.ndarray) -> float:
        self.evals += 1
        k = len(params) // 2
        psi = self.state(params[:k], params[k:])
        return float(np.real(np.vdot(psi, self.cost * psi)))

    def probabilities(self, params: np.ndarray) -> np.ndarray:
        k = len(params) // 2
        return np.abs(self.state(params[:k], params[k:])) ** 2


def _interp(params: np.ndarray) -> np.ndarray:
    """INTERP: extend optimal depth-p angles to a depth-(p+1) initial guess."""
    p = len(params)
    out = np.zeros(p + 1)
    for i in range(p + 1):
        left = params[i - 1] if i - 1 >= 0 else 0.0
        right = params[i] if i < p else 0.0
        out[i] = (i / p) * left + ((p - i) / p) * right
    return out


def qaoa(p: Ising, depth: int = 3, restarts: int = 4, shots: int = 1024, method: str = "L-BFGS-B",
         seed=None, maxiter: int = 300) -> Result:
    rng = _rng(seed)
    t0 = time.time()
    sim = QAOASimulator(p)
    # normalise the cost so angle landscapes are problem-scale independent
    scale = max(np.abs(sim.cost - sim.cost.mean()).max(), 1e-12)
    hist = []
    best = None
    gam, bet = None, None
    for layer in range(1, depth + 1):
        candidates = []
        if gam is not None:
            candidates.append(np.concatenate([_interp(gam), _interp(bet)]))
        n_rand = restarts if layer == 1 else max(1, restarts // 2)
        for _ in range(n_rand):
            candidates.append(np.concatenate([rng.uniform(0, np.pi / scale, layer), rng.uniform(0, np.pi / 2, layer)]))
        layer_best = None
        for x0 in candidates:
            res = minimize(sim.expectation, x0, method=method, options={"maxiter": maxiter})
            if layer_best is None or res.fun < layer_best.fun:
                layer_best = res
        gam, bet = layer_best.x[:layer], layer_best.x[layer:]
        hist.append(float(layer_best.fun))
        best = layer_best
    probs = sim.probabilities(best.x)
    samples = rng.choice(len(probs), size=shots, p=probs / probs.sum())
    energies = sim.cost[samples]
    k = int(np.argmin(energies))
    e_min, e_max = sim.cost.min(), sim.cost.max()
    approx = (e_max - best.fun) / (e_max - e_min) if e_max > e_min else 1.0
    return Result(sim.spins[samples[k]], float(energies[k]), "qaoa", hist, sim.evals, time.time() - t0,
                  {"depth": depth, "gammas": gam, "betas": bet, "expectation": float(best.fun),
                   "approximation_ratio": float(approx),
                   "p_ground": float(probs[np.isclose(sim.cost, e_min)].sum()),
                   "ground_energy": float(e_min)})
