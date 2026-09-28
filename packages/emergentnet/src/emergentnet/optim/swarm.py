"""Population methods for continuous box-constrained problems.

* :func:`qpso` Quantum-behaved PSO (Sun, Feng, Xu, CEC 2004): each particle
  samples around an attractor ``p = phi*pbest + (1-phi)*gbest`` from a
  delta-potential-well distribution,
  ``x = p +- alpha |mbest - x| ln(1/u)``, with the contraction-expansion
  coefficient ``alpha`` decreasing 1.0 -> 0.5. No velocities; one parameter.
* :func:`pso` Classical inertia-weight PSO (Shi & Eberhart 1998).
"""

from __future__ import annotations

import time

import numpy as np

from .annealing import Result, _rng
from .problems import Continuous


def _init(prob: Continuous, n: int, rng, init: np.ndarray | None):
    if init is not None:
        X = np.array(init, dtype=np.float64)
        if len(X) < n:
            extra = rng.uniform(prob.lower, prob.upper, size=(n - len(X), prob.dim))
            X = np.vstack([X, extra])
        return np.clip(X[:n], prob.lower, prob.upper)
    return rng.uniform(prob.lower, prob.upper, size=(n, prob.dim))


def qpso(prob: Continuous, particles: int = 40, iterations: int = 300, alpha=(1.0, 0.5), seed=None,
         init: np.ndarray | None = None, tol: float | None = None) -> Result:
    rng = _rng(seed)
    t0 = time.time()
    X = _init(prob, particles, rng, init)
    fx = prob.evaluate(X)
    P, fp = X.copy(), fx.copy()
    g = int(np.argmin(fp))
    hist, evals = [float(fp[g])], particles
    for it in range(iterations):
        a = alpha[0] - (alpha[0] - alpha[1]) * it / max(iterations - 1, 1)
        mbest = P.mean(axis=0)
        phi = rng.random((particles, prob.dim))
        attractor = phi * P + (1 - phi) * P[g]
        u = rng.random((particles, prob.dim))
        sign = np.where(rng.random((particles, prob.dim)) < 0.5, -1.0, 1.0)
        X = attractor + sign * a * np.abs(mbest - X) * np.log(1.0 / np.maximum(u, 1e-300))
        X = np.clip(X, prob.lower, prob.upper)
        fx = prob.evaluate(X)
        evals += particles
        imp = fx < fp
        P[imp], fp[imp] = X[imp], fx[imp]
        g = int(np.argmin(fp))
        hist.append(float(fp[g]))
        if tol is not None and len(hist) > 20 and abs(hist[-21] - hist[-1]) < tol:
            break
    return Result(P[g].copy(), float(fp[g]), "qpso", hist, evals, time.time() - t0,
                  {"iterations": len(hist) - 1, "swarm_diversity": float(np.mean(np.std(P, axis=0)))})


def pso(prob: Continuous, particles: int = 40, iterations: int = 300, w=(0.9, 0.4), c1: float = 2.0,
        c2: float = 2.0, seed=None, init: np.ndarray | None = None) -> Result:
    rng = _rng(seed)
    t0 = time.time()
    X = _init(prob, particles, rng, init)
    span = prob.upper - prob.lower
    V = rng.uniform(-span, span, X.shape) * 0.1
    fx = prob.evaluate(X)
    P, fp = X.copy(), fx.copy()
    g = int(np.argmin(fp))
    hist, evals = [float(fp[g])], particles
    for it in range(iterations):
        wt = w[0] - (w[0] - w[1]) * it / max(iterations - 1, 1)
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        V = wt * V + c1 * r1 * (P - X) + c2 * r2 * (P[g] - X)
        V = np.clip(V, -span, span)
        X = np.clip(X + V, prob.lower, prob.upper)
        fx = prob.evaluate(X)
        evals += particles
        imp = fx < fp
        P[imp], fp[imp] = X[imp], fx[imp]
        g = int(np.argmin(fp))
        hist.append(float(fp[g]))
    return Result(P[g].copy(), float(fp[g]), "pso", hist, evals, time.time() - t0)
