"""Drop-in replacement for the legacy ``QuantumOptimizationProtocol``.

The legacy class (``emergent_cognitive_network*.py``) built a 2x2 "rotation"
for an n-dimensional state (a shape error for n != 2) and never fed the
objective back into the state, so it could not optimise anything. This
adapter keeps its constructor and ``optimize`` signature and return keys, and
runs quantum-behaved PSO over the state space:

* the swarm is seeded with the given state plus perturbations of relative
  size ``coupling_strength``;
* the search box is ``+-scaling_factor * max(1, 2 * max|state|)`` per axis;
* ``maximize=True`` (default) matches the legacy call sites, which pass
  fitness-style objectives such as ``lambda x: -np.sum(x**2)``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .optim import Continuous, qpso


class QuantumOptimizationProtocol:
    def __init__(self, state_space: np.ndarray, scaling_factor: float = 1.0, coupling_strength: float = 0.5,
                 particles: int = 30, seed: int | None = None):
        self.state = np.asarray(state_space, dtype=np.float64)
        self.scaling_factor = float(scaling_factor)
        self.coupling_strength = float(coupling_strength)
        self.particles = particles
        self.seed = seed

    def optimize(self, objective_function: Callable[[np.ndarray], float], max_iterations: int = 100,
                 maximize: bool = True) -> dict:
        shape = self.state.shape
        x0 = self.state.ravel()
        bound = self.scaling_factor * max(1.0, 2.0 * float(np.max(np.abs(x0))) if x0.size else 1.0)
        sign = -1.0 if maximize else 1.0

        def f(x: np.ndarray) -> float:
            return sign * float(objective_function(x.reshape(shape)))

        prob = Continuous(f, -bound * np.ones(x0.size), bound * np.ones(x0.size))
        rng = np.random.default_rng(self.seed)
        init = x0[None, :] + self.coupling_strength * bound * rng.standard_normal((self.particles, x0.size))
        init[0] = x0
        res = qpso(prob, particles=self.particles, iterations=max_iterations, seed=rng, init=init)
        hist = [{"iteration": i, "objective": sign * v} for i, v in enumerate(res.history)]
        recent = [h["objective"] for h in hist[-10:]]
        return {
            "final_state": res.x.reshape(shape),
            "best_objective": sign * res.energy,
            "optimization_history": hist,
            "convergence": len(recent) >= 10 and float(np.std(recent)) < 1e-6 * max(1.0, abs(recent[-1])),
            "evaluations": res.evaluations,
            "method": "qpso",
        }
