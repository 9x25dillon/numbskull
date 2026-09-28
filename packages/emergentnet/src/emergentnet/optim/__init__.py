"""Optimisation: problem models and solvers behind one ``solve`` entry point.

>>> from emergentnet.optim import maxcut_from_edges, solve
>>> r = solve(maxcut_from_edges([(0, 1), (1, 2), (2, 0), (2, 3)]), method="sqa", seed=0)
>>> -r.energy   # cut size
3.0

Methods for Ising/QUBO: ``exact``, ``sa``, ``pt``, ``sqa``, ``qaoa``, ``auto``.
Methods for :class:`Continuous`: ``qpso``, ``pso``, ``auto`` (= qpso).
A QUBO is solved as its equivalent Ising model and the result mapped back to
bits (``x = (1 + s) / 2``); energies are identical under the mapping.
"""

from __future__ import annotations

import numpy as np

from .annealing import Result, exact, parallel_tempering, simulated_annealing, simulated_quantum_annealing
from .problems import (QUBO, Continuous, Ising, maxcut, maxcut_from_edges, number_partitioning,
                       random_regular_graph, sherrington_kirkpatrick)
from .qaoa import QAOASimulator, qaoa
from .swarm import pso, qpso

DISCRETE = {
    "exact": exact,
    "sa": simulated_annealing,
    "pt": parallel_tempering,
    "sqa": simulated_quantum_annealing,
    "qaoa": qaoa,
}
CONTINUOUS = {"qpso": qpso, "pso": pso}


def solve(problem, method: str = "auto", **kw) -> Result:
    if isinstance(problem, Continuous):
        m = "qpso" if method == "auto" else method
        if m not in CONTINUOUS:
            raise ValueError(f"method '{m}' not valid for continuous problems: {sorted(CONTINUOUS)}")
        return CONTINUOUS[m](problem, **kw)
    as_qubo = isinstance(problem, QUBO)
    ising = problem.to_ising() if as_qubo else problem
    if not isinstance(ising, Ising):
        raise TypeError("problem must be Ising, QUBO or Continuous")
    m = method
    if m == "auto":
        m = "exact" if ising.n <= 18 else "pt"
    if m not in DISCRETE:
        raise ValueError(f"unknown method '{m}': {sorted(DISCRETE)}")
    res = DISCRETE[m](ising, **kw)
    if as_qubo:
        res.x = ((1 + res.x.astype(np.int64)) // 2).astype(np.uint8)
    return res


__all__ = [
    "Ising", "QUBO", "Continuous", "Result", "solve", "exact", "simulated_annealing", "parallel_tempering",
    "simulated_quantum_annealing", "qaoa", "QAOASimulator", "qpso", "pso", "maxcut", "maxcut_from_edges",
    "number_partitioning", "sherrington_kirkpatrick", "random_regular_graph",
]
