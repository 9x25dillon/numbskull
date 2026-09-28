import numpy as np
import pytest

from emergentnet.compat import QuantumOptimizationProtocol
from emergentnet.optim import (QUBO, Continuous, Ising, QAOASimulator, maxcut, maxcut_from_edges,
                               number_partitioning, random_regular_graph, sherrington_kirkpatrick, solve)

RNG = np.random.default_rng(0)


def test_qubo_ising_roundtrip_energies():
    Q = QUBO(RNG.normal(size=(9, 9)), 0.7)
    I = Q.to_ising()
    x = RNG.integers(0, 2, (100, 9))
    assert np.allclose(Q.energy(x), I.energy(2 * x - 1))
    assert np.allclose(I.to_qubo().energy(x), Q.energy(x))


def test_qubo_from_dict_matches_dense():
    terms = {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0, (1, 2): -0.5}
    q = QUBO.from_dict(terms)
    x = np.array([1, 0, 1])
    assert q.energy(x) == pytest.approx(-1.0)


def test_maxcut_energy_is_negative_cut():
    I = maxcut_from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])
    s = np.array([1, -1, 1, -1])  # cut edges: (0,1),(1,2),(2,3) = 3
    assert I.energy(s) == pytest.approx(-3.0)


@pytest.mark.parametrize("method", ["sa", "pt", "sqa"])
def test_monte_carlo_solvers_find_exact_ground_state(method):
    p = sherrington_kirkpatrick(16, seed=4)
    ref = solve(p, "exact").energy
    r = solve(p, method, seed=1)
    assert r.energy == pytest.approx(ref, abs=1e-9)
    assert p.energy(r.x) == pytest.approx(r.energy)


def test_qubo_solution_returned_as_bits():
    Q = QUBO(RNG.normal(size=(10, 10)))
    r = solve(Q, "sa", seed=0)
    assert set(np.unique(r.x)) <= {0, 1}
    assert Q.energy(r.x) == pytest.approx(r.energy)
    assert r.energy == pytest.approx(solve(Q, "exact").energy)


def test_number_partitioning_perfect():
    r = solve(number_partitioning([4, 5, 6, 7, 8]), "exact")
    assert r.energy == pytest.approx(0.0)  # {4,5,6} vs {7,8}


def test_qaoa_statevector_is_normalised_and_unitary():
    sim = QAOASimulator(sherrington_kirkpatrick(6, seed=1))
    psi = sim.state([0.3, 0.7], [0.4, 0.1])
    assert np.vdot(psi, psi).real == pytest.approx(1.0)
    # beta = gamma = 0 leaves |+>^n: expectation = mean energy
    assert sim.expectation(np.array([0.0, 0.0])) == pytest.approx(sim.cost.mean())


def test_qaoa_p1_matches_analytic_single_edge():
    # One-edge MaxCut. For C_cut = (1 - Z0 Z1)/2 evolved with e^{-i gamma C_cut},
    # <C_cut> = 1/2 + 1/2 sin(4 beta) sin(gamma). Our cost is C = -C_cut and we
    # evolve with e^{-i gamma C} (gamma -> -gamma), so <C> = -1/2 + 1/2 sin(4b) sin(g).
    I = maxcut_from_edges([(0, 1)])
    sim = QAOASimulator(I)
    for g, b in [(0.3, 0.2), (1.1, 0.35), (2.0, 1.0)]:
        expect = -0.5 + 0.5 * np.sin(4 * b) * np.sin(g)
        assert sim.expectation(np.array([g, b])) == pytest.approx(expect, abs=1e-9)


def test_qaoa_solves_small_maxcut():
    A = random_regular_graph(8, 3, seed=2)
    I = maxcut(A)
    r = solve(I, "qaoa", depth=2, seed=0, shots=512)
    assert r.energy == pytest.approx(solve(I, "exact").energy)
    assert r.info["approximation_ratio"] > 0.75


@pytest.mark.parametrize("method", ["qpso", "pso"])
def test_continuous_solvers(method):
    sphere = Continuous(lambda X: np.sum((X - 1.5) ** 2, axis=1), -5 * np.ones(6), 5 * np.ones(6), vectorized=True)
    r = solve(sphere, method, iterations=300, seed=0)
    assert r.energy < 1e-4
    assert np.allclose(r.x, 1.5, atol=1e-2)


def test_legacy_protocol_now_optimises():
    state = RNG.uniform(-1, 1, 15)
    qop = QuantumOptimizationProtocol(state, scaling_factor=1.0, coupling_strength=0.7, seed=3)
    f = lambda x: -np.sum(x ** 2) + 0.1 * np.sum(np.sin(5 * x))
    res = qop.optimize(f, max_iterations=200)
    assert res["final_state"].shape == state.shape
    assert res["best_objective"] > f(state)
    hist = [h["objective"] for h in res["optimization_history"]]
    assert all(b >= a - 1e-12 for a, b in zip(hist, hist[1:]))  # monotone best-so-far


def test_bad_inputs():
    with pytest.raises(ValueError):
        Ising(np.zeros(3), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        solve(sherrington_kirkpatrick(30), "exact")
    with pytest.raises(ValueError):
        solve(sherrington_kirkpatrick(4), "nope")
