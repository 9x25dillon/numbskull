# emergentnet

Standalone optimisation and holographic-memory library. It has no dependency on wavecaster or on the legacy modules.

```
pip install -e packages/emergentnet                  # numpy + scipy
pip install -e "packages/emergentnet[gpu-torch]"     # GPU search via PyTorch (CUDA/MPS)
pip install -e "packages/emergentnet[embeddings]"    # sentence-transformers embedder
```

## Optimisation (`emergentnet.optim`)

Problems: `Ising(h, J)`, `QUBO(Q)` (exactly inter-convertible, `x = (1+s)/2`), `Continuous(f, lower, upper)`. Builders: `maxcut`, `maxcut_from_edges`, `number_partitioning`, `sherrington_kirkpatrick`, `QUBO.from_dict` (dimod-style `{(i, j): w}`).

| Method | Algorithm | Scale |
|---|---|---|
| `exact` | chunked brute-force enumeration (ground truth) | n ≤ 26 |
| `sa` | Metropolis simulated annealing, geometric β schedule (neal heuristic), R reads vectorised | dense n ≲ 2000 |
| `pt` | replica-exchange (parallel tempering) | dense n ≲ 2000 |
| `sqa` | simulated quantum annealing: path-integral Monte Carlo of the transverse-field Ising model, P Trotter slices, J⊥ = −(PT/2) ln tanh(Γ/PT) | dense n ≲ 1000 |
| `qaoa` | exact statevector QAOA, depth p, INTERP layer-growth initialisation, L-BFGS-B | n ≤ 22 qubits |
| `qpso` / `pso` | quantum-behaved PSO (Sun et al. 2004) / inertia-weight PSO | continuous |

```python
from emergentnet.optim import maxcut_from_edges, QUBO, Continuous, solve
r = solve(maxcut_from_edges([(0, 1), (1, 2), (2, 0), (2, 3)]), "sqa", seed=0)   # r.energy == -cut
r = solve(QUBO.from_dict({(0, 0): -1, (1, 1): -1, (0, 1): 2}), "exact")        # r.x are bits
r = solve(Continuous(f, lo, hi), "qpso", iterations=500)
```
```bash
emergentnet solve --maxcut edges.txt --method pt
emergentnet bench --sk 16 --methods exact sa pt sqa qaoa --seed 2
```

Verified (tests + `bench`, one CPU core):

| Instance | Result |
|---|---|
| SK n=16, 18 | `sa`, `pt`, `sqa` reach the exact ground state (0.3–0.4 s) |
| SK n=16, QAOA p=2 | samples the exact ground state (9 s, statevector 2¹⁶) |
| SK n=150 | `sa`/`pt`/`sqa` all reach −0.742 per spin in 300 sweeps (thermodynamic-limit Parisi value is −0.763) |
| 3-regular MaxCut n=24 | all three MC methods match exact (32/36 edges) |
| QAOA p=1, single edge | ⟨C⟩ matches the closed form −½ + ½ sin 4β sin γ to 1e-12 |

**What "quantum" means here.** SQA and QAOA are classical simulations of quantum algorithms (Level 0: established algorithms, computed exactly or by standard Monte Carlo). They do not run on quantum hardware and give no quantum speedup. QAOA's statevector is exact, so results are a faithful reference for what the algorithm would produce on an ideal device. The `QAOASimulator.cost` / `.state` API is the integration point for exporting circuits to real backends.

**Legacy compatibility.** `emergentnet.compat.QuantumOptimizationProtocol` keeps the legacy constructor and `optimize(objective, max_iterations)` return keys, and actually optimises (QPSO, maximising by default like the legacy call sites). With this package installed, the legacy `emergent_cognitive_network*.py` modules rebind to it automatically.

## Holographic memory for RAG (`emergentnet.memory`)

```python
from emergentnet.memory import HolographicStore
kb = HolographicStore(embedder="st:sentence-transformers/all-MiniLM-L6-v2",   # or "hashing:1024" offline
                      attributes=("source", "author"), backend="auto")         # torch-CUDA > cupy > MPS > numpy
kb.ingest(texts, batch_size=512, metadata=metas)                             # streaming, bounded memory
hits = kb.search("how do turbo decoders iterate?", k=5)                      # exact inner product on GPU
hits = kb.search("decoders", k=5, where={"author": "carol"})                 # soft HRR boost (one dot product)
hits = kb.search("decoders", k=5, where={"source": "spec.pdf"}, mode="hard")  # exact metadata filter
prompt_context = HolographicStore.format_context(hits)                        # "[id] text" blocks for an LLM
kb.recover_attribute(hits[0].id, "author")                                   # decoded from the vector alone
kb.save("kb/"); kb = HolographicStore.load("kb/")                             # atomic directory format
```

Design: each record keeps its content embedding and, for configured attributes, a composite `normalize(content + w·Σ bind(role_a, filler(a, v)))`. Circular-convolution binding (HRR, Plate 1995) makes the attribute terms quasi-orthogonal to content and to each other (|cos| ~ 1/√d). As a result:

- A soft query adds ≈ w²/(‖q‖‖r‖) to every record matching the attribute, in a single inner product. It works with any ANN index, including ones without metadata filtering. It is a boost, not a filter: at w = 0.5 a strong content match can still outrank it, and at w = 1 it behaves like a filter in tests.
- `unbind(r, role_a)` followed by cleanup recovers the attribute value (100% on test data at d = 1024).
- `AssociativeTrace` stores key→value pairs in O(d) memory. Measured recall with a 64-value codebook, d ∈ {512, 1024, 2048}: 100% at `capacity(d, M)` = d/(8 ln M), 94–98% at twice that load, and ~70% at four times.

`accel.get_backend()` runs the top-k scan on the GPU in bounded-memory chunks and returns NumPy. `SentenceTransformerEmbedder` picks CUDA → MPS → CPU and uses fp16 on CUDA. The hashing embedder is a deterministic lexical baseline for offline use and tests, not a semantic model.

Scale notes: search is exact, O(N·d) per query. Measured on this repo's CI-class CPU with 500k × 384 float32 (768 MB): 22 ms per single query (numpy), and 5.2 ms (numpy) or 2.2 ms (torch-CPU) per query in batches of 32. Cost is linear in N, and a CUDA GPU moves the scan to device memory bandwidth. Past ~10⁷ vectors, feed the content or composite vectors into FAISS/HNSW; the composite design exists so that attribute-aware retrieval survives that switch.
