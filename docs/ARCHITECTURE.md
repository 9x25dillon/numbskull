# Numbskull architecture (v0.2)

## 1. Summary

The repository is split into two independently installable packages with no dependency between them, plus the original prototype modules, which are kept for reference.

```
numbskull/
├── packages/
│   ├── wavecaster/      real-time software modem: FEC · modulation · PHY · radio I/O · link
│   └── emergentnet/     optimisation (SA/PT/SQA/QAOA/QPSO) · holographic RAG memory · GPU backend
├── docs/ARCHITECTURE.md
├── *.py (root)          legacy prototypes (TA-ULS, dual-LLM orchestrator, original protocols)
└── .github/workflows/ci.yml   tests both packages on Python 3.9–3.12
```

Dependency rule: `packages/*` import only numpy/scipy, plus optional extras loaded lazily (SoapySDR, uhd, sounddevice, pyserial, torch, cupy, sentence-transformers). Legacy modules may import packages; packages never import legacy modules.

## 2. wavecaster

### 2.1 Layering and contracts

```
            ┌────────────────────────── link.Link ───────────────────────────┐
            │   send(bytes) / frames() / send_message / stream_out|in         │
            │   RX thread: device.read(block) → Receiver.feed → queue        │
            └───────────────┬───────────────────────────────┬────────────────┘
                            │                               │
          phy.Transmitter   ▼                 phy.Receiver  ▼  (state machine)
   bytes→CRC32→FEC→scramble→Modem          SEARCH ──detect──► HEADER ──ok──► PAYLOAD
   header→conv→BPSK ┐                        ▲  (diff metric →   (Viterbi,     (Modem.demodulate
   preamble ────────┴→RRC                    │   local verify)    CRC16,        → FEC → CRC32)
                                             └──────── consume / emit Frame ◄──────┘
                            │                               ▲
                            ▼                               │
            ┌──────────── radio.RadioDevice (complex baseband) ──────────────┐
            │ Soapy · UHD · Audio(Passband+PTT) · File · Wav · Loopback      │
            └────────────────────────────────────────────────────────────────┘
```

| Interface | Contract |
|---|---|
| `BlockCodec` | `encode(bits)→coded`, `decode(llr, nbits)→DecodeResult`; fixed `k→n` blocks; LLR = ln P(0)/P(1) |
| `Constellation` | M points + bit-label permutation; `map`, `demap(y, N0, exact)` |
| `Modem` | `modulate(bits)` unit-power segment; `num_samples(nbits)` exact; `demodulate(seg, noise_var, nbits)→LLR` on aligned, CFO-corrected, gain-normalised input |
| `RadioDevice` | `transmit(iq)` one burst; `read(n, timeout)` ≤ n samples; `sample_rate` |
| Registries | `get_codec("ldpc:1024,0.5")`, `get_modem("ofdm:const=16qam")`, entry points `wavecaster.modems`/`.constellations`, `open_device("uhd:type=b200")` |

### 2.2 Acquisition and estimation

| Stage | Method | Cost / property |
|---|---|---|
| Candidate detection | differential correlation of `z[n+sps]·z*[n]` against `p_k p_{k−1}`, L1-normalised | 2 FFT correlations per block; CFO-invariant to ±Rs/2; noise floor ≈ 1.13/√L |
| Verification | segmented noncoherent metric Σ_p‖c_p‖/√(L·E), computed only around the candidate | bounded by 1 (Cauchy–Schwarz); CFO tolerance ∝ P |
| CFO | lag cascade 1, 4, 16, 64…, each resolving the previous residual | approaches the CRB for long preambles |
| Refinement | data-aided over preamble + re-encoded header | ~2× more known symbols, longer lags |
| Gain / noise | LS complex gain; residual variance → `noise_var`, SNR report | — |
| Payload tracking | SC: pilot every P symbols, residual-CFO ramp + W-pilot smoothing; OFDM: training symbol + CPE pilots | SNR loss ≈ 10 log₁₀(1+1/W) dB |

### 2.3 Complexity (per burst, N samples, E graph edges, K info bits)

| Component | Time | Memory |
|---|---|---|
| Search | O(N log N) (2 FFT correlations + local verification) | O(frame) |
| RS(n, k) | O(n·(n−k)) per codeword | O(n) |
| LDPC SPA | O(E·iterations), vectorised | O(E) |
| Turbo | O(K·S·iterations), S = 8 states | O(K·S) |
| Viterbi K = 7 | O(K·64) | O(K·64) |
| OFDM | O(N log nfft) | O(N) |

### 2.4 Failure modes and mitigations

| Failure | Detection | Mitigation |
|---|---|---|
| False preamble | header CRC-16 + version + profile byte | skip 1 symbol, keep searching |
| TX/RX configuration mismatch | `profile = crc8(modem\|fec)` | `stats["profile_mismatch"]` |
| Payload corruption | CRC-32 | frame reported `ok=False` (`--show-bad`), skipped as a whole |
| SDR overflow/underflow | driver return codes | counted in `device.stats`; stream continues |
| RX thread exception | `Link.error` | re-raised from `frames()` |
| Digital silence (exact zeros) | energy floor in metric | `stats["no_signal"]` |
| Clock drift on long bursts | none (see limits) | keep bursts short; next: Gardner timing loop |

## 3. emergentnet

```
optim/problems  Ising ⇄ QUBO, Continuous, builders
optim/annealing exact · SA · PT · SQA (PIMC)     ─┐
optim/qaoa      statevector QAOA, INTERP          ├─► solve(problem, method) → Result
optim/swarm     QPSO · PSO                        ─┘
memory/hrr      bind/unbind (FFT), Cleanup, AssociativeTrace, capacity()
memory/embed    HashingEmbedder · SentenceTransformerEmbedder · embed_stream
memory/store    HolographicStore: exact GPU top-k, soft/hard attribute queries, atomic persistence
accel           Backend(numpy | cupy | torch[cuda|mps|cpu]): topk (chunked), FFT, normalise
compat          working QuantumOptimizationProtocol (legacy API)
```

## 4. Evidence levels

| Claim | Level | Basis |
|---|---|---|
| FEC, modulation, sync and solver implementations are correct | 0 | textbook algorithms; tests against exact references (RS at the 2e+f ≤ n−k bound, exact Ising ground states, closed-form QAOA p=1) |
| Measured FER / throughput tables | 0 | reproducible via `wavecaster bench` and `emergentnet bench`, fixed seeds, simulated AWGN + CFO |
| Hardware drivers work on physical USRP/SoapySDR devices | 1 | written to the documented APIs; call sequences verified with fake SDK modules, but no bring-up on real radios yet |
| HRR composite vectors improve attribute-aware retrieval under ANN indexes | 1 | follows from quasi-orthogonality; demonstrated on small corpora only |
| SQA outperforms SA on specific problem classes | 2 | problem-dependent in the literature; use `emergentnet bench` per instance class |
| "Emergent" or cognitive interpretations of the legacy protocols | 3 | exploratory; not relied on by any package |

## 5. Roadmap (ordered by expected value)

1. **Hardware bring-up:** B200/B210 loopback through attenuators; `rx --show-bad` statistics; calibrate `tx_amplitude` for PA linearity (16-QAM/OFDM PAPR).
2. **Symbol timing recovery** (Gardner, or a polyphase filterbank as in GNU Radio's `pfb_clock_sync`) and SCO tracking, so bursts are no longer length-limited.
3. **ARQ + scheduler:** a selective-repeat layer on top of `transport` with ACK frames; TDD slotting via `UHDDevice(tx_delay=…)`.
4. **Adaptive coding and modulation:** choose the profile from `Frame.snr_db`, closing the loop with the legacy neuro-symbolic/RL engine as a policy.
5. **Performance:** move Viterbi/BCJR/PEG inner loops to numba or Rust (pyo3). The contracts above are the FFI boundary.
6. **emergentnet:** sparse-J kernels for n ≫ 10³; QAOA circuit export (Qiskit/Cirq) from `QAOASimulator`; FAISS/HNSW adapter over composite vectors.
7. **Open question:** the legacy `HolographicProtocol.recall_transform` compares a vector query with scalar field points (ill-defined). It is superseded by `emergentnet.memory`; it should be retired rather than patched.
