"""emergentnet: optimisation and holographic memory, usable standalone.

    optim/    Ising/QUBO/continuous problems; exact, SA, parallel tempering,
              simulated quantum annealing (PIMC), QAOA (statevector), QPSO, PSO
    memory/   HRR binding algebra, embedders, persistent RAG vector store
    accel     NumPy / CuPy / PyTorch (CUDA, MPS) backend selection
    compat    drop-in working replacement for the legacy QuantumOptimizationProtocol
"""

__version__ = "0.2.0"
