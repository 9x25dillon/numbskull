# Emergent Cognitive Network - Advanced Mathematical Abstraction

A sophisticated implementation of quantum-inspired optimization protocols, swarm intelligence, neuromorphic processing, and holographic memory systems for emergent cognitive networks.

## Overview
# Enhanced Dual LLM WaveCaster with TA ULS Integration

A sophisticated system combining Two-level Trans-Algorithmic Universal Learning System (TA ULS) architecture with dual LLM orchestration, neuro-symbolic adaptive reflection, and advanced signal processing for intelligent waveform generation.

## 🚀 Features

This project implements a comprehensive framework for orchestrating emergent technologies using advanced mathematical abstractions inspired by quantum mechanics, cognitive science, and complex systems theory. The system integrates multiple protocols to create a unified emergent cognitive network capable of adaptive learning and pattern formation.

1. **TA ULS Transformer Architecture** (`tauls_transformer.py`)
   - Kinetic Force Principle (KFP) layers for gradient-based optimization
   - Two-level control system (meta-control + automatic control)
   - Entropy regulation based on environmental stress
   - Enhanced transformer blocks with stability monitoring

2. **Dual LLM Orchestration** (`dual_llm_orchestrator.py`)
   - Local LLM for final inference and decision making
   - Remote LLM for resource-only summarization
   - Intelligent coordination between systems
   - Multiple backend support (OpenAI, llama.cpp, TextGen WebUI)

3. **Neuro-Symbolic Adaptive Engine** (`neuro_symbolic_engine.py`)
   - Multiple analytical modules (entropy, reflection, matrix transformation)
   - Feature extraction and neural-symbolic fusion
   - Reinforcement learning for adaptive decision making
   - Reflective database for self-tuning and memory

4. **Advanced Signal Processing** (`signal_processing.py`)
   - Multiple modulation schemes (BFSK, BPSK, QPSK, QAM16, OFDM, DSSS)
   - Forward Error Correction (Hamming, Reed-Solomon, LDPC, Turbo)
   - Framing, security (AES-GCM), and watermarking
   - Audio and IQ signal generation with visualization

5. **Integrated System** (`enhanced_wavecaster.py`)
   - Comprehensive CLI interface
   - Configuration management
   - Component integration and orchestration

## 📦 Installation

### Requirements

```bash
# Core dependencies (required)
pip install numpy scipy torch

# Optional dependencies for full functionality
pip install matplotlib sounddevice soundfile requests pycryptodome

# Or install all at once
pip install -r requirements.txt
```

### Quick Setup

```bash
git clone <repository>
cd enhanced-wavecaster
pip install -r requirements.txt
```

## 🎯 Quick Start

### 1. Direct Text Modulation

```bash
# Basic QPSK modulation
python enhanced_wavecaster.py modulate --text "Hello, World!" --scheme qpsk --wav

# With security features
python enhanced_wavecaster.py modulate \
    --text "Secure message" \
    --scheme ofdm \
    --password "secret123" \
    --watermark "my_watermark" \
    --fec hamming74 \
    --wav --iq
```

### 2. LLM-Orchestrated Casting

```bash
# Using local LLM (llama.cpp server)
python enhanced_wavecaster.py cast \
    --prompt "Summarize the key technical points" \
    --resource-file document.txt \
    --scheme qpsk \
    --local-url http://localhost:8080 \
    --adaptive \
    --wav

# Using remote LLM with local fallback
python enhanced_wavecaster.py cast \
    --prompt "Create a technical brief" \
    --resource-file specs.pdf \
    --resource-text "Additional context here" \
    --remote-url https://api.openai.com \
    --remote-key $OPENAI_API_KEY \
    --scheme ofdm \
    --adaptive
```

### 3. Adaptive Learning

```bash
# Train the adaptive system
python enhanced_wavecaster.py learn \
    --texts "Message 1" "Message 2" "Message 3" \
    --episodes 20 \
    --db-path learning_db.json
```

### 4. Component Demonstrations

```bash
# Demo all components
python enhanced_wavecaster.py demo --component all

# Demo specific components
python enhanced_wavecaster.py demo --component tauls
python enhanced_wavecaster.py demo --component neuro-symbolic
python enhanced_wavecaster.py demo --component signal-processing
```

### 5. Text Analysis

```bash
# Analyze text with neuro-symbolic engine
python enhanced_wavecaster.py analyze \
    --text "Complex technical document content..." \
    --plot
```

## 🔧 Configuration

### Configuration File

Create a JSON configuration file:

```json
{
  "db_path": "reflective_db.json",
  "llm": {
    "local": [
      {
        "base_url": "http://127.0.0.1:8080",
        "mode": "llama-cpp",
        "model": "local-model"
      }
    ],
    "remote": {
      "base_url": "https://api.openai.com",
      "api_key": "your-api-key",
      "model": "gpt-4o-mini"
    },
    "settings": {
      "temperature": 0.7,
      "max_tokens": 512,
      "style": "concise"
    }
  },
  "modulation": {
    "sample_rate": 48000,
    "symbol_rate": 1200,
    "amplitude": 0.7
  },
  "security": {
    "password": null,
    "watermark": null,
    "hmac_key": null
  }
}
```

Use with: `--config config.json`

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_system.py
```

Or use pytest:

```bash
pytest test_system.py -v
```

## 📊 Architecture Overview

```python
# Quantum Optimization
quantum_opt = QuantumOptimizationProtocol(input_data, scaling_factor=1.0)
quantum_results = quantum_opt.optimize(lambda x: -np.sum(x**2))

# Swarm Cognitive Processing
agents = [input_data + np.random.normal(0, 0.1, input_data.shape) for _ in range(5)]
swarm_cog = SwarmCognitiveProtocol(agents, phi=0.5)
swarm_results = swarm_cog.execute_swarm_protocol()

# Neuromorphic Dynamics
neural_field = np.random.uniform(-1, 1, input_data.shape)
theta_params = np.random.uniform(0, 1, 10)
neuromorphic = NeuromorphicDynamics(neural_field, theta_params)
neuromorphic_results = neuromorphic.execute_neuromorphic_protocol()
```

## Visualization

The system includes comprehensive visualization tools:

### Static Visualizations
- Quantum optimization dynamics
- Swarm intelligence patterns
- Neuromorphic spiking patterns
- Holographic encoding patterns
- Morphogenetic field evolution
- Emergence metrics dashboard

### Interactive Dashboard
- Real-time protocol monitoring
- Multi-dimensional data exploration
- Interactive parameter adjustment
- Performance metrics tracking

### Animation Support
- Quantum state evolution
- Pattern formation dynamics
- Network adaptation processes

## Example Results

The system generates comprehensive metrics including:

- **Quantum Entropy**: Information content of quantum states
- **Swarm Intelligence**: Collective intelligence measures
- **Neuromorphic Criticality**: Neural network criticality
- **Holographic Coherence**: Memory system coherence
- **Morphogenetic Convergence**: Pattern formation convergence

## Mathematical Background

The implementation is based on advanced mathematical concepts:

1. **Quantum Mechanics**: Superposition, entanglement, and unitary evolution
2. **Complex Systems**: Emergence, self-organization, and criticality
3. **Information Theory**: Entropy, mutual information, and holographic principles
4. **Dynamical Systems**: Attractors, bifurcations, and pattern formation
5. **Cognitive Science**: Neural networks, learning, and adaptation

## Performance Considerations

- **Scalability**: Optimized for large-scale data processing
- **Memory Efficiency**: Efficient memory usage for holographic storage
- **Computational Complexity**: Polynomial time complexity for most operations
- **Parallel Processing**: Support for distributed computing

## Future Extensions

- **Quantum Machine Learning**: Integration with quantum ML algorithms
- **Distributed Computing**: Multi-node processing capabilities
- **Real-time Processing**: Streaming data processing support
- **Hardware Acceleration**: GPU and quantum hardware integration

## Contributing

Contributions are welcome! Please see the contributing guidelines for details on:
- Code style and standards
- Testing requirements
- Documentation standards
- Pull request process
MIT License - see LICENSE file for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
This system integrates concepts from:
- Transformer architectures and attention mechanisms
- Neuro-symbolic AI and hybrid reasoning systems
- Digital signal processing and communication theory
- Reinforcement learning and adaptive systems
- Information theory and error correction coding

## Citation

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the test suite for usage examples
- Review the comprehensive docstrings in each module

```bibtex
@software{emergent_cognitive_network,
  title={Emergent Cognitive Network - Advanced Mathematical Abstraction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/emergent-cognitive-network}
}
```

## Acknowledgments

- Quantum computing community for theoretical foundations
- Complex systems researchers for emergence principles
- Cognitive science community for neural network insights
- Open source contributors for supporting libraries
