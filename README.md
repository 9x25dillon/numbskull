# Enhanced Dual LLM WaveCaster with TA ULS Integration

A sophisticated system combining Two-level Trans-Algorithmic Universal Learning System (TA ULS) architecture with dual LLM orchestration, neuro-symbolic adaptive reflection, and advanced signal processing for intelligent waveform generation.

## 🚀 Features

### Core Components

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

```
┌─────────────────────────────────────────────────────────────────┐
│                    Enhanced WaveCaster System                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   TA ULS        │  │  Dual LLM       │  │ Neuro-Symbolic  │  │
│  │  Transformer    │  │ Orchestrator    │  │   Engine        │  │
│  │                 │  │                 │  │                 │  │
│  │ • KFP Layers    │  │ • Local LLM     │  │ • Analytics     │  │
│  │ • Control Unit  │  │ • Remote LLM    │  │ • Feature Ext.  │  │
│  │ • Entropy Reg.  │  │ • Coordination  │  │ • RL Agent      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                │                                │
│  ┌─────────────────────────────┼─────────────────────────────┐  │
│  │            Signal Processing & Modulation                 │  │
│  │                                                           │  │
│  │ • BFSK/BPSK/QPSK/QAM16/OFDM/DSSS                        │  │
│  │ • FEC (Hamming/Reed-Solomon/LDPC/Turbo)                  │  │
│  │ • Security (AES-GCM/HMAC/Watermarking)                   │  │
│  │ • Audio/IQ Generation & Visualization                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔬 Technical Details

### TA ULS Architecture

The Two-level Trans-Algorithmic Universal Learning System implements:

- **Higher Level**: Meta-control for learning and adaptation
- **Lower Level**: Automatic control for real-time processing
- **KFP Layers**: Gradient-based optimization toward minimal fluctuation
- **Entropy Regulation**: Environmental stress-based parameter modulation

### Neuro-Symbolic Fusion

Combines neural features with symbolic metrics:

- **Neural Features**: N-gram hashing, embedding extraction
- **Symbolic Metrics**: Entropy, complexity, semantic density, harmony
- **RL Agent**: Contextual bandit for adaptive decision making
- **Reflective DB**: Self-tuning memory system

### Signal Processing Pipeline

```
Text → Encoding → FEC → Framing → Security → Modulation → Audio/IQ
  ↑                                                          ↓
Analysis ← Adaptive Planning ← Neuro-Symbolic Engine ← Feedback
```

## 📈 Performance Characteristics

### Modulation Schemes

| Scheme    | Spectral Efficiency | Complexity | Robustness |
|-----------|-------------------|------------|------------|
| BFSK      | Low               | Low        | High       |
| BPSK      | Medium            | Low        | High       |
| QPSK      | Medium            | Medium     | Medium     |
| QAM16     | High              | High       | Low        |
| OFDM      | High              | High       | Medium     |
| DSSS-BPSK | Low               | Medium     | Very High  |

### FEC Performance

| Scheme     | Code Rate | Error Correction | Complexity |
|------------|-----------|------------------|------------|
| None       | 1.0       | None            | Minimal    |
| Hamming74  | 4/7       | Single bit      | Low        |
| Reed-Solomon| Variable  | Burst errors    | Medium     |
| LDPC       | Variable  | Near capacity   | High       |
| Turbo      | Variable  | Near capacity   | Very High  |

## 🛠️ Development

### Project Structure

```
enhanced-wavecaster/
├── tauls_transformer.py       # TA ULS architecture
├── dual_llm_orchestrator.py   # LLM coordination
├── neuro_symbolic_engine.py   # Adaptive analytics
├── signal_processing.py       # Modulation & DSP
├── enhanced_wavecaster.py     # Main integration
├── test_system.py            # Comprehensive tests
├── requirements.txt          # Dependencies
└── README.md                # This file
```

### Adding New Components

1. **Modulation Schemes**: Extend `Modulators` class in `signal_processing.py`
2. **FEC Codes**: Add to `fec_encode`/`fec_decode` functions
3. **Analytics**: Add modules to `neuro_symbolic_engine.py`
4. **LLM Backends**: Extend `LocalLLM` class in `dual_llm_orchestrator.py`

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

This system integrates concepts from:
- Transformer architectures and attention mechanisms
- Neuro-symbolic AI and hybrid reasoning systems
- Digital signal processing and communication theory
- Reinforcement learning and adaptive systems
- Information theory and error correction coding

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the test suite for usage examples
- Review the comprehensive docstrings in each module

---

*Enhanced Dual LLM WaveCaster - Bridging AI and Signal Processing* 🚀