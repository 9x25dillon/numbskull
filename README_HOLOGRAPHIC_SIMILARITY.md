# Holographic Similarity Engine
## Deep Mathematical Structure for Cognitive Network Protocols

A sophisticated quantum-inspired similarity calculation system that implements the deep mathematical structure underlying emergent cognitive network protocols for calculating holographic similarity between query vectors and stored memo vectors.

### Mathematical Foundation

The system implements the mathematical framework:
```
⟨ ℰ | 𝕿𝖗𝖆𝖓𝖘𝖈𝖗𝖎𝖕𝖙𝖎𝖔𝖓 ⟩ → Ξ_cypherT
```

Where:
- `ℰ` represents the embedding space
- `𝕿𝖗𝖆𝖓𝖘𝖈𝖗𝖎𝖕𝖙𝖎𝖔𝖓` represents the cognitive transcription process
- `Ξ_cypherT` represents the cypher transformation engine

## 🌟 Features

### Core Components

- **Cognitive Network Protocol**: Implements cognitive state transcription and protocol layers
- **Cypher Transformation Engine (Ξ_cypherT)**: Quantum-inspired vector transformations
- **Holographic Similarity Engine**: Multi-method similarity calculations
- **Integration Layer**: Seamless integration with existing embedding pipeline

### Advanced Capabilities

- **Quantum-Inspired Mathematics**: Complex vector operations and quantum interference
- **Holographic Transformations**: Fractal-based embedding processing
- **Cognitive Resonance**: Frequency domain similarity analysis
- **Entanglement Measures**: Quantum entanglement-based similarity
- **Multi-Method Fusion**: Weighted combination of similarity methods
- **Parallel Processing**: Concurrent similarity calculations
- **Intelligent Caching**: Memory and computation optimization

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements_holographic.txt

# Run the demo
python holographic_similarity_engine.py
```

### Basic Usage

```python
import asyncio
import numpy as np
from holographic_similarity_engine import HolographicSimilarityEngine, HolographicConfig

async def main():
    # Create configuration
    config = HolographicConfig(
        quantum_dimension=512,
        holographic_depth=6,
        cognitive_layers=4
    )
    
    # Initialize engine
    engine = HolographicSimilarityEngine(config)
    
    # Create sample embeddings
    query_embedding = np.random.randn(512).astype(np.float32)
    memo_embeddings = [
        np.random.randn(512).astype(np.float32) for _ in range(5)
    ]
    
    # Calculate similarity
    result = await engine.calculate_holographic_similarity(
        query_embedding, memo_embeddings
    )
    
    print(f"Overall similarity: {result['overall_similarity']}")
    print(f"Found {len(result['similarities'])} similar memos")
    
    # Cleanup
    await engine.close()

# Run the example
asyncio.run(main())
```

### Integrated Usage

```python
import asyncio
from holographic_integration import HolographicQueryMemoSystem, HolographicIntegrationConfig

async def main():
    # Create configuration
    config = HolographicIntegrationConfig(
        quantum_dimension=512,
        holographic_depth=6,
        cognitive_layers=4,
        parallel_processing=True
    )
    
    # Initialize system
    system = HolographicQueryMemoSystem(config)
    
    try:
        # Add memos
        memo_texts = [
            "Quantum mechanics involves complex Hilbert spaces",
            "Fractal geometry reveals infinite complexity",
            "Neural networks learn hierarchical representations"
        ]
        await system.add_memo_batch(memo_texts)
        
        # Query for similar memos
        result = await system.query_similarity(
            "What is quantum mechanics?", 
            top_k=3
        )
        
        print(f"Found {len(result['similar_memos'])} similar memos:")
        for memo in result['similar_memos']:
            print(f"  Score: {memo['similarity_score']:.4f}")
            print(f"  Text: {memo['text']}")
    
    finally:
        await system.close()

# Run the example
asyncio.run(main())
```

## 🔧 Configuration

### Holographic Configuration

```python
from holographic_similarity_engine import HolographicConfig

config = HolographicConfig(
    # Quantum-inspired parameters
    quantum_dimension=1024,
    holographic_depth=8,
    cognitive_layers=6,
    
    # Cypher transformation parameters
    cypher_rotation_angle=np.pi / 4,
    cypher_scaling_factor=1.0,
    cypher_phase_shift=0.0,
    
    # Cognitive network parameters
    network_topology="fractal",
    protocol_layers=4,
    transcription_fidelity=0.95,
    
    # Similarity calculation parameters
    similarity_methods=[
        "quantum_cosine",
        "holographic_overlap", 
        "cognitive_resonance",
        "cypher_transformation",
        "fractal_similarity",
        "entanglement_measure"
    ],
    fusion_weights={
        "quantum_cosine": 0.20,
        "holographic_overlap": 0.25,
        "cognitive_resonance": 0.20,
        "cypher_transformation": 0.15,
        "fractal_similarity": 0.10,
        "entanglement_measure": 0.10
    },
    
    # Performance parameters
    parallel_processing=True,
    max_workers=4,
    cache_calculations=True,
    precision_threshold=1e-12
)
```

### Integration Configuration

```python
from holographic_integration import HolographicIntegrationConfig

config = HolographicIntegrationConfig(
    # Embedding pipeline settings
    use_semantic=True,
    use_mathematical=True,
    use_fractal=True,
    fusion_method="weighted_average",
    
    # Holographic similarity settings
    quantum_dimension=1024,
    holographic_depth=8,
    cognitive_layers=6,
    
    # Integration settings
    enable_caching=True,
    parallel_processing=True,
    max_workers=4,
    
    # Service URLs
    eopiez_url="http://localhost:8001",
    limps_url="http://localhost:8000"
)
```

## 📊 Similarity Methods

### 1. Quantum Cosine Similarity
- Complex vector dot product with magnitude normalization
- Captures quantum interference effects
- Weight: 20%

### 2. Holographic Overlap Similarity
- Overlap integral of holographically transformed vectors
- Captures holographic information encoding
- Weight: 25%

### 3. Cognitive Resonance Similarity
- Frequency domain resonance analysis
- Captures cognitive frequency patterns
- Weight: 20%

### 4. Cypher Transformation Similarity
- Similarity in cypher-transformed space
- Captures quantum transformation effects
- Weight: 15%

### 5. Fractal Similarity
- Fractal dimension comparison
- Captures self-similarity patterns
- Weight: 10%

### 6. Entanglement Measure Similarity
- Quantum entanglement measure
- Captures quantum correlations
- Weight: 10%

## 🧪 Testing

### Run Tests

```bash
# Run all tests
python test_holographic_system.py

# Run specific test categories
pytest test_holographic_system.py::TestHolographicSimilarityEngine
pytest test_holographic_system.py::TestHolographicIntegration
pytest test_holographic_system.py::TestPerformanceAndScalability
```

### Performance Benchmark

```python
from test_holographic_system import run_performance_benchmark

# Run performance benchmark
run_performance_benchmark()
```

### Comprehensive Demo

```python
from test_holographic_system import run_comprehensive_demo

# Run comprehensive demonstration
run_comprehensive_demo()
```

## 🔍 Advanced Features

### Custom Similarity Methods

```python
# Add custom similarity method
def custom_similarity(self, query: np.ndarray, memo: np.ndarray) -> float:
    # Custom similarity calculation
    return similarity_score

# Register custom method
engine._calculate_method_similarity = custom_similarity
```

### Custom Cognitive Protocols

```python
# Create custom cognitive protocol
class CustomCognitiveProtocol(CognitiveNetworkProtocol):
    def transcribe_cognitive_state(self, embedding: np.ndarray) -> np.ndarray:
        # Custom transcription logic
        return transcribed_embedding
```

### Custom Cypher Transformations

```python
# Create custom cypher transformation
class CustomCypherEngine(CypherTransformationEngine):
    def apply_cypher_transformation(self, vector: np.ndarray) -> np.ndarray:
        # Custom cypher transformation
        return transformed_vector
```

## 📈 Performance Optimization

### Memory Management

```python
# Monitor memory usage
import psutil

def check_memory_usage():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")
```

### Caching Strategy

```python
# Enable aggressive caching
config = HolographicConfig(
    cache_calculations=True,
    precision_threshold=1e-12
)

# Clear caches when needed
engine.clear_cache()
```

### Parallel Processing

```python
# Optimize parallel processing
config = HolographicConfig(
    parallel_processing=True,
    max_workers=8  # Adjust based on CPU cores
)
```

## 🛠️ Integration with Existing Systems

### Embedding Pipeline Integration

The holographic similarity engine integrates seamlessly with the existing advanced embedding pipeline:

```python
from advanced_embedding_pipeline import HybridEmbeddingPipeline
from holographic_similarity_engine import HolographicSimilarityEngine

# Use existing pipeline for embeddings
pipeline = HybridEmbeddingPipeline(config)
embeddings = await pipeline.embed_batch(texts)

# Use holographic engine for similarity
holographic_engine = HolographicSimilarityEngine(config)
similarities = await holographic_engine.calculate_holographic_similarity(
    query_embedding, memo_embeddings
)
```

### Database Integration

```python
# Store and retrieve embeddings with holographic similarity
import asyncpg

async def store_with_holographic_similarity(embeddings, similarities):
    conn = await asyncpg.connect("postgresql://user:pass@localhost/db")
    
    for i, (embedding, similarity) in enumerate(zip(embeddings, similarities)):
        await conn.execute(
            "INSERT INTO holographic_embeddings (id, embedding, similarity_score, metadata) VALUES ($1, $2, $3, $4)",
            i,
            embedding.tobytes(),
            similarity['similarity_score'],
            json.dumps(similarity['method_scores'])
        )
```

## 🔧 Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce `quantum_dimension` in configuration
   - Enable disk caching
   - Use memory mapping for large datasets

2. **Performance Issues**
   - Enable parallel processing
   - Adjust `max_workers` based on CPU cores
   - Use smaller `holographic_depth` for faster processing

3. **Precision Issues**
   - Adjust `precision_threshold` in configuration
   - Check for numerical stability in calculations
   - Use higher precision data types if needed

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
engine = HolographicSimilarityEngine(config)
```

## 📚 API Reference

### HolographicSimilarityEngine

- `calculate_holographic_similarity(query, memos, texts=None)`: Calculate similarity
- `get_metrics()`: Get performance metrics
- `clear_cache()`: Clear calculation cache
- `get_cache_stats()`: Get cache statistics

### HolographicQueryMemoSystem

- `add_memo(text, metadata=None)`: Add single memo
- `add_memo_batch(texts, metadata_list=None)`: Add multiple memos
- `query_similarity(query, top_k=5, threshold=0.0)`: Query similar memos
- `get_memo(index)`: Get specific memo
- `update_memo(index, text, metadata=None)`: Update memo
- `delete_memo(index)`: Delete memo
- `get_system_stats()`: Get system statistics
- `clear_all_caches()`: Clear all caches
- `close()`: Close system

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is part of the Fractal Cascade Simulation system. See the main project license for details.

## 🙏 Acknowledgments

- Quantum mechanics research community
- Fractal mathematics research
- Cognitive science research
- Machine learning and AI research
- Open source scientific computing libraries

---

**Holographic Similarity Engine** - Implementing the deep mathematical structure underlying emergent cognitive network protocols for advanced AI similarity calculations.

## Mathematical Notation Reference

- `⟨ ℰ | 𝕿𝖗𝖆𝖓𝖘𝖈𝖗𝖎𝖕𝖙𝖎𝖔𝖓 ⟩`: Bra-ket notation for cognitive state transcription
- `Ξ_cypherT`: Cypher transformation operator
- `ℰ`: Embedding space
- `𝕿𝖗𝖆𝖓𝖘𝖈𝖗𝖎𝖕𝖙𝖎𝖔𝖓`: Cognitive transcription process
- `→`: Transformation operator
- `⟨ | ⟩`: Inner product notation
- `Ξ`: Transformation operator
- `cypherT`: Cypher transformation subscript