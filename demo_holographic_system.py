#!/usr/bin/env python3
"""
Holographic Similarity System Demonstration
Demonstrates the complete holographic similarity calculation system
with deep mathematical structure for cognitive network protocols.
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any
import time

# Import the holographic similarity system
from holographic_similarity_engine import HolographicSimilarityEngine, HolographicConfig
from holographic_integration import HolographicQueryMemoSystem, HolographicIntegrationConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_basic_holographic_engine():
    """Demonstrate the basic holographic similarity engine"""
    print("\n" + "="*60)
    print("🔮 HOLOGRAPHIC SIMILARITY ENGINE DEMONSTRATION")
    print("="*60)
    
    # Create configuration
    config = HolographicConfig(
        quantum_dimension=256,
        holographic_depth=4,
        cognitive_layers=3,
        parallel_processing=True
    )
    
    # Initialize engine
    engine = HolographicSimilarityEngine(config)
    print("✅ Holographic Similarity Engine initialized")
    
    # Create sample embeddings
    print("\n📊 Creating sample embeddings...")
    query_embedding = np.random.randn(256).astype(np.float32)
    memo_embeddings = [
        np.random.randn(256).astype(np.float32) for _ in range(5)
    ]
    memo_texts = [
        "Quantum mechanics: The mathematical framework for atomic and subatomic physics",
        "Fractal geometry: Mathematical structures exhibiting self-similarity at all scales",
        "Neural networks: Computational models inspired by biological neural systems",
        "Information theory: Mathematical study of information quantification and communication",
        "Machine learning: Algorithms that improve performance through experience"
    ]
    
    # Calculate holographic similarity
    print("🧮 Calculating holographic similarity...")
    start_time = time.time()
    
    result = await engine.calculate_holographic_similarity(
        query_embedding, memo_embeddings, memo_texts
    )
    
    calculation_time = time.time() - start_time
    
    # Display results
    print(f"\n📈 Results (calculated in {calculation_time:.3f}s):")
    print(f"Overall Similarity: {result['overall_similarity']}")
    print(f"Methods Used: {result['metadata']['methods_used']}")
    
    print(f"\n🔍 Individual Similarities:")
    for i, sim in enumerate(result['similarities']):
        print(f"  Memo {i+1}: {sim['similarity_score']:.4f}")
        if 'memo_text' in sim and sim['memo_text']:
            print(f"    Text: {sim['memo_text'][:60]}...")
        
        # Show method breakdown
        if 'method_scores' in sim:
            print(f"    Method scores:")
            for method, score in sim['method_scores'].items():
                print(f"      {method}: {score:.4f}")
        print()
    
    # Display metrics
    metrics = engine.get_metrics()
    print(f"📊 Performance Metrics:")
    print(f"  Total calculations: {metrics['total_calculations']}")
    print(f"  Average time: {metrics['average_calculation_time']:.3f}s")
    print(f"  Quantum cosine calculations: {metrics['quantum_cosine_calculations']}")
    print(f"  Holographic overlap calculations: {metrics['holographic_overlap_calculations']}")
    print(f"  Cognitive resonance calculations: {metrics['cognitive_resonance_calculations']}")
    
    return engine


async def demo_integrated_system():
    """Demonstrate the integrated holographic query-memo system"""
    print("\n" + "="*60)
    print("🔗 INTEGRATED HOLOGRAPHIC QUERY-MEMO SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Create configuration
    config = HolographicIntegrationConfig(
        quantum_dimension=256,
        holographic_depth=4,
        cognitive_layers=3,
        parallel_processing=True,
        use_semantic=False,  # Disable external services for demo
        use_mathematical=False,
        use_fractal=True
    )
    
    # Initialize system
    system = HolographicQueryMemoSystem(config)
    print("✅ Integrated Holographic Query-Memo System initialized")
    
    try:
        # Add diverse memos
        print("\n📝 Adding diverse memos to the system...")
        diverse_memos = [
            "Quantum mechanics involves complex Hilbert spaces and linear operators for describing atomic behavior",
            "Fractal geometry reveals infinite complexity through recursive self-similarity at different scales",
            "Neural networks learn hierarchical representations through backpropagation and gradient descent",
            "Information theory quantifies the amount of information in signals and communication systems",
            "Machine learning algorithms optimize objective functions to minimize prediction errors",
            "The holographic principle suggests information in a volume can be encoded on its boundary",
            "Cognitive science studies mind and intelligence through interdisciplinary approaches",
            "Complexity theory examines complex systems and their emergent properties",
            "Quantum entanglement enables non-local correlations between distant particles",
            "Deep learning uses multiple neural network layers to learn abstract features"
        ]
        
        start_time = time.time()
        memo_indices = await system.add_memo_batch(diverse_memos)
        addition_time = time.time() - start_time
        
        print(f"✅ Added {len(memo_indices)} memos in {addition_time:.3f}s")
        
        # Test various queries
        test_queries = [
            "What is quantum mechanics?",
            "How do fractals work?",
            "Explain neural networks",
            "What is information theory?",
            "How does machine learning work?",
            "What is the holographic principle?",
            "Explain cognitive science",
            "What is complexity theory?",
            "How does quantum entanglement work?",
            "What is deep learning?"
        ]
        
        print(f"\n🔍 Testing {len(test_queries)} different queries...")
        
        total_query_time = 0
        for i, query in enumerate(test_queries):
            print(f"\n--- Query {i+1}: {query} ---")
            
            start_time = time.time()
            result = await system.query_similarity(query, top_k=3, similarity_threshold=0.1)
            query_time = time.time() - start_time
            total_query_time += query_time
            
            print(f"Found {len(result['similar_memos'])} similar memos (calculated in {query_time:.3f}s):")
            for j, memo in enumerate(result['similar_memos']):
                print(f"  {j+1}. Score: {memo['similarity_score']:.4f}")
                print(f"     Text: {memo['text'][:70]}...")
            
            print(f"Overall similarity: {result['overall_similarity']}")
        
        avg_query_time = total_query_time / len(test_queries)
        print(f"\n📊 Query Performance:")
        print(f"  Average query time: {avg_query_time:.3f}s")
        print(f"  Total query time: {total_query_time:.3f}s")
        
        # Test edge cases
        print(f"\n🧪 Testing edge cases...")
        
        # Empty query
        empty_result = await system.query_similarity("", top_k=3)
        print(f"Empty query results: {len(empty_result['similar_memos'])} memos")
        
        # Very specific query
        specific_result = await system.query_similarity(
            "quantum entanglement non-local correlations particles", 
            top_k=5
        )
        print(f"Specific query results: {len(specific_result['similar_memos'])} memos")
        
        # Get system statistics
        print(f"\n📈 System Statistics:")
        stats = system.get_system_stats()
        print(f"  Total memos: {stats['memo_count']}")
        print(f"  Total queries: {stats['query_count']}")
        print(f"  Embedding metrics: {stats['embedding_metrics']}")
        print(f"  Holographic metrics: {stats['holographic_metrics']}")
        
    finally:
        await system.close()
        print("✅ System closed")


async def demo_mathematical_foundations():
    """Demonstrate the mathematical foundations of the system"""
    print("\n" + "="*60)
    print("🧮 MATHEMATICAL FOUNDATIONS DEMONSTRATION")
    print("="*60)
    
    # Create configuration
    config = HolographicConfig(
        quantum_dimension=64,
        holographic_depth=3,
        cognitive_layers=2,
        parallel_processing=False
    )
    
    # Initialize engine
    engine = HolographicSimilarityEngine(config)
    
    # Test individual similarity methods
    print("🔬 Testing individual similarity methods...")
    
    # Create test vectors
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    memo = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    
    print(f"Query vector: {query}")
    print(f"Memo vector: {memo}")
    print()
    
    # Test each similarity method
    methods = [
        ("Quantum Cosine Similarity", engine._quantum_cosine_similarity),
        ("Holographic Overlap Similarity", engine._holographic_overlap_similarity),
        ("Cognitive Resonance Similarity", engine._cognitive_resonance_similarity),
        ("Cypher Transformation Similarity", engine._cypher_transformation_similarity),
        ("Fractal Similarity", engine._fractal_similarity),
        ("Entanglement Measure Similarity", engine._entanglement_measure_similarity)
    ]
    
    for method_name, method_func in methods:
        try:
            similarity = method_func(query, memo)
            print(f"{method_name}: {similarity:.4f}")
        except Exception as e:
            print(f"{method_name}: Error - {e}")
    
    print(f"\n🔍 Testing cognitive state transcription...")
    
    # Test cognitive protocol
    from holographic_similarity_engine import CognitiveNetworkProtocol
    protocol = CognitiveNetworkProtocol(config)
    
    original_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    transcribed_vector = protocol.transcribe_cognitive_state(original_vector)
    
    print(f"Original vector: {original_vector}")
    print(f"Transcribed vector: {transcribed_vector}")
    print(f"Transcription fidelity: {np.linalg.norm(transcribed_vector):.4f}")
    
    print(f"\n🔧 Testing cypher transformation...")
    
    # Test cypher engine
    from holographic_similarity_engine import CypherTransformationEngine
    cypher_engine = CypherTransformationEngine(config)
    
    original_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    transformed_vector = cypher_engine.apply_cypher_transformation(original_vector)
    
    print(f"Original vector: {original_vector}")
    print(f"Transformed vector: {transformed_vector}")
    print(f"Transformation magnitude: {np.linalg.norm(transformed_vector):.4f}")


async def demo_performance_benchmark():
    """Demonstrate performance benchmarking"""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE BENCHMARK DEMONSTRATION")
    print("="*60)
    
    # Test different configurations
    configurations = [
        ("Small", HolographicConfig(quantum_dimension=64, holographic_depth=2, cognitive_layers=2)),
        ("Medium", HolographicConfig(quantum_dimension=256, holographic_depth=4, cognitive_layers=3)),
        ("Large", HolographicConfig(quantum_dimension=512, holographic_depth=6, cognitive_layers=4))
    ]
    
    for config_name, config in configurations:
        print(f"\n🔧 Testing {config_name} configuration...")
        
        engine = HolographicSimilarityEngine(config)
        
        # Create test data
        query_embedding = np.random.randn(config.quantum_dimension).astype(np.float32)
        memo_embeddings = [
            np.random.randn(config.quantum_dimension).astype(np.float32) 
            for _ in range(10)
        ]
        
        # Benchmark similarity calculation
        start_time = time.time()
        result = await engine.calculate_holographic_similarity(query_embedding, memo_embeddings)
        calculation_time = time.time() - start_time
        
        print(f"  Configuration: {config_name}")
        print(f"  Quantum dimension: {config.quantum_dimension}")
        print(f"  Holographic depth: {config.holographic_depth}")
        print(f"  Cognitive layers: {config.cognitive_layers}")
        print(f"  Calculation time: {calculation_time:.3f}s")
        print(f"  Memos processed: {len(memo_embeddings)}")
        print(f"  Time per memo: {calculation_time/len(memo_embeddings)*1000:.1f}ms")
        
        # Get metrics
        metrics = engine.get_metrics()
        print(f"  Total calculations: {metrics['total_calculations']}")
        print(f"  Average time: {metrics['average_calculation_time']:.3f}s")


async def main():
    """Main demonstration function"""
    print("🎭 HOLOGRAPHIC SIMILARITY SYSTEM COMPREHENSIVE DEMONSTRATION")
    print("Implementing deep mathematical structure for cognitive network protocols")
    print("Mathematical Foundation: ⟨ ℰ | 𝕿𝖗𝖆𝖓𝖘𝖈𝖗𝖎𝖕𝖙𝖎𝖔𝖓 ⟩ → Ξ_cypherT")
    
    try:
        # Demo 1: Basic holographic engine
        await demo_basic_holographic_engine()
        
        # Demo 2: Integrated system
        await demo_integrated_system()
        
        # Demo 3: Mathematical foundations
        await demo_mathematical_foundations()
        
        # Demo 4: Performance benchmark
        await demo_performance_benchmark()
        
        print("\n" + "="*60)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe holographic similarity system successfully implements:")
        print("• Quantum-inspired mathematical similarity calculations")
        print("• Cognitive network protocol transcriptions")
        print("• Cypher transformation engine (Ξ_cypherT)")
        print("• Multi-method similarity fusion")
        print("• Integration with existing embedding pipeline")
        print("• Comprehensive testing and benchmarking")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    # Run the comprehensive demonstration
    asyncio.run(main())