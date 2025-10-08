#!/usr/bin/env python3
"""
Comprehensive Test Suite for Holographic Similarity System
Tests the holographic similarity engine and integration layer
"""

import asyncio
import logging
import numpy as np
import pytest
import time
from typing import List, Dict, Any
import json
from pathlib import Path

# Import the modules to test
from holographic_similarity_engine import (
    HolographicSimilarityEngine, 
    HolographicConfig,
    CognitiveNetworkProtocol,
    CypherTransformationEngine
)
from holographic_integration import (
    HolographicQueryMemoSystem,
    HolographicIntegrationConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestHolographicSimilarityEngine:
    """Test suite for the holographic similarity engine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = HolographicConfig(
            quantum_dimension=256,
            holographic_depth=4,
            cognitive_layers=3,
            parallel_processing=False  # Use sequential for testing
        )
        self.engine = HolographicSimilarityEngine(self.config)
        
        # Create test embeddings
        self.query_embedding = np.random.randn(256).astype(np.float32)
        self.memo_embeddings = [
            np.random.randn(256).astype(np.float32) for _ in range(5)
        ]
        self.memo_texts = [
            "Test memo 1: Quantum mechanics",
            "Test memo 2: Fractal geometry", 
            "Test memo 3: Neural networks",
            "Test memo 4: Information theory",
            "Test memo 5: Machine learning"
        ]
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        assert self.engine is not None
        assert self.engine.config.quantum_dimension == 256
        assert self.engine.config.holographic_depth == 4
        assert self.engine.config.cognitive_layers == 3
    
    def test_cognitive_protocol_initialization(self):
        """Test cognitive network protocol initialization"""
        protocol = CognitiveNetworkProtocol(self.config)
        assert protocol is not None
        assert protocol.config.protocol_layers == 4
    
    def test_cypher_engine_initialization(self):
        """Test cypher transformation engine initialization"""
        cypher_engine = CypherTransformationEngine(self.config)
        assert cypher_engine is not None
        assert cypher_engine.config.cypher_rotation_angle == np.pi / 4
    
    def test_cognitive_state_transcription(self):
        """Test cognitive state transcription"""
        protocol = CognitiveNetworkProtocol(self.config)
        
        # Test with simple vector
        test_vector = np.array([1.0, 0.0, 0.0, 0.0])
        transcribed = protocol.transcribe_cognitive_state(test_vector)
        
        assert transcribed is not None
        assert len(transcribed) == len(test_vector)
        assert np.isfinite(transcribed).all()
    
    def test_cypher_transformation(self):
        """Test cypher transformation"""
        cypher_engine = CypherTransformationEngine(self.config)
        
        # Test with simple vector
        test_vector = np.array([1.0, 0.0, 0.0, 0.0])
        transformed = cypher_engine.apply_cypher_transformation(test_vector)
        
        assert transformed is not None
        assert len(transformed) == len(test_vector)
        assert np.isfinite(transformed).all()
    
    def test_quantum_cosine_similarity(self):
        """Test quantum cosine similarity calculation"""
        query = np.array([1.0, 0.0, 0.0])
        memo = np.array([0.0, 1.0, 0.0])
        
        similarity = self.engine._quantum_cosine_similarity(query, memo)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_holographic_overlap_similarity(self):
        """Test holographic overlap similarity calculation"""
        query = np.array([1.0, 0.5, 0.0])
        memo = np.array([0.5, 1.0, 0.0])
        
        similarity = self.engine._holographic_overlap_similarity(query, memo)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_cognitive_resonance_similarity(self):
        """Test cognitive resonance similarity calculation"""
        query = np.array([1.0, 0.0, 0.0])
        memo = np.array([0.0, 1.0, 0.0])
        
        similarity = self.engine._cognitive_resonance_similarity(query, memo)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_cypher_transformation_similarity(self):
        """Test cypher transformation similarity calculation"""
        query = np.array([1.0, 0.0, 0.0])
        memo = np.array([0.0, 1.0, 0.0])
        
        similarity = self.engine._cypher_transformation_similarity(query, memo)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_fractal_similarity(self):
        """Test fractal similarity calculation"""
        query = np.array([1.0, 0.0, 0.0])
        memo = np.array([0.0, 1.0, 0.0])
        
        similarity = self.engine._fractal_similarity(query, memo)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_entanglement_measure_similarity(self):
        """Test entanglement measure similarity calculation"""
        query = np.array([1.0, 0.0, 0.0])
        memo = np.array([0.0, 1.0, 0.0])
        
        similarity = self.engine._entanglement_measure_similarity(query, memo)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    @pytest.mark.asyncio
    async def test_single_similarity_calculation(self):
        """Test single similarity calculation"""
        query = np.array([1.0, 0.0, 0.0, 0.0])
        memo = np.array([0.0, 1.0, 0.0, 0.0])
        
        result = await self.engine._calculate_single_similarity(
            query, memo, 0, "Test memo"
        )
        
        assert result is not None
        assert "similarity_score" in result
        assert "method_scores" in result
        assert isinstance(result["similarity_score"], float)
        assert 0.0 <= result["similarity_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_holographic_similarity_calculation(self):
        """Test full holographic similarity calculation"""
        result = await self.engine.calculate_holographic_similarity(
            self.query_embedding,
            self.memo_embeddings,
            self.memo_texts
        )
        
        assert result is not None
        assert "query_embedding" in result
        assert "transcribed_query" in result
        assert "similarities" in result
        assert "overall_similarity" in result
        assert "metadata" in result
        
        assert len(result["similarities"]) == len(self.memo_embeddings)
        
        for sim in result["similarities"]:
            assert "similarity_score" in sim
            assert "method_scores" in sim
            assert isinstance(sim["similarity_score"], float)
            assert 0.0 <= sim["similarity_score"] <= 1.0
    
    def test_metrics_tracking(self):
        """Test metrics tracking"""
        initial_metrics = self.engine.get_metrics()
        assert "total_calculations" in initial_metrics
        assert "cache_hits" in initial_metrics
        assert initial_metrics["total_calculations"] == 0
    
    def test_cache_operations(self):
        """Test cache operations"""
        # Test cache stats
        cache_stats = self.engine.get_cache_stats()
        assert "similarity_cache_size" in cache_stats
        assert "protocol_cache_size" in cache_stats
        assert "cypher_cache_size" in cache_stats
        
        # Test cache clearing
        self.engine.clear_cache()
        cache_stats_after = self.engine.get_cache_stats()
        assert cache_stats_after["similarity_cache_size"] == 0


class TestHolographicIntegration:
    """Test suite for the holographic integration layer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = HolographicIntegrationConfig(
            quantum_dimension=256,
            holographic_depth=4,
            cognitive_layers=3,
            parallel_processing=False,  # Use sequential for testing
            use_semantic=False,  # Disable external services for testing
            use_mathematical=False,
            use_fractal=True  # Only use fractal for testing
        )
        self.system = HolographicQueryMemoSystem(self.config)
    
    @pytest.mark.asyncio
    async def test_system_initialization(self):
        """Test system initialization"""
        assert self.system is not None
        assert self.system.embedding_pipeline is not None
        assert self.system.holographic_engine is not None
        assert len(self.system.memo_embeddings) == 0
    
    @pytest.mark.asyncio
    async def test_add_single_memo(self):
        """Test adding a single memo"""
        memo_text = "Test memo: Quantum mechanics and fractals"
        memo_index = await self.system.add_memo(memo_text)
        
        assert memo_index == 0
        assert len(self.system.memo_embeddings) == 1
        assert self.system.memo_texts[0] == memo_text
        assert self.system.memo_metadata[0] == {}
    
    @pytest.mark.asyncio
    async def test_add_memo_batch(self):
        """Test adding multiple memos"""
        memo_texts = [
            "Test memo 1: Quantum mechanics",
            "Test memo 2: Fractal geometry",
            "Test memo 3: Neural networks"
        ]
        
        memo_indices = await self.system.add_memo_batch(memo_texts)
        
        assert len(memo_indices) == 3
        assert len(self.system.memo_embeddings) == 3
        assert self.system.memo_texts == memo_texts
    
    @pytest.mark.asyncio
    async def test_query_similarity(self):
        """Test query similarity calculation"""
        # Add test memos
        memo_texts = [
            "Quantum mechanics involves complex Hilbert spaces",
            "Fractal geometry reveals infinite complexity",
            "Neural networks learn hierarchical representations"
        ]
        await self.system.add_memo_batch(memo_texts)
        
        # Test query
        query_text = "What is quantum mechanics?"
        result = await self.system.query_similarity(query_text, top_k=2)
        
        assert result is not None
        assert "query_text" in result
        assert "similar_memos" in result
        assert "overall_similarity" in result
        assert "metadata" in result
        
        assert len(result["similar_memos"]) <= 2
        assert result["query_text"] == query_text
    
    @pytest.mark.asyncio
    async def test_get_memo(self):
        """Test getting a specific memo"""
        memo_text = "Test memo: Information theory"
        memo_index = await self.system.add_memo(memo_text)
        
        memo = await self.system.get_memo(memo_index)
        
        assert memo is not None
        assert memo["memo_index"] == memo_index
        assert memo["text"] == memo_text
        assert "embedding" in memo
    
    @pytest.mark.asyncio
    async def test_update_memo(self):
        """Test updating a memo"""
        # Add initial memo
        memo_text = "Original memo text"
        memo_index = await self.system.add_memo(memo_text)
        
        # Update memo
        new_text = "Updated memo text"
        success = await self.system.update_memo(memo_index, new_text)
        
        assert success
        assert self.system.memo_texts[memo_index] == new_text
    
    @pytest.mark.asyncio
    async def test_delete_memo(self):
        """Test deleting a memo"""
        # Add memo
        memo_text = "Memo to delete"
        memo_index = await self.system.add_memo(memo_text)
        
        # Delete memo
        success = await self.system.delete_memo(memo_index)
        
        assert success
        assert len(self.system.memo_embeddings) == 0
    
    def test_system_stats(self):
        """Test system statistics"""
        stats = self.system.get_system_stats()
        
        assert "memo_count" in stats
        assert "query_count" in stats
        assert "embedding_metrics" in stats
        assert "holographic_metrics" in stats
        assert "cache_stats" in stats
        assert "system_metrics" in stats
    
    def test_cache_operations(self):
        """Test cache operations"""
        self.system.clear_all_caches()
        # Should not raise an exception
    
    @pytest.mark.asyncio
    async def test_system_cleanup(self):
        """Test system cleanup"""
        await self.system.close()
        # Should not raise an exception


class TestPerformanceAndScalability:
    """Test performance and scalability aspects"""
    
    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test processing large batches of memos"""
        config = HolographicIntegrationConfig(
            quantum_dimension=128,  # Smaller for faster testing
            holographic_depth=3,
            cognitive_layers=2,
            parallel_processing=True
        )
        system = HolographicQueryMemoSystem(config)
        
        try:
            # Create large batch of memos
            large_batch = [f"Test memo {i}: Content about topic {i % 10}" for i in range(50)]
            
            start_time = time.time()
            memo_indices = await system.add_memo_batch(large_batch)
            batch_time = time.time() - start_time
            
            assert len(memo_indices) == 50
            assert batch_time < 30.0  # Should complete within 30 seconds
            
            # Test query performance
            query_start = time.time()
            result = await system.query_similarity("Test query", top_k=10)
            query_time = time.time() - query_start
            
            assert query_time < 10.0  # Should complete within 10 seconds
            assert len(result["similar_memos"]) <= 10
            
        finally:
            await system.close()
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage with large datasets"""
        config = HolographicIntegrationConfig(
            quantum_dimension=256,
            holographic_depth=4,
            cognitive_layers=3,
            parallel_processing=False
        )
        system = HolographicQueryMemoSystem(config)
        
        try:
            # Add many memos
            for i in range(100):
                await system.add_memo(f"Memory test memo {i}")
            
            # Get system stats
            stats = system.get_system_stats()
            assert stats["memo_count"] == 100
            
            # Test that system still works
            result = await system.query_similarity("Test query")
            assert result is not None
            
        finally:
            await system.close()


def run_performance_benchmark():
    """Run performance benchmark"""
    print("🚀 Running Performance Benchmark...")
    
    async def benchmark():
        config = HolographicIntegrationConfig(
            quantum_dimension=512,
            holographic_depth=6,
            cognitive_layers=4,
            parallel_processing=True
        )
        system = HolographicQueryMemoSystem(config)
        
        try:
            # Benchmark memo addition
            print("📝 Benchmarking memo addition...")
            memo_texts = [f"Benchmark memo {i}" for i in range(100)]
            
            start_time = time.time()
            await system.add_memo_batch(memo_texts)
            addition_time = time.time() - start_time
            
            print(f"✅ Added 100 memos in {addition_time:.3f}s ({addition_time/100*1000:.1f}ms per memo)")
            
            # Benchmark queries
            print("🔍 Benchmarking queries...")
            query_texts = [
                "What is quantum mechanics?",
                "Explain fractal geometry",
                "How do neural networks work?",
                "What is information theory?",
                "Describe machine learning"
            ]
            
            total_query_time = 0
            for query in query_texts:
                start_time = time.time()
                result = await system.query_similarity(query, top_k=5)
                query_time = time.time() - start_time
                total_query_time += query_time
                
                print(f"  Query: {query[:30]}... ({query_time:.3f}s, {len(result['similar_memos'])} results)")
            
            avg_query_time = total_query_time / len(query_texts)
            print(f"✅ Average query time: {avg_query_time:.3f}s")
            
            # Print system stats
            stats = system.get_system_stats()
            print(f"\n📊 System Statistics:")
            print(f"  Memos: {stats['memo_count']}")
            print(f"  Queries: {stats['query_count']}")
            print(f"  Embedding metrics: {stats['embedding_metrics']}")
            print(f"  Holographic metrics: {stats['holographic_metrics']}")
            
        finally:
            await system.close()
    
    asyncio.run(benchmark())


def run_comprehensive_demo():
    """Run comprehensive demonstration"""
    print("🎭 Running Comprehensive Holographic Similarity Demo...")
    
    async def demo():
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
            # Add diverse memos
            print("📝 Adding diverse memos...")
            diverse_memos = [
                "Quantum mechanics: The mathematical framework describing the behavior of matter and energy at atomic and subatomic scales.",
                "Fractal geometry: Mathematical structures that exhibit self-similarity at different scales, revealing infinite complexity.",
                "Neural networks: Computational models inspired by biological neural networks, capable of learning complex patterns.",
                "Information theory: The mathematical study of information, including its quantification, storage, and communication.",
                "Machine learning: Algorithms that improve their performance on a task through experience and data.",
                "Holographic principle: The idea that information in a volume of space can be encoded on its boundary surface.",
                "Cognitive science: The interdisciplinary study of mind and intelligence, including philosophy, psychology, and AI.",
                "Complexity theory: The study of complex systems and their emergent properties and behaviors.",
                "Quantum entanglement: A quantum mechanical phenomenon where particles become interconnected and share quantum states.",
                "Deep learning: Machine learning methods based on artificial neural networks with multiple layers."
            ]
            
            memo_indices = await system.add_memo_batch(diverse_memos)
            print(f"✅ Added {len(memo_indices)} diverse memos")
            
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
            
            print("\n🔍 Testing various queries...")
            for i, query in enumerate(test_queries):
                print(f"\n--- Query {i+1}: {query} ---")
                
                result = await system.query_similarity(query, top_k=3, similarity_threshold=0.1)
                
                print(f"Found {len(result['similar_memos'])} similar memos:")
                for j, memo in enumerate(result['similar_memos']):
                    print(f"  {j+1}. Score: {memo['similarity_score']:.4f}")
                    print(f"     Text: {memo['text'][:80]}...")
                
                print(f"Overall similarity: {result['overall_similarity']}")
                print(f"Calculation time: {result['metadata']['total_time']:.3f}s")
            
            # Test edge cases
            print("\n🧪 Testing edge cases...")
            
            # Empty query
            empty_result = await system.query_similarity("", top_k=3)
            print(f"Empty query results: {len(empty_result['similar_memos'])} memos")
            
            # Very specific query
            specific_result = await system.query_similarity(
                "quantum entanglement non-local correlations particles", 
                top_k=5
            )
            print(f"Specific query results: {len(specific_result['similar_memos'])} memos")
            
            # Get final system stats
            print("\n📊 Final System Statistics:")
            stats = system.get_system_stats()
            print(f"Total memos: {stats['memo_count']}")
            print(f"Total queries: {stats['query_count']}")
            print(f"Embedding metrics: {stats['embedding_metrics']}")
            print(f"Holographic metrics: {stats['holographic_metrics']}")
            
        finally:
            await system.close()
    
    asyncio.run(demo())


if __name__ == "__main__":
    print("🧪 Holographic Similarity System Test Suite")
    print("=" * 50)
    
    # Run performance benchmark
    run_performance_benchmark()
    
    print("\n" + "=" * 50)
    
    # Run comprehensive demo
    run_comprehensive_demo()
    
    print("\n✅ All tests and demos completed!")