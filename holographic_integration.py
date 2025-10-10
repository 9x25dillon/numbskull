#!/usr/bin/env python3
"""
Holographic Integration Layer - Integration with Advanced Embedding Pipeline
Connects the holographic similarity engine with the existing embedding pipeline
for seamless query-memo similarity calculations.
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import json
import time
from pathlib import Path

# Import existing pipeline components
from advanced_embedding_pipeline.hybrid_pipeline import HybridEmbeddingPipeline, HybridConfig
from advanced_embedding_pipeline.semantic_embedder import SemanticConfig
from advanced_embedding_pipeline.mathematical_embedder import MathematicalConfig
from advanced_embedding_pipeline.fractal_cascade_embedder import FractalConfig

# Import holographic similarity engine
from holographic_similarity_engine import HolographicSimilarityEngine, HolographicConfig

logger = logging.getLogger(__name__)


@dataclass
class HolographicIntegrationConfig:
    """Configuration for holographic integration"""
    # Embedding pipeline configuration
    use_semantic: bool = True
    use_mathematical: bool = True
    use_fractal: bool = True
    fusion_method: str = "weighted_average"
    
    # Holographic similarity configuration
    quantum_dimension: int = 1024
    holographic_depth: int = 8
    cognitive_layers: int = 6
    
    # Integration settings
    enable_caching: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Service URLs
    eopiez_url: str = "http://localhost:8001"
    limps_url: str = "http://localhost:8000"


class HolographicQueryMemoSystem:
    """
    Integrated system for holographic query-memo similarity calculations.
    Combines the advanced embedding pipeline with holographic similarity engine.
    """
    
    def __init__(self, config: Optional[HolographicIntegrationConfig] = None):
        self.config = config or HolographicIntegrationConfig()
        
        # Initialize embedding pipeline
        self.embedding_pipeline = self._create_embedding_pipeline()
        
        # Initialize holographic similarity engine
        self.holographic_engine = self._create_holographic_engine()
        
        # Memo storage
        self.memo_embeddings = []
        self.memo_texts = []
        self.memo_metadata = []
        
        # Performance metrics
        self.metrics = {
            "total_queries": 0,
            "total_memos": 0,
            "embedding_generation_time": 0.0,
            "similarity_calculation_time": 0.0,
            "cache_hits": 0
        }
        
        logger.info("✅ Holographic Query-Memo System initialized")
    
    def _create_embedding_pipeline(self) -> HybridEmbeddingPipeline:
        """Create and configure the embedding pipeline"""
        try:
            # Create component configurations
            semantic_config = SemanticConfig(
                eopiez_url=self.config.eopiez_url,
                embedding_dim=768,
                batch_size=32,
                use_cache=self.config.enable_caching
            )
            
            mathematical_config = MathematicalConfig(
                limps_url=self.config.limps_url,
                max_dimension=1024,
                polynomial_degree=3,
                use_matrix_optimization=True
            )
            
            fractal_config = FractalConfig(
                max_depth=6,
                branching_factor=3,
                embedding_dim=1024,
                fractal_type="mandelbrot",
                use_entropy=True
            )
            
            # Create hybrid configuration
            hybrid_config = HybridConfig(
                semantic_config=semantic_config,
                mathematical_config=mathematical_config,
                fractal_config=fractal_config,
                use_semantic=self.config.use_semantic,
                use_mathematical=self.config.use_mathematical,
                use_fractal=self.config.use_fractal,
                fusion_method=self.config.fusion_method,
                parallel_processing=self.config.parallel_processing,
                max_workers=self.config.max_workers
            )
            
            return HybridEmbeddingPipeline(hybrid_config)
            
        except Exception as e:
            logger.error(f"Failed to create embedding pipeline: {e}")
            raise
    
    def _create_holographic_engine(self) -> HolographicSimilarityEngine:
        """Create and configure the holographic similarity engine"""
        try:
            holographic_config = HolographicConfig(
                quantum_dimension=self.config.quantum_dimension,
                holographic_depth=self.config.holographic_depth,
                cognitive_layers=self.config.cognitive_layers,
                parallel_processing=self.config.parallel_processing,
                max_workers=self.config.max_workers,
                cache_calculations=self.config.enable_caching
            )
            
            return HolographicSimilarityEngine(holographic_config)
            
        except Exception as e:
            logger.error(f"Failed to create holographic engine: {e}")
            raise
    
    async def add_memo(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a memo to the system
        
        Args:
            text: Memo text content
            metadata: Optional metadata for the memo
            
        Returns:
            Memo index
        """
        try:
            # Generate embedding for memo
            start_time = time.time()
            embedding_result = await self.embedding_pipeline.embed(text)
            embedding_time = time.time() - start_time
            
            # Store memo
            memo_index = len(self.memo_embeddings)
            self.memo_embeddings.append(embedding_result['fused_embedding'])
            self.memo_texts.append(text)
            self.memo_metadata.append(metadata or {})
            
            # Update metrics
            self.metrics["total_memos"] += 1
            self.metrics["embedding_generation_time"] += embedding_time
            
            logger.info(f"✅ Memo {memo_index} added: {text[:50]}...")
            return memo_index
            
        except Exception as e:
            logger.error(f"Failed to add memo: {e}")
            raise
    
    async def add_memo_batch(self, texts: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        Add multiple memos to the system
        
        Args:
            texts: List of memo texts
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            List of memo indices
        """
        try:
            # Generate embeddings for all memos
            start_time = time.time()
            embedding_results = await self.embedding_pipeline.embed_batch(texts)
            embedding_time = time.time() - start_time
            
            # Store memos
            memo_indices = []
            for i, result in enumerate(embedding_results):
                memo_index = len(self.memo_embeddings)
                self.memo_embeddings.append(result['fused_embedding'])
                self.memo_texts.append(texts[i])
                self.memo_metadata.append(metadata_list[i] if metadata_list and i < len(metadata_list) else {})
                memo_indices.append(memo_index)
            
            # Update metrics
            self.metrics["total_memos"] += len(texts)
            self.metrics["embedding_generation_time"] += embedding_time
            
            logger.info(f"✅ Added {len(texts)} memos to system")
            return memo_indices
            
        except Exception as e:
            logger.error(f"Failed to add memo batch: {e}")
            raise
    
    async def query_similarity(
        self, 
        query_text: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Query the system for similar memos
        
        Args:
            query_text: Query text
            top_k: Number of top similar memos to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Query results with similar memos
        """
        try:
            if not self.memo_embeddings:
                return {
                    "query_text": query_text,
                    "similar_memos": [],
                    "message": "No memos in system"
                }
            
            # Generate query embedding
            start_time = time.time()
            query_result = await self.embedding_pipeline.embed(query_text)
            embedding_time = time.time() - start_time
            
            # Calculate holographic similarity
            similarity_start = time.time()
            similarity_result = await self.holographic_engine.calculate_holographic_similarity(
                query_result['fused_embedding'],
                self.memo_embeddings,
                self.memo_texts
            )
            similarity_time = time.time() - similarity_start
            
            # Process results
            similarities = similarity_result['similarities']
            
            # Filter by threshold and sort by similarity
            filtered_similarities = [
                sim for sim in similarities
                if sim['similarity_score'] >= similarity_threshold
            ]
            
            # Sort by similarity score (descending)
            sorted_similarities = sorted(
                filtered_similarities,
                key=lambda x: x['similarity_score'],
                reverse=True
            )
            
            # Take top_k results
            top_similarities = sorted_similarities[:top_k]
            
            # Prepare similar memos
            similar_memos = []
            for sim in top_similarities:
                memo_index = sim['memo_index']
                similar_memos.append({
                    "memo_index": memo_index,
                    "text": self.memo_texts[memo_index],
                    "metadata": self.memo_metadata[memo_index],
                    "similarity_score": sim['similarity_score'],
                    "method_scores": sim.get('method_scores', {}),
                    "transcribed_memo": sim.get('transcribed_memo', None)
                })
            
            # Update metrics
            self.metrics["total_queries"] += 1
            self.metrics["embedding_generation_time"] += embedding_time
            self.metrics["similarity_calculation_time"] += similarity_time
            
            # Prepare result
            result = {
                "query_text": query_text,
                "query_embedding": query_result['fused_embedding'],
                "transcribed_query": similarity_result['transcribed_query'],
                "similar_memos": similar_memos,
                "total_memos_searched": len(self.memo_embeddings),
                "similarities_found": len(filtered_similarities),
                "overall_similarity": similarity_result['overall_similarity'],
                "metadata": {
                    "embedding_time": embedding_time,
                    "similarity_time": similarity_time,
                    "total_time": embedding_time + similarity_time,
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold,
                    "methods_used": similarity_result['metadata']['methods_used']
                }
            }
            
            logger.info(f"✅ Query processed: {len(similar_memos)} similar memos found")
            return result
            
        except Exception as e:
            logger.error(f"Query similarity failed: {e}")
            return {
                "query_text": query_text,
                "error": str(e),
                "similar_memos": [],
                "metadata": {"total_time": 0.0}
            }
    
    async def get_memo(self, memo_index: int) -> Optional[Dict[str, Any]]:
        """Get a specific memo by index"""
        try:
            if 0 <= memo_index < len(self.memo_embeddings):
                return {
                    "memo_index": memo_index,
                    "text": self.memo_texts[memo_index],
                    "metadata": self.memo_metadata[memo_index],
                    "embedding": self.memo_embeddings[memo_index]
                }
            else:
                return None
        except Exception as e:
            logger.warning(f"Failed to get memo {memo_index}: {e}")
            return None
    
    async def update_memo(self, memo_index: int, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing memo"""
        try:
            if 0 <= memo_index < len(self.memo_embeddings):
                # Generate new embedding
                embedding_result = await self.embedding_pipeline.embed(text)
                
                # Update memo
                self.memo_texts[memo_index] = text
                self.memo_embeddings[memo_index] = embedding_result['fused_embedding']
                if metadata is not None:
                    self.memo_metadata[memo_index] = metadata
                
                logger.info(f"✅ Memo {memo_index} updated")
                return True
            else:
                logger.warning(f"Memo index {memo_index} out of range")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update memo {memo_index}: {e}")
            return False
    
    async def delete_memo(self, memo_index: int) -> bool:
        """Delete a memo by index"""
        try:
            if 0 <= memo_index < len(self.memo_embeddings):
                # Remove memo
                del self.memo_texts[memo_index]
                del self.memo_embeddings[memo_index]
                del self.memo_metadata[memo_index]
                
                # Update metrics
                self.metrics["total_memos"] -= 1
                
                logger.info(f"✅ Memo {memo_index} deleted")
                return True
            else:
                logger.warning(f"Memo index {memo_index} out of range")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete memo {memo_index}: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            # Get embedding pipeline metrics
            embedding_metrics = self.embedding_pipeline.get_metrics()
            
            # Get holographic engine metrics
            holographic_metrics = self.holographic_engine.get_metrics()
            
            # Get cache stats
            cache_stats = self.holographic_engine.get_cache_stats()
            
            return {
                "memo_count": len(self.memo_embeddings),
                "query_count": self.metrics["total_queries"],
                "embedding_metrics": embedding_metrics,
                "holographic_metrics": holographic_metrics,
                "cache_stats": cache_stats,
                "system_metrics": self.metrics
            }
            
        except Exception as e:
            logger.warning(f"Failed to get system stats: {e}")
            return {"error": str(e)}
    
    def clear_all_caches(self):
        """Clear all caches"""
        try:
            self.embedding_pipeline.clear_cache()
            self.holographic_engine.clear_cache()
            logger.info("✅ All caches cleared")
        except Exception as e:
            logger.warning(f"Failed to clear caches: {e}")
    
    async def close(self):
        """Close the system and cleanup resources"""
        try:
            await self.embedding_pipeline.close()
            logger.info("✅ Holographic Query-Memo System closed")
        except Exception as e:
            logger.warning(f"Error closing system: {e}")


# Example usage and testing
async def demo_holographic_integration():
    """Demonstrate the integrated holographic query-memo system"""
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
        # Add sample memos
        sample_memos = [
            "The mathematical foundation of quantum mechanics involves complex Hilbert spaces and linear operators.",
            "Fractal geometry reveals infinite complexity in finite spaces through recursive self-similarity.",
            "Neural networks learn hierarchical representations through backpropagation and gradient descent.",
            "The holographic principle suggests that information in a volume can be encoded on its boundary.",
            "Cognitive architectures model human-like reasoning through symbolic and connectionist approaches.",
            "Machine learning algorithms optimize objective functions to minimize prediction errors.",
            "Quantum entanglement enables non-local correlations between distant particles.",
            "Deep learning models use multiple layers to learn increasingly abstract features.",
            "Information theory quantifies the amount of information in signals and communication systems.",
            "The theory of relativity describes the relationship between space, time, and gravity."
        ]
        
        print("📝 Adding memos to system...")
        memo_indices = await system.add_memo_batch(sample_memos)
        print(f"✅ Added {len(memo_indices)} memos")
        
        # Test queries
        test_queries = [
            "What is quantum mechanics?",
            "How do fractals work?",
            "Explain neural networks",
            "What is holographic principle?",
            "How does machine learning work?"
        ]
        
        print("\n🔍 Testing queries...")
        for query in test_queries:
            print(f"\nQuery: {query}")
            result = await system.query_similarity(query, top_k=3, similarity_threshold=0.1)
            
            print(f"Found {len(result['similar_memos'])} similar memos:")
            for i, memo in enumerate(result['similar_memos']):
                print(f"  {i+1}. Score: {memo['similarity_score']:.4f}")
                print(f"     Text: {memo['text'][:80]}...")
        
        # Get system stats
        print("\n📊 System Statistics:")
        stats = system.get_system_stats()
        print(f"Total memos: {stats['memo_count']}")
        print(f"Total queries: {stats['query_count']}")
        print(f"Embedding metrics: {stats['embedding_metrics']}")
        print(f"Holographic metrics: {stats['holographic_metrics']}")
        
    finally:
        # Cleanup
        await system.close()


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_holographic_integration())