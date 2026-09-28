"""Holographic memory: HRR algebra, embedders and a persistent RAG vector store."""

from .embed import HashingEmbedder, SentenceTransformerEmbedder, embed_stream, get_embedder
from .hrr import AssociativeTrace, Cleanup, bind, bundle, capacity, normalize, symbol, unbind, unitary
from .store import Hit, HolographicStore

__all__ = [
    "HolographicStore", "Hit", "HashingEmbedder", "SentenceTransformerEmbedder", "embed_stream", "get_embedder",
    "AssociativeTrace", "Cleanup", "bind", "unbind", "bundle", "normalize", "symbol", "unitary", "capacity",
]
