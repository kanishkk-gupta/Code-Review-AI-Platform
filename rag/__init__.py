"""rag package — Vector store and retrieval components."""
from rag.embedder import embed_text, embed_chunks, is_embedder_ready
from rag.vector_store import FAISSIndex, build_index, similarity_search
from rag.retriever import retrieve_chunks, build_chunk_map, ChunkRetriever

__all__ = [
    "embed_text",
    "embed_chunks",
    "is_embedder_ready",
    "FAISSIndex",
    "build_index",
    "similarity_search",
    "retrieve_chunks",
    "build_chunk_map",
    "ChunkRetriever",
]
