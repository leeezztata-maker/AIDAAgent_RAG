from __future__ import annotations

from dataclasses import dataclass
import math

from app.rag.chunking import KnowledgeChunk


@dataclass(slots=True)
class StoredVectorDocument:
    chunk: KnowledgeChunk
    embedding: list[float]


@dataclass(slots=True)
class VectorSearchResult:
    chunk: KnowledgeChunk
    score: float


class InMemoryVectorStore:
    """Simple in-memory store tuned for explainability over infrastructure complexity.

    Why this design?
    - For an interview MVP, we want retrieval behavior to be visible and debuggable.
    - Keeping the store in-process makes it easy to explain scoring, metadata filtering,
      and indexing mechanics without adding external services too early.
    - The obvious next step would be FAISS, pgvector, or a managed vector DB.
    """

    def __init__(self) -> None:
        self._documents: dict[str, list[StoredVectorDocument]] = {}

    def add_documents(
        self,
        dataset_id: str,
        chunks: list[KnowledgeChunk],
        embeddings: list[list[float]],
    ) -> None:
        self._documents[dataset_id] = [
            StoredVectorDocument(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

    def similarity_search(
        self,
        dataset_id: str,
        query_embedding: list[float],
        top_k: int,
        source_types: set[str] | None = None,
        column_names: set[str] | None = None,
        minimum_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        matches: list[VectorSearchResult] = []
        for item in self._documents.get(dataset_id, []):
            if source_types and item.chunk.source_type not in source_types:
                continue
            if column_names and item.chunk.column_name not in column_names:
                continue
            score = self._cosine_similarity(query_embedding, item.embedding)
            if score < minimum_score:
                continue
            matches.append(VectorSearchResult(chunk=item.chunk, score=score))

        matches.sort(key=lambda result: result.score, reverse=True)
        return matches[:top_k]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(l * r for l, r in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
        right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
        return numerator / (left_norm * right_norm)


vector_store = InMemoryVectorStore()
