from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.models.schemas import DatasetProfileResponse
from app.rag.chunking import dataset_chunker
from app.rag.embedding import EmbeddingProvider, get_embedding_provider
from app.rag.vector_store import VectorSearchResult, vector_store


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    source_type: str
    column_name: str | None
    importance_level: str
    text: str
    score: float


class DatasetRetriever:
    def __init__(self, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.settings = get_settings()
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self._indexed_profiles: set[str] = set()

    def index_profile(self, profile: DatasetProfileResponse) -> None:
        if profile.dataset_id in self._indexed_profiles:
            return

        chunks = dataset_chunker.build_chunks(profile)
        embeddings = self.embedding_provider.embed_documents([chunk.text for chunk in chunks])
        vector_store.add_documents(profile.dataset_id, chunks, embeddings)
        self._indexed_profiles.add(profile.dataset_id)

    def retrieve(
        self,
        profile: DatasetProfileResponse,
        query: str,
        top_k: int | None = None,
        source_types: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.index_profile(profile)
        effective_top_k = top_k or self.settings.retrieval_top_k
        inferred_filters = self._infer_source_filters(query)
        active_filters = source_types or inferred_filters

        query_embedding = self.embedding_provider.embed_query(query)
        results = vector_store.similarity_search(
            dataset_id=profile.dataset_id,
            query_embedding=query_embedding,
            top_k=effective_top_k,
            source_types=active_filters,
            minimum_score=self.settings.retrieval_score_threshold,
        )
        return [self._to_retrieved_chunk(item) for item in results]

    def build_context(self, retrieved_chunks: list[RetrievedChunk]) -> str:
        """Limit context size so retrieval stays relevant and prompt cost stays predictable.

        Why this design?
        - A bounded context window is easier to explain than "we pass whatever comes back."
        - It prevents a common RAG failure mode where good retrieval is diluted by too much context.
        """

        selected_lines: list[str] = []
        current_size = 0
        for item in retrieved_chunks:
            line = (
                f"[{item.chunk_id}] score={item.score:.3f} source={item.source_type} "
                f"column={item.column_name or 'dataset'} importance={item.importance_level}: {item.text}"
            )
            projected_size = current_size + len(line)
            if projected_size > self.settings.rag_context_char_limit:
                break
            selected_lines.append(line)
            current_size = projected_size
        return "\n".join(selected_lines)

    def _infer_source_filters(self, query: str) -> set[str] | None:
        lowered = query.lower()
        if any(token in lowered for token in ["mean", "median", "average", "std", "variance", "numeric"]):
            return {"numeric", "missing", "schema"}
        if any(token in lowered for token in ["category", "categorical", "most common", "distribution"]):
            return {"categorical", "missing", "schema"}
        if any(token in lowered for token in ["missing", "null", "empty", "quality"]):
            return {"missing", "schema"}
        return None

    @staticmethod
    def _to_retrieved_chunk(item: VectorSearchResult) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=item.chunk.chunk_id,
            source_type=item.chunk.source_type,
            column_name=item.chunk.column_name,
            importance_level=item.chunk.importance_level,
            text=item.chunk.text,
            score=round(item.score, 4),
        )


dataset_retriever = DatasetRetriever()
