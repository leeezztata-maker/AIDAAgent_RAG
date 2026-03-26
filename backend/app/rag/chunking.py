from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import DatasetProfileResponse


@dataclass(slots=True)
class KnowledgeChunk:
    chunk_id: str
    dataset_id: str
    source_type: str
    column_name: str | None
    importance_level: str
    text: str


class DatasetChunker:
    """Transform deterministic profile outputs into semantic units for retrieval.

    Why this design?
    - The source profile is already structured and reliable.
    - Converting each analytical fact into a short natural-language chunk makes retrieval
      explainable in interviews: every embedded item corresponds to a concrete statistic.
    - We avoid chunking raw CSV rows because the product's current trust model is based on
      computed summaries, not broad row-level search.
    """

    def build_chunks(self, profile: DatasetProfileResponse) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        chunks.extend(self._schema_chunks(profile))
        chunks.extend(self._missing_value_chunks(profile))
        chunks.extend(self._numeric_chunks(profile))
        chunks.extend(self._categorical_chunks(profile))
        return chunks

    def _schema_chunks(self, profile: DatasetProfileResponse) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for item in profile.schema:
            sample_values = ", ".join(item.sample_values) if item.sample_values else "no sample values"
            text = (
                f"Schema for column '{item.name}': dtype is {item.dtype}, non-null count is "
                f"{item.non_null_count}, null count is {item.null_count}, unique count is "
                f"{item.unique_count}, and sample values are {sample_values}."
            )
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"schema::{item.name}",
                    dataset_id=profile.dataset_id,
                    source_type="schema",
                    column_name=item.name,
                    importance_level="medium",
                    text=text,
                )
            )
        return chunks

    def _missing_value_chunks(self, profile: DatasetProfileResponse) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for item in profile.missing_values:
            importance = "high" if item.missing_percentage >= 20 else "medium" if item.missing_count else "low"
            text = (
                f"Data quality for column '{item.column}': missing count is {item.missing_count}, "
                f"which is {item.missing_percentage}% of rows."
            )
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"missing::{item.column}",
                    dataset_id=profile.dataset_id,
                    source_type="missing",
                    column_name=item.column,
                    importance_level=importance,
                    text=text,
                )
            )
        return chunks

    def _numeric_chunks(self, profile: DatasetProfileResponse) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for item in profile.numeric_summary:
            variability = "high" if (item.std or 0) > abs(item.mean or 0) else "moderate"
            text = (
                f"Numeric summary for column '{item.column}': count={item.count}, mean={item.mean}, "
                f"median={item.median}, standard deviation={item.std}, min={item.minimum}, "
                f"max={item.maximum}, first quartile={item.q1}, third quartile={item.q3}, "
                f"with {variability} relative variability."
            )
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"numeric::{item.column}",
                    dataset_id=profile.dataset_id,
                    source_type="numeric",
                    column_name=item.column,
                    importance_level="high",
                    text=text,
                )
            )
        return chunks

    def _categorical_chunks(self, profile: DatasetProfileResponse) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for item in profile.categorical_summary:
            top_values = ", ".join(
                f"{entry.value} ({entry.count})" for entry in item.top_values
            ) or "no frequent values available"
            text = (
                f"Categorical distribution for column '{item.column}': distinct count is "
                f"{item.distinct_count}. Most common values are {top_values}."
            )
            importance = "high" if item.distinct_count <= 20 else "medium"
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"categorical::{item.column}",
                    dataset_id=profile.dataset_id,
                    source_type="categorical",
                    column_name=item.column,
                    importance_level=importance,
                    text=text,
                )
            )
        return chunks


dataset_chunker = DatasetChunker()
