from typing import Any

from pydantic import BaseModel, Field


class ColumnSchema(BaseModel):
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    unique_count: int
    sample_values: list[str]


class MissingValueSummary(BaseModel):
    column: str
    missing_count: int
    missing_percentage: float


class NumericSummary(BaseModel):
    column: str
    count: int
    mean: float | None
    median: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    q1: float | None
    q3: float | None


class CategoryValueCount(BaseModel):
    value: str
    count: int


class CategoricalSummary(BaseModel):
    column: str
    distinct_count: int
    top_values: list[CategoryValueCount]


class ChartSpec(BaseModel):
    title: str
    chart_type: str
    x_field: str | None = None
    y_field: str | None = None
    payload: dict[str, Any]


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    source_type: str
    column_name: str | None
    importance_level: str
    text: str
    score: float


class DatasetStructureSummary(BaseModel):
    duplicate_row_count: int
    total_missing_cells: int
    sample_rows: list[dict[str, Any]]


class DatasetOverview(BaseModel):
    dataset_id: str
    row_count: int
    column_count: int
    numeric_column_count: int
    categorical_column_count: int


class DatasetProfileResponse(BaseModel):
    dataset_id: str
    filename: str
    overview: DatasetOverview
    structure: DatasetStructureSummary
    schema: list[ColumnSchema]
    missing_values: list[MissingValueSummary]
    numeric_summary: list[NumericSummary]
    categorical_summary: list[CategoricalSummary]
    charts: list[ChartSpec]


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    columns: int


class DatasetContextResponse(BaseModel):
    dataset_id: str
    filename: str
    overview: DatasetOverview
    structure: DatasetStructureSummary
    schema: list[ColumnSchema]
    missing_values: list[MissingValueSummary]


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class QuestionResponse(BaseModel):
    dataset_id: str
    question: str
    answer: str
    supporting_points: list[str]
    data_supported_facts: list[str]
    inference: list[str]
    evidence: list[str]
    grounded_in: list[str]
    caveats: list[str]
    retrieval_top_k: int
    retrieved_chunks: list[RetrievedChunkResponse]


class ExecutiveSummaryResponse(BaseModel):
    dataset_id: str
    summary: str
    highlights: list[str]
    caveats: list[str]
    grounded_in: list[str]


class ChartListResponse(BaseModel):
    dataset_id: str
    charts: list[ChartSpec]


class EvaluationRequest(BaseModel):
    questions: list[str] = Field(min_length=1)


class EvaluationResponse(BaseModel):
    dataset_id: str
    questions_evaluated: int
    records: list[dict[str, Any]]
