from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.models.schemas import (
    CategoryValueCount,
    CategoricalSummary,
    ColumnSchema,
    DatasetOverview,
    DatasetProfileResponse,
    DatasetStructureSummary,
    MissingValueSummary,
    NumericSummary,
)
from app.services.charts import chart_service


def _clean_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    return value


class ProfileService:
    def build_profile(
        self,
        dataframe: pd.DataFrame,
        dataset_id: str,
        filename: str = "uploaded.csv",
    ) -> DatasetProfileResponse:
        normalized = dataframe.copy()
        normalized.columns = [str(column).strip() for column in normalized.columns]

        schema = self._build_schema(normalized)
        missing_values = self._build_missing_values(normalized)
        numeric_summary = self._build_numeric_summary(normalized)
        categorical_summary = self._build_categorical_summary(normalized)
        charts = chart_service.build_chart_specs(normalized)
        structure = self._build_structure_summary(normalized)

        overview = DatasetOverview(
            dataset_id=dataset_id,
            row_count=int(len(normalized)),
            column_count=int(len(normalized.columns)),
            numeric_column_count=int(len(normalized.select_dtypes(include="number").columns)),
            categorical_column_count=int(len(normalized.select_dtypes(exclude="number").columns)),
        )

        return DatasetProfileResponse(
            dataset_id=dataset_id,
            filename=filename,
            overview=overview,
            structure=structure,
            schema=schema,
            missing_values=missing_values,
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            charts=charts,
        )

    def _build_structure_summary(self, dataframe: pd.DataFrame) -> DatasetStructureSummary:
        preview = dataframe.head(5).replace({np.nan: None})
        sample_rows = [
            {str(column): _clean_scalar(value) for column, value in row.items()}
            for row in preview.to_dict(orient="records")
        ]
        return DatasetStructureSummary(
            duplicate_row_count=int(dataframe.duplicated().sum()),
            total_missing_cells=int(dataframe.isna().sum().sum()),
            sample_rows=sample_rows,
        )

    def _build_schema(self, dataframe: pd.DataFrame) -> list[ColumnSchema]:
        schema: list[ColumnSchema] = []
        for column in dataframe.columns:
            series = dataframe[column]
            sample_values = [str(value) for value in series.dropna().astype(str).head(3).tolist()]
            schema.append(
                ColumnSchema(
                    name=column,
                    dtype=str(series.dtype),
                    non_null_count=int(series.notna().sum()),
                    null_count=int(series.isna().sum()),
                    unique_count=int(series.nunique(dropna=True)),
                    sample_values=sample_values,
                )
            )
        return schema

    def _build_missing_values(self, dataframe: pd.DataFrame) -> list[MissingValueSummary]:
        row_count = max(len(dataframe), 1)
        summaries: list[MissingValueSummary] = []
        for column in dataframe.columns:
            missing_count = int(dataframe[column].isna().sum())
            summaries.append(
                MissingValueSummary(
                    column=column,
                    missing_count=missing_count,
                    missing_percentage=round((missing_count / row_count) * 100, 2),
                )
            )
        return sorted(summaries, key=lambda item: item.missing_count, reverse=True)

    def _build_numeric_summary(self, dataframe: pd.DataFrame) -> list[NumericSummary]:
        summaries: list[NumericSummary] = []
        numeric_frame = dataframe.select_dtypes(include="number")

        for column in numeric_frame.columns:
            series = numeric_frame[column].dropna()
            summaries.append(
                NumericSummary(
                    column=column,
                    count=int(series.count()),
                    mean=_clean_scalar(series.mean()),
                    median=_clean_scalar(series.median()),
                    std=_clean_scalar(series.std()),
                    minimum=_clean_scalar(series.min()),
                    maximum=_clean_scalar(series.max()),
                    q1=_clean_scalar(series.quantile(0.25)),
                    q3=_clean_scalar(series.quantile(0.75)),
                )
            )

        return summaries

    def _build_categorical_summary(self, dataframe: pd.DataFrame) -> list[CategoricalSummary]:
        summaries: list[CategoricalSummary] = []
        categorical_frame = dataframe.select_dtypes(exclude="number")

        for column in categorical_frame.columns:
            series = categorical_frame[column].fillna("Missing").astype(str)
            value_counts = series.value_counts().head(5)
            summaries.append(
                CategoricalSummary(
                    column=column,
                    distinct_count=int(series.nunique(dropna=False)),
                    top_values=[
                        CategoryValueCount(value=str(index), count=int(count))
                        for index, count in value_counts.items()
                    ],
                )
            )

        return summaries


profile_service = ProfileService()
