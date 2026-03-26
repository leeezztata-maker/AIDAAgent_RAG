from __future__ import annotations

import pandas as pd
import plotly.express as px

from app.models.schemas import ChartSpec


class ChartService:
    def build_chart_specs(self, dataframe: pd.DataFrame) -> list[ChartSpec]:
        charts: list[ChartSpec] = []
        numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
        categorical_columns = dataframe.select_dtypes(exclude="number").columns.tolist()

        for column in numeric_columns[:2]:
            fig = px.histogram(dataframe, x=column, nbins=20, title=f"Distribution of {column}")
            charts.append(
                ChartSpec(
                    title=f"Distribution of {column}",
                    chart_type="histogram",
                    x_field=column,
                    payload=fig.to_plotly_json(),
                )
            )

        for column in categorical_columns[:2]:
            top_counts = dataframe[column].fillna("Missing").astype(str).value_counts().head(10)
            fig = px.bar(
                x=top_counts.index.tolist(),
                y=top_counts.values.tolist(),
                title=f"Top categories for {column}",
                labels={"x": column, "y": "Count"},
            )
            charts.append(
                ChartSpec(
                    title=f"Top categories for {column}",
                    chart_type="bar",
                    x_field=column,
                    y_field="count",
                    payload=fig.to_plotly_json(),
                )
            )

        if len(numeric_columns) >= 2:
            x_field, y_field = numeric_columns[:2]
            sample = dataframe[[x_field, y_field]].dropna().head(500)
            if not sample.empty:
                fig = px.scatter(sample, x=x_field, y=y_field, title=f"{x_field} vs {y_field}")
                charts.append(
                    ChartSpec(
                        title=f"{x_field} vs {y_field}",
                        chart_type="scatter",
                        x_field=x_field,
                        y_field=y_field,
                        payload=fig.to_plotly_json(),
                    )
                )

        return charts


chart_service = ChartService()
