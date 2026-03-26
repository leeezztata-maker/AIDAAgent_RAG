from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/v1")


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "dataset_id": None,
        "dataset_filename": None,
        "dataset_context": None,
        "analysis": None,
        "charts": None,
        "question_response": None,
        "executive_summary": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _backend_url() -> str:
    return st.session_state.get("backend_url", DEFAULT_BACKEND_URL).rstrip("/")


def _api_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    try:
        response = requests.request(
            method=method,
            url=f"{_backend_url()}{path}",
            timeout=60,
            **kwargs,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach backend at {_backend_url()}.") from exc

    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text or "Unknown API error."}

    if response.ok:
        return payload

    detail = payload["detail"] if isinstance(payload, dict) and "detail" in payload else payload
    raise RuntimeError(str(detail))


def _upload_dataset(uploaded_file: Any) -> None:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
    upload_payload = _api_request("POST", "/datasets/upload", files=files)
    dataset_id = upload_payload["dataset_id"]

    st.session_state.dataset_id = dataset_id
    st.session_state.dataset_filename = upload_payload["filename"]
    st.session_state.analysis = None
    st.session_state.charts = None
    st.session_state.question_response = None
    st.session_state.executive_summary = None
    st.session_state.dataset_context = _api_request("GET", f"/datasets/{dataset_id}/context")


def _run_analysis(dataset_id: str) -> None:
    st.session_state.analysis = _api_request("GET", f"/datasets/{dataset_id}/analysis")
    st.session_state.charts = _api_request("GET", f"/datasets/{dataset_id}/charts")
    try:
        st.session_state.executive_summary = _api_request(
            "POST", f"/datasets/{dataset_id}/executive-summary"
        )
    except RuntimeError as exc:
        st.warning(f"Executive summary unavailable: {exc}")


def _ask_question(dataset_id: str, question: str) -> None:
    st.session_state.question_response = _api_request(
        "POST",
        f"/datasets/{dataset_id}/ask",
        json={"question": question},
    )


def _refresh_summary(dataset_id: str) -> None:
    st.session_state.executive_summary = _api_request(
        "POST", f"/datasets/{dataset_id}/executive-summary"
    )


def _render_context(context: dict[str, Any]) -> None:
    overview = context["overview"]
    structure = context["structure"]
    schema_df = pd.DataFrame(context["schema"])
    missing_df = pd.DataFrame(context["missing_values"])

    st.subheader("Dataset Overview")
    metrics = st.columns(5)
    metrics[0].metric("Rows", overview["row_count"])
    metrics[1].metric("Columns", overview["column_count"])
    metrics[2].metric("Numeric Columns", overview["numeric_column_count"])
    metrics[3].metric("Categorical Columns", overview["categorical_column_count"])
    metrics[4].metric("Duplicate Rows", structure["duplicate_row_count"])

    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("**Schema**")
        st.dataframe(schema_df, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Missing Values**")
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

    if structure["sample_rows"]:
        st.markdown("**Sample-Level Structural Preview**")
        st.dataframe(pd.DataFrame(structure["sample_rows"]), use_container_width=True, hide_index=True)


def _render_analysis(analysis: dict[str, Any]) -> None:
    st.subheader("Auto EDA")

    numeric_df = pd.DataFrame(analysis["numeric_summary"])
    categorical_rows: list[dict[str, Any]] = []
    for item in analysis["categorical_summary"]:
        top_values = ", ".join(
            f"{entry['value']} ({entry['count']})" for entry in item.get("top_values", [])
        )
        categorical_rows.append(
            {
                "column": item["column"],
                "distinct_count": item["distinct_count"],
                "top_values": top_values,
            }
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**Numeric Summaries**")
        if numeric_df.empty:
            st.info("No numeric columns found.")
        else:
            st.dataframe(numeric_df, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Categorical Summaries**")
        categorical_df = pd.DataFrame(categorical_rows)
        if categorical_df.empty:
            st.info("No categorical columns found.")
        else:
            st.dataframe(categorical_df, use_container_width=True, hide_index=True)


def _render_charts(charts: dict[str, Any] | None) -> None:
    st.subheader("Charts")
    if not charts or not charts.get("charts"):
        st.info("Run analysis to generate chart metadata and visualizations.")
        return

    for chart in charts["charts"]:
        st.markdown(f"**{chart['title']}**")
        figure = go.Figure(chart["payload"])
        st.plotly_chart(figure, use_container_width=True)


def _render_question_response(payload: dict[str, Any]) -> None:
    st.markdown("**Grounded Answer**")
    st.write(payload["answer"])

    if payload.get("supporting_points"):
        st.markdown("**Supporting Points**")
        for point in payload["supporting_points"]:
            st.write(f"- {point}")

    if payload.get("caveats"):
        st.markdown("**Caveats**")
        for item in payload["caveats"]:
            st.write(f"- {item}")

    if payload.get("grounded_in"):
        st.caption("Grounded in: " + ", ".join(payload["grounded_in"]))


def _render_summary(summary: dict[str, Any] | None) -> None:
    st.subheader("Executive Summary")
    if not summary:
        st.info("Generate the executive summary after running analysis.")
        return

    st.write(summary["summary"])
    if summary.get("highlights"):
        st.markdown("**Highlights**")
        for item in summary["highlights"]:
            st.write(f"- {item}")
    if summary.get("caveats"):
        st.markdown("**Caveats**")
        for item in summary["caveats"]:
            st.write(f"- {item}")
    if summary.get("grounded_in"):
        st.caption("Grounded in: " + ", ".join(summary["grounded_in"]))

st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")
_init_state()

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.3rem 1.5rem;
        border: 1px solid #d9e2ec;
        border-radius: 18px;
        background: linear-gradient(135deg, #f6fbff 0%, #eef6ef 100%);
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0;">AI Data Analysis Agent</h1>
      <p style="margin:0.5rem 0 0 0;">
        A grounded analytics assistant for CSV datasets. Upload a file, inspect schema and data quality,
        generate automated EDA, review charts, and ask questions that are constrained to computed dataset context.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuration")
    st.text_input("Backend API URL", key="backend_url", value=DEFAULT_BACKEND_URL)
    st.caption("Expected FastAPI base URL, including `/api/v1`.")

st.subheader("Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
upload_col, info_col = st.columns([0.7, 1.3])

with upload_col:
    if st.button("Upload CSV", type="primary", disabled=uploaded_file is None):
        try:
            _upload_dataset(uploaded_file)
            st.success(f"Uploaded `{st.session_state.dataset_filename}`.")
        except RuntimeError as exc:
            st.error(f"Upload failed: {exc}")

with info_col:
    if st.session_state.dataset_id:
        st.caption(f"Dataset ID: `{st.session_state.dataset_id}`")
        st.caption(f"Active file: `{st.session_state.dataset_filename}`")

if st.session_state.dataset_context:
    _render_context(st.session_state.dataset_context)

    st.subheader("Run Analysis")
    run_col, refresh_col = st.columns([0.7, 1.3])
    with run_col:
        if st.button("Run Full Analysis", disabled=st.session_state.dataset_id is None):
            try:
                _run_analysis(st.session_state.dataset_id)
                st.success("Analysis complete.")
            except RuntimeError as exc:
                st.error(f"Analysis failed: {exc}")
    with refresh_col:
        st.caption(
            "This runs deterministic profiling, retrieves chart metadata, and attempts a grounded executive summary."
        )

if st.session_state.analysis:
    _render_analysis(st.session_state.analysis)

_render_charts(st.session_state.charts)

st.subheader("Ask the Agent")
question = st.text_input(
    "Ask a question about the dataset",
    placeholder="Example: Which columns have the most missing values, and what numeric fields look most variable?",
)
if st.button("Get Grounded Answer", disabled=not st.session_state.dataset_id or not question.strip()):
    try:
        _ask_question(st.session_state.dataset_id, question.strip())
    except RuntimeError as exc:
        st.error(f"Question failed: {exc}")

if st.session_state.question_response:
    _render_question_response(st.session_state.question_response)

summary_action_col, summary_text_col = st.columns([0.7, 1.3])
with summary_action_col:
    if st.button("Refresh Executive Summary", disabled=st.session_state.dataset_id is None):
        try:
            _refresh_summary(st.session_state.dataset_id)
        except RuntimeError as exc:
            st.error(f"Summary generation failed: {exc}")
with summary_text_col:
    st.caption("The summary is constrained to computed dataset profile outputs and chart metadata.")

_render_summary(st.session_state.executive_summary)
