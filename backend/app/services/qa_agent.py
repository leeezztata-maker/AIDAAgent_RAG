from __future__ import annotations

import json
from typing import Literal
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.schemas import (
    DatasetProfileResponse,
    ExecutiveSummaryResponse,
    QuestionResponse,
    RetrievedChunkResponse,
)
from app.rag.retriever import RetrievedChunk, dataset_retriever
from app.services.dataset_store import DatasetRecord, dataset_store
from app.services.llm_service import llm_service
from app.services.profiling import profile_service


class QAResultPayload(BaseModel):
    answer: str = Field(min_length=1)
    supporting_points: list[str] = Field(default_factory=list)
    data_supported_facts: list[str] = Field(default_factory=list)
    inference: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class SummaryResultPayload(BaseModel):
    summary: str = Field(min_length=1)
    highlights: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class AnalysisGraphState(TypedDict, total=False):
    dataset_id: str
    task: Literal["question_answer", "executive_summary"]
    question: str
    dataset: DatasetRecord
    profile: DatasetProfileResponse
    full_profile_context: str
    chart_context: list[dict[str, str | None]]
    retrieved_chunks: list[RetrievedChunk]
    rag_context: str
    answer: str
    summary: str
    supporting_points: list[str]
    data_supported_facts: list[str]
    inference: list[str]
    evidence: list[str]
    highlights: list[str]
    caveats: list[str]
    grounded_in: list[str]


GROUNDED_CONTEXT_SECTIONS = [
    "retrieved semantic chunks derived from schema, missing value profiling, numeric summaries, and categorical summaries",
    "generated chart metadata for supporting context",
]


class QAAgent:
    def __init__(self) -> None:
        self.settings = get_settings()
        graph = StateGraph(AnalysisGraphState)
        graph.add_node("load_dataset", self._load_dataset)
        graph.add_node("prepare_profile_context", self._prepare_profile_context)
        graph.add_node("prepare_chart_context", self._prepare_chart_context)
        graph.add_node("retrieve_relevant_chunks", self._retrieve_relevant_chunks)
        graph.add_node("answer_question", self._answer_question)
        graph.add_node("generate_executive_summary", self._generate_executive_summary)
        graph.add_edge(START, "load_dataset")
        graph.add_edge("load_dataset", "prepare_profile_context")
        graph.add_edge("prepare_profile_context", "prepare_chart_context")
        graph.add_conditional_edges(
            "prepare_chart_context",
            self._route_after_context_prep,
            {
                "retrieve_relevant_chunks": "retrieve_relevant_chunks",
                "generate_executive_summary": "generate_executive_summary",
            },
        )
        graph.add_edge("retrieve_relevant_chunks", "answer_question")
        graph.add_edge("answer_question", END)
        graph.add_edge("generate_executive_summary", END)
        self.graph = graph.compile()

    def answer_question(self, dataset_id: str, question: str) -> QuestionResponse:
        state = self.graph.invoke(
            {
                "dataset_id": dataset_id,
                "task": "question_answer",
                "question": question,
            }
        )
        return QuestionResponse(
            dataset_id=dataset_id,
            question=question,
            answer=state["answer"],
            supporting_points=state["supporting_points"],
            data_supported_facts=state["data_supported_facts"],
            inference=state["inference"],
            evidence=state["evidence"],
            grounded_in=state["grounded_in"],
            caveats=state["caveats"],
            retrieval_top_k=len(state["retrieved_chunks"]),
            retrieved_chunks=[
                RetrievedChunkResponse(
                    chunk_id=item.chunk_id,
                    source_type=item.source_type,
                    column_name=item.column_name,
                    importance_level=item.importance_level,
                    text=item.text,
                    score=item.score,
                )
                for item in state["retrieved_chunks"]
            ],
        )

    def generate_executive_summary(self, dataset_id: str) -> ExecutiveSummaryResponse:
        state = self.graph.invoke(
            {
                "dataset_id": dataset_id,
                "task": "executive_summary",
            }
        )
        return ExecutiveSummaryResponse(
            dataset_id=dataset_id,
            summary=state["summary"],
            highlights=state["highlights"],
            caveats=state["caveats"],
            grounded_in=state["grounded_in"],
        )

    def _load_dataset(self, state: AnalysisGraphState) -> AnalysisGraphState:
        dataset = dataset_store.load_dataset(state["dataset_id"])
        return {"dataset": dataset}

    def _prepare_profile_context(self, state: AnalysisGraphState) -> AnalysisGraphState:
        dataset = state["dataset"]
        profile = profile_service.build_profile(
            dataset.dataframe,
            dataset_id=dataset.dataset_id,
            filename=dataset.filename,
        )
        full_profile_context = json.dumps(
            {
                "dataset_id": profile.dataset_id,
                "filename": profile.filename,
                "overview": profile.overview.model_dump(mode="json"),
                "structure": profile.structure.model_dump(mode="json"),
                "schema": [item.model_dump(mode="json") for item in profile.schema],
                "missing_values": [item.model_dump(mode="json") for item in profile.missing_values],
                "numeric_summary": [item.model_dump(mode="json") for item in profile.numeric_summary],
                "categorical_summary": [
                    item.model_dump(mode="json") for item in profile.categorical_summary
                ],
            },
            indent=2,
        )
        return {"profile": profile, "full_profile_context": full_profile_context}

    def _prepare_chart_context(self, state: AnalysisGraphState) -> AnalysisGraphState:
        profile = state["profile"]
        chart_context = [
            {
                "title": chart.title,
                "chart_type": chart.chart_type,
                "x_field": chart.x_field,
                "y_field": chart.y_field,
            }
            for chart in profile.charts
        ]
        return {"chart_context": chart_context}

    def _route_after_context_prep(
        self, state: AnalysisGraphState
    ) -> Literal["retrieve_relevant_chunks", "generate_executive_summary"]:
        if state["task"] == "question_answer":
            return "retrieve_relevant_chunks"
        return "generate_executive_summary"

    def _retrieve_relevant_chunks(self, state: AnalysisGraphState) -> AnalysisGraphState:
        profile = state["profile"]
        retrieved_chunks = dataset_retriever.retrieve(
            profile=profile,
            query=state["question"],
            top_k=self.settings.retrieval_top_k,
        )
        rag_context = dataset_retriever.build_context(retrieved_chunks)
        return {"retrieved_chunks": retrieved_chunks, "rag_context": rag_context}

    def _answer_question(self, state: AnalysisGraphState) -> AnalysisGraphState:
        if not state["retrieved_chunks"]:
            return {
                "answer": "I do not have enough retrieved evidence to answer that from the dataset safely.",
                "supporting_points": [],
                "data_supported_facts": [],
                "inference": [],
                "evidence": [],
                "caveats": ["No chunks passed the retrieval threshold for this question."],
                "grounded_in": GROUNDED_CONTEXT_SECTIONS,
            }

        prompt = (
            "You must answer strictly from the retrieved evidence below. "
            "Do not use world knowledge to fill gaps. Do not infer causality, business outcomes, "
            "or row-level facts not present in the evidence. Separate directly supported facts "
            "from any cautious inference. If the evidence is insufficient, say so explicitly.\n\n"
            f"Question:\n{state['question']}\n\n"
            f"Retrieved evidence context:\n{state['rag_context']}\n\n"
            f"Chart metadata:\n{json.dumps(state['chart_context'], indent=2)}\n\n"
            "Return JSON with keys: answer, supporting_points, data_supported_facts, "
            "inference, evidence, caveats."
        )
        payload = QAResultPayload.model_validate(
            llm_service.generate_json(
                system_prompt=(
                    "You are a retrieval-grounded data analysis assistant. "
                    "Every claim must be traceable to retrieved evidence. "
                    "Evidence entries must cite chunk ids such as [numeric::sales]."
                ),
                user_prompt=prompt,
            )
        )
        caveats = payload.caveats
        if not caveats:
            caveats = ["The answer is limited to retrieved profile chunks, not raw-row investigation."]
        evidence = payload.evidence
        if not evidence and state["retrieved_chunks"]:
            evidence = [
                f"[{item.chunk_id}] {item.text}"
                for item in state["retrieved_chunks"][: min(3, len(state["retrieved_chunks"]))]
            ]
        return {
            "answer": payload.answer.strip(),
            "supporting_points": payload.supporting_points[:3],
            "data_supported_facts": payload.data_supported_facts[:4],
            "inference": payload.inference[:3],
            "evidence": evidence[:4],
            "caveats": caveats[:3],
            "grounded_in": GROUNDED_CONTEXT_SECTIONS,
        }

    def _generate_executive_summary(self, state: AnalysisGraphState) -> AnalysisGraphState:
        prompt = (
            "Create an executive summary for a non-technical stakeholder using only the dataset "
            "context provided. Mention major patterns, data quality concerns, and what the data "
            "does not tell us. Do not invent business impact, causes, or recommendations not "
            "supported by the context.\n\n"
            f"Structured dataset context:\n{state['full_profile_context']}\n\n"
            f"Chart metadata:\n{json.dumps(state['chart_context'], indent=2)}\n\n"
            "Return JSON with keys: summary, highlights, caveats."
        )
        payload = SummaryResultPayload.model_validate(
            llm_service.generate_json(
                system_prompt=(
                    "You are a careful analytics translator for business stakeholders. "
                    "You summarize only supported facts from structured dataset analysis."
                ),
                user_prompt=prompt,
            )
        )
        highlights = payload.highlights[:3]
        if not highlights:
            highlights = [payload.summary.split(".")[0].strip()]
        caveats = payload.caveats
        if not caveats:
            caveats = ["This summary is based on aggregated profiling outputs, not domain-specific interpretation."]
        return {
            "summary": payload.summary.strip(),
            "highlights": highlights,
            "caveats": caveats[:3],
            "grounded_in": GROUNDED_CONTEXT_SECTIONS,
        }


qa_agent = QAAgent()
