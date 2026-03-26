from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.schemas import DatasetProfileResponse
from app.rag.retriever import RetrievedChunk, dataset_retriever
from app.services.llm_service import llm_service


class EvaluationAnswerPayload(BaseModel):
    answer: str = Field(min_length=1)
    supporting_points: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    data_supported_facts: list[str] = Field(default_factory=list)
    inference: list[str] = Field(default_factory=list)


class RAGEvaluationRunner:
    """Compare full-context prompting vs retrieval-grounded prompting.

    Why this design?
    - Interviewers usually care less about absolute benchmark rigor in an MVP and more about
      whether you can reason about evaluation loops.
    - This module creates a repeatable harness with lightweight heuristics and persisted logs.
    - The trade-off is that correctness still benefits from human review, which we surface clearly.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def evaluate(
        self,
        profile: DatasetProfileResponse,
        questions: list[str],
    ) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        dataset_retriever.index_profile(profile)

        full_context = json.dumps(profile.model_dump(mode="json"), indent=2)
        for question in questions:
            rag_chunks = dataset_retriever.retrieve(profile=profile, query=question)
            rag_context = dataset_retriever.build_context(rag_chunks)

            full_context_answer = self._ask_full_context(question, full_context)
            rag_answer = self._ask_rag(question, rag_context)

            records.append(
                {
                    "question": question,
                    "full_context": self._score_answer(full_context_answer, []),
                    "rag": self._score_answer(rag_answer, rag_chunks),
                }
            )

        output = {
            "dataset_id": profile.dataset_id,
            "questions_evaluated": len(questions),
            "records": records,
        }
        self._write_log(profile.dataset_id, output)
        return output

    def _ask_full_context(self, question: str, context: str) -> EvaluationAnswerPayload:
        prompt = (
            "Answer the question using the full dataset profile below. "
            "Return JSON with keys: answer, supporting_points, evidence, caveats, "
            "data_supported_facts, inference.\n\n"
            f"Question:\n{question}\n\n"
            f"Dataset profile:\n{context}"
        )
        return EvaluationAnswerPayload.model_validate(
            llm_service.generate_json(
                system_prompt="You are evaluating a full-context prompting baseline.",
                user_prompt=prompt,
            )
        )

    def _ask_rag(self, question: str, context: str) -> EvaluationAnswerPayload:
        prompt = (
            "Answer the question using retrieved evidence only. "
            "Return JSON with keys: answer, supporting_points, evidence, caveats, "
            "data_supported_facts, inference.\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved evidence:\n{context}"
        )
        return EvaluationAnswerPayload.model_validate(
            llm_service.generate_json(
                system_prompt="You are evaluating a retrieval-grounded prompting strategy.",
                user_prompt=prompt,
            )
        )

    def _score_answer(
        self,
        answer: EvaluationAnswerPayload,
        retrieved_chunks: list[RetrievedChunk],
    ) -> dict[str, Any]:
        evidence_count = len(answer.evidence)
        inference_count = len(answer.inference)
        caveat_count = len(answer.caveats)
        retrieved_count = len(retrieved_chunks)

        hallucination_risk = 0.0
        if inference_count > evidence_count:
            hallucination_risk += 0.4
        if evidence_count == 0:
            hallucination_risk += 0.4
        if caveat_count == 0:
            hallucination_risk += 0.2

        consistency = 1.0 if answer.data_supported_facts else 0.5
        if retrieved_count and evidence_count:
            consistency += 0.25
        consistency = min(consistency, 1.0)

        correctness = 0.5
        if answer.data_supported_facts:
            correctness += 0.25
        if answer.evidence:
            correctness += 0.25

        return {
            "answer": answer.model_dump(mode="json"),
            "metrics": {
                "correctness_estimate": round(correctness, 2),
                "consistency_estimate": round(consistency, 2),
                "hallucination_risk_estimate": round(hallucination_risk, 2),
            },
        }

    def _write_log(self, dataset_id: str, payload: dict[str, Any]) -> None:
        output_dir = Path(self.settings.exports_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"rag_vs_full_context_{dataset_id}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


rag_evaluation_runner = RAGEvaluationRunner()
