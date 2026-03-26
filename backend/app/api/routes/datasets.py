from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.models.schemas import (
    ChartListResponse,
    DatasetContextResponse,
    DatasetProfileResponse,
    DatasetUploadResponse,
    EvaluationRequest,
    EvaluationResponse,
    ExecutiveSummaryResponse,
    QuestionRequest,
    QuestionResponse,
)
from app.evaluation.rag_vs_full_context import rag_evaluation_runner
from app.services.dataset_store import DatasetNotFoundError, dataset_store
from app.services.llm_service import LLMConfigurationError, LLMOutputError
from app.services.profiling import profile_service
from app.services.qa_agent import qa_agent

router = APIRouter()


def _load_profile(dataset_id: str) -> DatasetProfileResponse:
    dataset = dataset_store.load_dataset(dataset_id)
    return profile_service.build_profile(
        dataset.dataframe,
        dataset_id=dataset.dataset_id,
        filename=dataset.filename,
    )


@router.post("/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported.")

    try:
        dataset = await dataset_store.save_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    profile = profile_service.build_profile(
        dataset.dataframe,
        dataset_id=dataset.dataset_id,
        filename=dataset.filename,
    )
    return DatasetUploadResponse(
        dataset_id=dataset.dataset_id,
        filename=dataset.filename,
        rows=profile.overview.row_count,
        columns=profile.overview.column_count,
    )


@router.get("/{dataset_id}/context", response_model=DatasetContextResponse)
def get_dataset_context(dataset_id: str) -> DatasetContextResponse:
    try:
        profile = _load_profile(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return DatasetContextResponse(
        dataset_id=dataset_id,
        filename=profile.filename,
        overview=profile.overview,
        structure=profile.structure,
        schema=profile.schema,
        missing_values=profile.missing_values,
    )


@router.get("/{dataset_id}/profile", response_model=DatasetProfileResponse)
def get_dataset_profile(dataset_id: str) -> DatasetProfileResponse:
    try:
        return _load_profile(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{dataset_id}/analysis", response_model=DatasetProfileResponse)
def run_full_analysis(dataset_id: str) -> DatasetProfileResponse:
    try:
        return _load_profile(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{dataset_id}/charts", response_model=ChartListResponse)
def list_charts(dataset_id: str) -> ChartListResponse:
    try:
        profile = _load_profile(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ChartListResponse(dataset_id=dataset_id, charts=profile.charts)


@router.post("/{dataset_id}/ask", response_model=QuestionResponse)
def ask_question(dataset_id: str, request: QuestionRequest) -> QuestionResponse:
    try:
        return qa_agent.answer_question(dataset_id, request.question)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMOutputError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/{dataset_id}/executive-summary", response_model=ExecutiveSummaryResponse)
def generate_executive_summary(dataset_id: str) -> ExecutiveSummaryResponse:
    try:
        return qa_agent.generate_executive_summary(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMOutputError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/{dataset_id}/evaluate-rag", response_model=EvaluationResponse)
def evaluate_rag_vs_full_context(
    dataset_id: str,
    request: EvaluationRequest,
) -> EvaluationResponse:
    try:
        profile = _load_profile(dataset_id)
        payload = rag_evaluation_runner.evaluate(profile=profile, questions=request.questions)
        return EvaluationResponse.model_validate(payload)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMOutputError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
