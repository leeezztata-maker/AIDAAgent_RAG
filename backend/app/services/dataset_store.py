from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import UploadFile

from app.core.config import get_settings


class DatasetNotFoundError(FileNotFoundError):
    pass


@dataclass(slots=True)
class DatasetRecord:
    dataset_id: str
    filename: str
    path: Path
    dataframe: pd.DataFrame


class DatasetStore:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def save_upload(self, file: UploadFile) -> DatasetRecord:
        content = await file.read()
        if not content:
            raise ValueError("Uploaded file is empty.")

        dataframe = self._read_csv(content)
        dataset_id = uuid4().hex
        filename = file.filename or f"{dataset_id}.csv"
        path = self.settings.uploads_dir / f"{dataset_id}.csv"
        path.write_bytes(content)
        self._metadata_path(dataset_id).write_text(json.dumps({"filename": filename}), encoding="utf-8")

        return DatasetRecord(
            dataset_id=dataset_id,
            filename=filename,
            path=path,
            dataframe=dataframe,
        )

    def load_dataset(self, dataset_id: str) -> DatasetRecord:
        path = self.settings.uploads_dir / f"{dataset_id}.csv"
        if not path.exists():
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' was not found.")

        dataframe = pd.read_csv(path)
        filename = self._load_filename(dataset_id, fallback=path.name)
        return DatasetRecord(
            dataset_id=dataset_id,
            filename=filename,
            path=path,
            dataframe=dataframe,
        )

    @staticmethod
    def _read_csv(content: bytes) -> pd.DataFrame:
        try:
            dataframe = pd.read_csv(BytesIO(content))
        except Exception as exc:  # noqa: BLE001
            raise ValueError("The uploaded file could not be parsed as a CSV.") from exc

        if dataframe.empty:
            raise ValueError("The uploaded CSV has no rows.")

        return dataframe

    def _metadata_path(self, dataset_id: str) -> Path:
        return self.settings.uploads_dir / f"{dataset_id}.json"

    def _load_filename(self, dataset_id: str, fallback: str) -> str:
        metadata_path = self._metadata_path(dataset_id)
        if not metadata_path.exists():
            return fallback

        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return fallback

        filename = payload.get("filename")
        return filename if isinstance(filename, str) and filename else fallback


dataset_store = DatasetStore()
