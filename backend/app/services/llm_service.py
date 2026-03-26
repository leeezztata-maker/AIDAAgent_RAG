from __future__ import annotations

import json

from openai import OpenAI

from app.core.config import get_settings


class LLMConfigurationError(RuntimeError):
    pass


class LLMOutputError(RuntimeError):
    pass


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _client(self) -> OpenAI:
        if not self.settings.openai_api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is not configured.")
        return OpenAI(api_key=self.settings.openai_api_key)

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        client = self._client()
        response = client.responses.create(
            model=self.settings.openai_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text.strip()

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        response_text = self.generate_text(
            system_prompt=system_prompt,
            user_prompt=(
                f"{user_prompt}\n\n"
                "Return valid JSON only. Do not include markdown fences or extra commentary."
            ),
        )
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            start = response_text.find("{")
            end = response_text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise LLMOutputError("Model response was not valid JSON.") from None
            try:
                return json.loads(response_text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise LLMOutputError("Model response was not valid JSON.") from exc


llm_service = LLMService()
