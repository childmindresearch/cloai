"""Ollama LLM client implementation."""

from typing import TypeVar

import ollama
import pydantic

from cloai.llm.utils import LlmBaseClass

T = TypeVar("T")


class OllamaLlm(LlmBaseClass):
    """Client for Ollama API."""

    def __init__(
        self,
        model: str,
        base_url: str,
    ) -> None:
        """Initialize Ollama client."""
        self.model = model
        self.client = ollama.AsyncClient(host=base_url)

    async def run(self, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama model."""
        response = await self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )
        return response["message"]["content"]

    async def call_instructor(
        self,
        response_model: type[T],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> T:
        """Run a type-safe large language model query.

        Args:
            response_model: The Pydantic response model.
            system_prompt: The system prompt.
            user_prompt: The user prompt.
            max_tokens: The maximum number of tokens to allow.
        """
        default_max_tokens = 4096
        if max_tokens != default_max_tokens:
            msg = "max_tokens has not yet been implemented in Ollama."
            raise NotImplementedError(msg)

        if not isinstance(response_model, pydantic.BaseModel):
            msg = "Ollama is not compatible with non-Pydantic Basemodel inputs yet."
            raise NotImplementedError(msg)

        response = await self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\nYou must respond in JSON format matching "
                        f"this schema: {response_model.model_json_schema()}"
                    ),
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            format=response_model.model_json_schema(),
        )
        return response_model.model_validate_json(response)
