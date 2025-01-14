"""Ollama LLM client implementation."""

import json
from typing import Any, TypeVar, get_args, get_origin

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
        """Initialize Ollama client.

        Args:
            model: The model to run, must already be installed on the host via ollama.
            base_url: The URL of the Ollama API.
        """
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

        This function uses Pydantic to convert any arbitrary class to JSON
        schema. This is unlikely to be fool-proof, but we can deal with issues
        as they arise.

        Args:
            response_model: The Pydantic response model.
            system_prompt: The system prompt.
            user_prompt: The user prompt.
            max_tokens: The maximum number of tokens to allow.

        Returns:
            The response as the requested object.
        """
        default_max_tokens = 4096
        if max_tokens != default_max_tokens:
            msg = "max_tokens has not yet been implemented in Ollama."
            raise NotImplementedError(msg)

        # Use Pydantic for converting an arbitrary class to JSON schema.
        schema = pydantic.create_model(
            response_model.__name__,
            field=(response_model, ...),
        ).model_json_schema()

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
            format=schema,
        )

        data = json.loads(response.message.content)["field"]  # type: ignore[arg-type]
        return _model_and_data_to_object(response_model, data)


def _model_and_data_to_object(cls: type[T], data: Any) -> Any:  # noqa: ANN401
    """Convert JSON data to the specified type.

    Args:
        cls: The target class type.
        data: The JSON data to convert.

    Returns:
        An instance of the target class.
    """
    # Pydantic models
    try:
        return cls.model_validate(data)  # type: ignore[call-arg, attr-defined]
    except AttributeError:
        # Not a Pydantic model.
        pass

    # Lists/tuples
    if cls in (list, tuple):
        return cls(data)  # type: ignore[call-arg]

    if get_origin(cls) in (list, tuple):
        item_types = get_args(cls)
        if len(item_types) > 1:
            msg = "Only one item type may be present in a list/tuple type."
            raise NotImplementedError(msg)
        return cls(_model_and_data_to_object(item_types[0], item) for item in data)  # type: ignore[call-arg]

    # Basic Python types
    if cls in (int, float, str, bool):
        return cls(data)  # type: ignore[call-arg]

    # If we get here, we don't know how to handle this type
    msg = f"Unable to convert data to type {cls}"
    raise ValueError(msg)
