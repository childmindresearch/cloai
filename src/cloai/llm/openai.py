"""Tools for Large Language Models with an OpenAI interface."""

import abc
from typing import TypeVar

import instructor
import openai
from openai.types import chat_model

from cloai.llm import utils

T = TypeVar("T")


class _OpenAiBase(utils.LlmBaseClass, abc.ABC):
    """A class to interact with OpenAI Language Model service."""

    model: str
    client: openai.AsyncAzureOpenAI | openai.AsyncOpenAI
    _instructor: instructor.AsyncInstructor

    async def run(self, system_prompt: str, user_prompt: str) -> str:
        """Runs the model with the given prompts.

        Args:
            system_prompt: The system prompt.
            user_prompt: The user prompt.

        Returns:
            The output text.

        Raises:
            ValueError if the response is invalid.
        """
        system_message = {
            "role": "system",
            "content": system_prompt,
        }
        user_message = {
            "role": "user",
            "content": user_prompt,
        }
        response = await self.client.chat.completions.create(
            messages=[system_message, user_message],  # type: ignore[list-item]
            model=self.model,
        )
        if not response.choices[0].message.content:
            msg = "No response from Azure OpenAI."
            raise ValueError(msg)
        return response.choices[0].message.content

    async def call_instructor(
        self,
        response_model: type[T],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> T:
        """Run a type-safe large language model query.

        Args:
            response_model: The output type.
            system_prompt: The system prompt.
            user_prompt: The user prompt.
            max_tokens: The maximum number of tokens to allow.
        """
        return await self._instructor.chat.completions.create(  # type: ignore[type-var]
            response_model=response_model,
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
            model=self.model,
            max_tokens=max_tokens,
        )


class AzureLlm(_OpenAiBase):
    """Azure OpenAI Large Language Models.

    This class serves as an interface for Azure Large Language Models.
    Both this class and OpenAiLlm inherit from the same base class as,
    apart from initialization, the LLM clients behave the same.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str,
        deployment: str,
    ) -> None:
        """Initialize the Azure Language Model client.

        Args:
            api_key: The Azure OpenAI API key.
            endpoint: The model's endpoint.
            api_version: The Azure OpenAI API version.
            deployment: The Azure OpenAI deployment.
        """
        self.client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.model = deployment
        self._instructor = instructor.from_openai(self.client)


class OpenAiLlm(_OpenAiBase):
    """OpenAI Large Language Models.

    This class serves as a generic interface to any model that uses OpenAIs
    interface such as OpenAI's models, Ollama, and LiteLLM.

    Both this class and AzureLlm inherit from the same base class as,
    apart from initialization, the LLM clients behave the same.
    """

    def __init__(
        self,
        model: chat_model.ChatModel | str,
        api_key: str,
        base_url: str | None = None,
        instructor_mode: instructor.Mode = instructor.Mode.TOOLS,
    ) -> None:
        """Initialize the OpenAI Language Model client.

        Args:
            model: The model to use for the language model.
            api_key: The OpenAI API key.
            base_url: The URL for the endpoint, defaults to OpenAI's endpoint.
            instructor_mode: The instructor mode to use.
        """
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._instructor = instructor.from_openai(
            self.client,
            mode=instructor_mode,
        )
