"""Utilities for defining large language models."""

import abc
from typing import TypeVar, overload

import pydantic

InstructorResponse = TypeVar("InstructorResponse", bound=pydantic.BaseModel)


class LlmBaseClass(abc.ABC):
    """The interface required of any large language model client."""

    @abc.abstractmethod
    async def run(self, system_prompt: str, user_prompt: str) -> str:
        """Abstract method for calling a large langauge model."""

    @overload
    async def call_instructor(
        self,
        response_model: type[InstructorResponse],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> InstructorResponse: ...

    @overload
    async def call_instructor(
        self,
        response_model: type[list[InstructorResponse]],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> list[InstructorResponse]: ...
    @abc.abstractmethod
    async def call_instructor(
        self,
        response_model: type[InstructorResponse] | type[list[InstructorResponse]],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> InstructorResponse | list[InstructorResponse]:
        """Abstract method for calling a large language model with instructor."""
