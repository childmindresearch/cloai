"""Utilities for defining large language models."""

import abc
from typing import TypeVar

T = TypeVar("T")


class LlmBaseClass(abc.ABC):
    """The interface required of any large language model client."""

    @abc.abstractmethod
    async def run(self, system_prompt: str, user_prompt: str) -> str:
        """Abstract method for calling a large langauge model."""

    @abc.abstractmethod
    async def call_instructor(
        self,
        response_model: type[T],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> T:
        """Abstract method for calling a large language model with instructor."""
