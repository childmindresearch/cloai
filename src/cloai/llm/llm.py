"""This module coalesces all large language models from different microservices."""

import asyncio
from collections.abc import Iterable
from typing import Literal, overload

import pydantic

from cloai.llm import prompts, utils
from cloai.llm.utils import LlmBaseClass


class _GeneratedStatement(pydantic.BaseModel):
    """A class for a statement about the correctness of an LLM result.

    Required for chain of verification.
    """

    statement: str = pydantic.Field(
        ...,
        description="A True or False statement about the text.",
    )

    @pydantic.field_validator("statement")
    @classmethod
    def statement_validation(cls, value: str) -> str:
        """Check whether the phrase is actually a statement."""
        if value[0].isnumeric():
            msg = "statements should not be numbered."
            raise ValueError(msg)
        return value


class _VerificationStatement(pydantic.BaseModel):
    """A class for a statement verifying the correctness of an LLM result.

    Required for chain of verification.
    """

    statement: _GeneratedStatement = pydantic.Field(
        ...,
        description="A True or False statement about the text.",
    )
    correct: bool = pydantic.Field(
        ...,
        description="True if the answer to the statement is true, False otherwise.",
    )


class _RewrittenText(pydantic.BaseModel):
    """Class for rewriting text based on verification statements.

    Required for chain of verification.
    """

    text: str = pydantic.Field(..., description="The edited text.")
    statements: tuple[_VerificationStatement] = pydantic.Field(
        ...,
        description=(
            "The statements along with whether they are True or False about the "
            "edited text."
        ),
    )


class LargeLanguageModel(pydantic.BaseModel):
    """Llm class that provides access to all available LLMs.

    Attributes:
        client: The client for the large language model.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)
    client: LlmBaseClass

    async def run(self, system_prompt: str, user_prompt: str) -> str:
        """Runs the model with the given prompts.

        Args:
            system_prompt: The system prompt.
            user_prompt: The user prompt.

        Returns:
            The output text.
        """
        return await self.client.run(system_prompt, user_prompt)

    @overload
    async def call_instructor(
        self,
        response_model: type[utils.InstructorResponse],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = ...,
    ) -> utils.InstructorResponse: ...

    @overload
    async def call_instructor(
        self,
        response_model: type[list[utils.InstructorResponse]],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = ...,
    ) -> list[utils.InstructorResponse]: ...

    async def call_instructor(
        self,
        response_model: type[utils.InstructorResponse]
        | type[list[utils.InstructorResponse]],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> utils.InstructorResponse | list[utils.InstructorResponse]:
        """Run a type-safe large language model query.

        Args:
            response_model: The Pydantic response model.
            system_prompt: The system prompt.
            user_prompt: The user prompt.
            max_tokens: The maximum number of tokens to allow.
        """
        return await self.client.call_instructor(
            response_model,
            system_prompt,
            user_prompt,
            max_tokens,
        )

    @overload
    async def chain_of_verification(
        self,
        system_prompt: str,
        user_prompt: str,
        statements: list[str] = ...,
        max_verifications: int = ...,
        *,
        create_new_statements: bool,
    ) -> str:
        pass

    @overload
    async def chain_of_verification(
        self,
        system_prompt: str,
        user_prompt: str,
        statements: None = None,
        max_verifications: int = ...,
        *,
        create_new_statements: Literal[True],
    ) -> str:
        pass

    async def chain_of_verification(
        self,
        system_prompt: str,
        user_prompt: str,
        statements: list[str] | None = None,
        max_verifications: int = 3,
        *,
        create_new_statements: bool = False,
    ) -> str:
        """Runs an LLM prompt that is self-assessed by the LLM.

        Args:
            system_prompt: The system prompt for the initial prompt.
            user_prompt: The user prompt for the initial prompt.
            statements: Statements to verify the results. Defaults to None.
            max_verifications: The maximum number of times to verify the results.
                Defaults to 3.
            create_new_statements: If True, generate new statements from the system
                prompt. Defaults to False.

        Returns:
            The edited text result.
        """
        if statements is None and not create_new_statements:
            msg = "If no statements are provided, then they must be generated."
            raise ValueError(msg)
        statements = statements or []

        text_promise = self.run(system_prompt, user_prompt)
        if create_new_statements:
            statements_promise = self._create_statements(system_prompt)
            text, new_statements = await asyncio.gather(
                text_promise,
                statements_promise,
            )
            statements += [statement.statement for statement in new_statements]
        else:
            text = await text_promise

        for _ in range(max_verifications):
            rewrite = await self._verify(
                text,
                statements,
                user_prompt,
            )
            if all(verification.correct for verification in rewrite.statements):
                break
            text = rewrite.text

        return text

    async def chain_of_density(
        self,
        text: str,
        repeats: int = 3,
        *,
        max_informative_entities: int = 3,
    ) -> str:
        """Iterative summarization of an input text.

        Chain of density performs an iterative summarization of an input text. It
        should,
        in theory, provide more robust summaries than single-shot approaches.

        Args:
            text: The input text to summarize
            repeats: The number of times to summarize.
            max_informative_entities: The maximum number of new entities to include in
                each summary.

        Returns:
            The summarized text.

        References:
            Adams, G., Fabbri, A. R., Ladhak, F., Lehman, E., & Elhadad, N. (2023,
            December). From sparse to dense: GPT-4 summarization with chain of
            density prompting. In Proceedings of the Conference on Empirical Methods
            in Natural Language Processing. Conference on Empirical Methods in
            Natural Language Processing (Vol. 2023, No. 4th New Frontier
            Summarization Workshop, p. 68).
        """
        system_prompt = prompts.chain_of_density(article=text)
        if repeats < 1:
            msg = "Repeat count must be positive"
            raise ValueError(msg)

        class Response(pydantic.BaseModel):
            missing_informative_entity: list[str] = pydantic.Field(
                ...,
                min_length=1,
                max_length=max_informative_entities,
            )
            summary: str

        summary = ""
        for _ in range(repeats):
            user_prompt = f"Current Summary: {summary}"
            iteration = await self.call_instructor(
                response_model=Response,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            summary = iteration.summary
        return summary

    async def _create_statements(self, instructions: str) -> list[_GeneratedStatement]:
        """Creates statements for prompt result validation.

        Args:
            instructions: The instructions provided to the model, commonly
                the system prompt.

        Returns:
            List of verification statements as strings.
        """
        return await self.call_instructor(
            list[_GeneratedStatement],
            system_prompt=prompts.chain_of_verification_create_statements(),
            user_prompt=instructions,
            max_tokens=4096,
        )

    async def _verify(
        self,
        text: str,
        statements: Iterable[str],
        source: str,
    ) -> _RewrittenText:
        return await self.call_instructor(
            response_model=_RewrittenText,
            system_prompt=prompts.chain_of_verification_verify(statements, source),
            user_prompt=text,
            max_tokens=4096,
        )
