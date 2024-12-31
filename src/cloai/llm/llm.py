"""This module coalesces all large language models from different microservices."""

import json
from typing import Any, Literal, TypeVar, overload

import pydantic

from cloai import exceptions, logs
from cloai.llm import prompts
from cloai.llm.utils import LlmBaseClass

T = TypeVar("T")

logger = logs.get_logger()


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


class _VerificationResponse(pydantic.BaseModel):
    """A class for an LLM verification response.

    Required for chain of verification.
    """

    statements: tuple[_VerificationStatement]
    model: Any  # More specific types all seem to run into mypy issues.


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
        logger.debug(
            "Calling LLM run with: \n\nSystem Prompt: %s\nUser Prompt: %s",
            system_prompt,
            user_prompt,
        )
        return await self.client.run(system_prompt, user_prompt)

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
        logger.debug(
            "Calling instructor with: \n\nSystem Prompt: %s\n\nUser Prompt: %s",
            system_prompt,
            user_prompt,
        )
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
        response_model: type[T],
        statements: list[str] = ...,
        *,
        max_verifications: int = ...,
        create_new_statements: bool,
    ) -> T:
        pass

    @overload
    async def chain_of_verification(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        statements: None = None,
        *,
        max_verifications: int = ...,
        create_new_statements: Literal[True],
    ) -> T:
        pass

    async def chain_of_verification(  # noqa: PLR0913
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        statements: list[str] | None = None,
        *,
        max_verifications: int = 3,
        create_new_statements: bool = False,
        error_on_iteration_limit: bool = False,
    ) -> T:
        """Runs an LLM prompt that is self-assessed by the LLM.

        Args:
            system_prompt: The system prompt for the initial prompt.
            user_prompt: The user prompt for the initial prompt.
            response_model: The type of the response to return from Instructor.
            statements: Statements to verify the results.
            max_verifications: The maximum number of times to verify the results.
            create_new_statements: If True, generate new statements from the system
                prompt.
            error_on_iteration_limit: If True, raise an exception when the
                iteration limit is reached. Otherwise, returns the last result.

        Returns:
            The edited text result.
        """
        if max_verifications <= 0:
            msg = "max_verifications must be positive"
            raise ValueError(msg)

        if statements is None and not create_new_statements:
            msg = "If no statements are provided, then they must be generated."
            raise ValueError(msg)
        statements = statements or []

        if create_new_statements:
            new_statements = await self._create_statements(system_prompt)
            statements += [statement.statement for statement in new_statements]
        verification_prompt = prompts.chain_of_verification_verify(statements)

        rewrite_prompt = prompts.chain_of_verification_rewrite(
            statements=statements,
            instructions=system_prompt,
            source=user_prompt,
        )

        model = None
        for idx in range(max_verifications):
            logger.debug("Running verification iteration %s", idx)
            if model is None:
                model = await self.call_instructor(
                    response_model=response_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            else:
                model = await self.call_instructor(
                    response_model=response_model,
                    system_prompt=rewrite_prompt,
                    user_prompt=_model_to_string(model),
                )

            text = _model_to_string(model)
            verify = await self.call_instructor(
                response_model=list[_VerificationStatement],
                system_prompt=verification_prompt,
                user_prompt=text,
                max_tokens=4096,
            )

            if idx == 0:
                continue
            if all(verification.correct for verification in verify):
                break
        else:
            if error_on_iteration_limit:
                msg = "Maximum number of iterations reached."
                raise exceptions.IterationLimitError(msg)

        return model  # type: ignore[return-value] # model will never be None as the for-loop is always entered.

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
        statements = await self.call_instructor(
            list[_GeneratedStatement],
            system_prompt=prompts.chain_of_verification_create_statements(),
            user_prompt=instructions,
            max_tokens=4096,
        )
        logger.debug("Created statements: %s", statements)
        return statements


def _model_to_string(model: Any) -> str:  # noqa: ANN401
    """Converts a model to a string.

    Used to handle the dual input of both a Pydantic model and an arbitrary class
    to Instructor.

    Args:
        model: The instructor model to convert.

    Returns:
        The string representation of the input.
    """
    if isinstance(model, pydantic.BaseModel):
        return json.dumps(_recursive_pydantic_model_dump(model))
    return str(model)


def _recursive_pydantic_model_dump(model: pydantic.BaseModel) -> dict[str, Any]:
    """Pydantic model_dump with recursion."""
    dump: dict[str, Any] = {}
    for key in model.model_fields:
        value = getattr(model, key)
        if isinstance(value, pydantic.BaseModel):
            dump[key] = _recursive_pydantic_model_dump(value)
        else:
            dump[key] = value

    return dump
