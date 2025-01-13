"""Tools for Large Language Models on AWS Bedrock."""

from typing import Literal, TypeVar

import anthropic
import instructor

from cloai.llm import utils

# This cannot use the type from the Anthropic package as that includes models
# which do not exist on Bedrock.
ANTHROPIC_BEDROCK_MODELS = Literal[
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
]

T = TypeVar("T")


class AnthropicBedrockLlm(utils.LlmBaseClass):
    """Class for Anthropic Large Language models on Bedrock.

    Attributes:
        client: The BedRock client.
        model: The model that is invoked.

    """

    def __init__(
        self,
        model: ANTHROPIC_BEDROCK_MODELS,
        *,
        aws_access_key: str,
        aws_secret_key: str,
        region: str,
    ) -> None:
        """Initializes the BedRock client."""
        self.client = anthropic.AsyncAnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=region,
        )
        self.model = model
        self._instructor = instructor.from_anthropic(self.client)

    async def run(self, system_prompt: str, user_prompt: str) -> str:
        """Runs the model with the given prompts.

        Args:
            system_prompt: The system prompt.
            user_prompt: The user prompt.

        Returns:
            The output text.
        """
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text  # type: ignore[union-attr]

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
        return await self._instructor.chat.completions.create(  # type: ignore[type-var]
            response_model=response_model,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            system=system_prompt,
            model=self.model,
            max_tokens=max_tokens,
        )
