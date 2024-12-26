"""Integration tests using GPT."""

import os

import pydantic
import pytest

from cloai.llm import bedrock, llm, openai


@pytest.fixture
def openai_model() -> llm.LargeLanguageModel:
    """Creates the GPT client."""
    client = openai.OpenAiLlm(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )
    return llm.LargeLanguageModel(client=client)


@pytest.fixture
def bedrock_anthropic_model() -> llm.LargeLanguageModel:
    """Creates the AWS Anthropic client."""
    client = bedrock.AnthropicBedrockLlm(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        aws_access_key=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_key=os.getenv("AWS_SECRET_KEY"),
        region="us-west-2",
    )
    return llm.LargeLanguageModel(client=client)


@pytest.fixture
def model(
    request: pytest.FixtureRequest,
    openai_model: llm.LargeLanguageModel,
    bedrock_anthropic_model: llm.LargeLanguageModel,
) -> llm.LargeLanguageModel:
    """Fetches the LLM."""
    name = request.param
    if name == "openai":
        return openai_model
    if name == "bedrock":
        return bedrock_anthropic_model

    msg = "Wrong model name."
    raise ValueError(msg)


@pytest.mark.parametrize("model", ["openai", "bedrock"], indirect=True)
@pytest.mark.asyncio
async def test_run(model: llm.LargeLanguageModel) -> None:
    """Test the run command."""
    actual = await model.run(
        system_prompt="Repeat the user's message back to them.",
        user_prompt="Hello world!",
    )

    assert isinstance(actual, str)
    assert len(actual) > 0


@pytest.mark.parametrize("model", ["openai", "bedrock"], indirect=True)
@pytest.mark.asyncio
async def test_call_instructor(model: llm.LargeLanguageModel) -> None:
    """Test the call_instructor command."""

    class Response(pydantic.BaseModel):
        grade: int = pydantic.Field(..., lt=10, gt=0)

    system_prompt = "Return the user message."
    user_prompt = "{'grade': 3}"

    actual = await model.call_instructor(
        response_model=Response,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    assert isinstance(actual, Response)


@pytest.mark.parametrize("model", ["openai", "bedrock"], indirect=True)
@pytest.mark.asyncio
async def test_chain_of_density(model: llm.LargeLanguageModel) -> None:
    """Test the chain_of_density command."""
    text = """
        Lorem ipsum is a placeholder text commonly used in design, publishing,
        and printing to demonstrate the visual form of a document or typeface without
        relying on meaningful content. The text begins with "Lorem ipsum dolor sit
        amet..." and is derived from a scrambled version of "De finibus bonorum et
        malorum" (On the Ends of Good and Evil), a philosophical work by Cicero
        written in 45 BC.

        The purpose of lorem ipsum is to help designers and publishers focus on the
        visual elements of their work, such as layout, typography, and spacing,
        without being distracted by readable content. The text appears to be Latin
        but is intentionally nonsensical, making it neutral and less likely to draw
        attention away from the design elements being evaluated.

        This dummy text has been an industry standard since the 1500s when an unknown
        printer scrambled Cicero's text to make a type specimen book, and it continues
        to be widely used in digital and print media today."""

    actual = await model.chain_of_density(text, repeats=2)

    assert isinstance(actual, str)
    assert len(actual) > 0


@pytest.mark.parametrize("model", ["openai", "bedrock"], indirect=True)
@pytest.mark.asyncio
async def test_chain_of_verification_str(model: llm.LargeLanguageModel) -> None:
    """Test the chain_of_verification command."""
    text = "Lea is 9 years old. She likes riding horses."

    actual = await model.chain_of_verification(
        system_prompt="What animal does the person in question like?",
        user_prompt=text,
        create_new_statements=True,
        response_model=str,
    )

    assert isinstance(actual, str)
    assert "horse" in actual.lower()


@pytest.mark.parametrize("model", ["openai", "bedrock"], indirect=True)
@pytest.mark.asyncio
async def test_chain_of_verification_model(model: llm.LargeLanguageModel) -> None:
    """Test the chain_of_verification command."""
    text = "Lea is 9 years old. She likes riding horses."

    class Response(pydantic.BaseModel):
        animal: str
        child_name: str
        child_age: int

    actual = await model.chain_of_verification(
        system_prompt="What is the child's name, age, and what animal does they like?",
        user_prompt=text,
        create_new_statements=True,
        response_model=Response,
    )

    assert isinstance(actual, Response)
    assert "horse" in actual.animal.lower()
    assert "lea" in actual.child_name.lower()
    assert actual.child_age == 9  # noqa: PLR2004
