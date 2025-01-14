"""Tests for the bedrock large language models.

This mocks all the interfaces to remote servers. As such, they're not great tests,
but they are the best we can do without connecting to remote servers on every test.
"""

import json
import types
from unittest import mock

import pydantic
import pytest
import pytest_mock

from cloai.llm import bedrock, ollama, openai, utils

TEST_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
TEST_SYSTEM_PROMPT = "You are a helpful assistant."
TEST_USER_PROMPT = "What is 2+2?"
TEST_RUN_RESPONSE = "Hello world!"

LLM_TYPE = (
    bedrock.AnthropicBedrockLlm | openai.OpenAiLlm | openai.AzureLlm | ollama.OllamaLlm
)
llms = ("azure", "anthropic_bedrock", "openai", "ollama")


class _TestResponse(pydantic.BaseModel):
    answer: str


@pytest.fixture
def mock_anthropic_client(mocker: pytest_mock.MockerFixture) -> mock.MagicMock:
    """Mock the anthropic client."""
    response = types.SimpleNamespace(
        content=[
            types.SimpleNamespace(
                text=TEST_RUN_RESPONSE,
            ),
        ],
    )

    mock_client = mocker.MagicMock()
    mock_client.messages = mocker.MagicMock()
    mock_client.messages.create = mocker.AsyncMock(return_value=response)
    mocker.patch("anthropic.AsyncAnthropicBedrock", return_value=mock_client)
    return mock_client


@pytest.fixture
def mock_anthropic_instructor(mocker: pytest_mock.MockerFixture) -> mock.MagicMock:
    """Mock the anthropic instructor calls."""
    mock_inst = mocker.MagicMock()
    mock_inst.chat = mocker.MagicMock()
    mock_inst.chat.completions = mocker.MagicMock()
    mock_inst.chat.completions.create = mocker.AsyncMock()
    mocker.patch("instructor.from_anthropic", return_value=mock_inst)
    return mock_inst


@pytest.fixture
def anthropic_bedrock_llm(
    mock_anthropic_client: mock.MagicMock,
    mock_anthropic_instructor: pytest_mock.MockerFixture,
) -> bedrock.AnthropicBedrockLlm:
    """Create the mocked anthropic bedrock llm."""
    return bedrock.AnthropicBedrockLlm(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        aws_access_key="test_access_key",
        aws_secret_key="test_secret_key",  # noqa: S106
        region="us-east-1",
    )


@pytest.fixture
def mock_openai_base(mocker: pytest_mock.MockerFixture) -> mock.MagicMock:
    """Base mocks for the Azure and regular OpenAI client."""
    response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=TEST_RUN_RESPONSE,
                ),
            ),
        ],
    )

    mock_client = mocker.MagicMock()
    mock_client.chat = mocker.MagicMock()
    mock_client.chat.completions = mocker.MagicMock()
    mock_client.chat.completions.create = mocker.AsyncMock(
        return_value=response,
    )
    return mock_client


@pytest.fixture
def mock_openai_client(
    mocker: pytest_mock.MockerFixture,
    mock_openai_base: mock.MagicMock,
) -> mock.MagicMock:
    """Mock the openai client."""
    mocker.patch("openai.AsyncOpenAI", return_value=mock_openai_base)
    return mock_openai_base


@pytest.fixture
def mock_azure_client(
    mocker: pytest_mock.MockerFixture,
    mock_openai_base: mock.MagicMock,
) -> mock.MagicMock:
    """Mock the openai client."""
    mocker.patch("openai.AsyncAzureOpenAI", return_value=mock_openai_base)
    return mock_openai_base


@pytest.fixture
def mock_openai_instructor(mocker: pytest_mock.MockerFixture) -> mock.MagicMock:
    """Mock the openai instructor calls."""
    mock_inst = mocker.MagicMock()
    mock_inst.chat = mocker.MagicMock()
    mock_inst.chat.completions = mocker.MagicMock()
    mock_inst.chat.completions.create = mocker.AsyncMock()
    mocker.patch("instructor.from_openai", return_value=mock_inst)
    return mock_inst


@pytest.fixture
def openai_llm(
    mock_openai_client: mock.MagicMock,
    mock_openai_instructor: pytest_mock.MockerFixture,
) -> openai.OpenAiLlm:
    """Create the mocked openai llm."""
    return openai.OpenAiLlm(
        model=TEST_MODEL,
        api_key="fake",
    )


@pytest.fixture
def ollama_llm(mocker: pytest_mock.MockerFixture) -> ollama.OllamaLlm:
    """Create the mocked anthropic bedrock llm."""
    response = {"message": {"content": TEST_RUN_RESPONSE}}
    mocker.patch("ollama.AsyncClient.chat", return_value=response)
    return ollama.OllamaLlm(
        model=TEST_MODEL,
        base_url="somethinglocal",
    )


@pytest.fixture
def azure_llm(
    mock_azure_client: mock.MagicMock,
    mock_openai_instructor: pytest_mock.MockerFixture,
) -> openai.AzureLlm:
    """Create the mocked openai llm."""
    return openai.AzureLlm(
        api_key="fake",
        endpoint="fake",
        api_version="fake",
        deployment=TEST_MODEL,
    )


@pytest.fixture
def llm(
    request: pytest.FixtureRequest,
    openai_llm: openai.OpenAiLlm,
    azure_llm: openai.AzureLlm,
    anthropic_bedrock_llm: bedrock.AnthropicBedrockLlm,
    ollama_llm: ollama.OllamaLlm,
) -> utils.LlmBaseClass:
    """Create the mocked llm."""
    name = request.param
    if name == "openai":
        return openai_llm
    if name == "anthropic_bedrock":
        return anthropic_bedrock_llm
    if name == "azure":
        return azure_llm
    if name == "ollama":
        return ollama_llm
    raise NotImplementedError


@pytest.mark.parametrize("llm", llms, indirect=True)
def test_initialization(llm: LLM_TYPE) -> None:
    """Test proper initialization of ClaudeLlm."""
    assert llm.model == TEST_MODEL
    assert llm.client is not None


@pytest.mark.parametrize("llm", llms, indirect=True)
@pytest.mark.asyncio
async def test_run_method(
    llm: LLM_TYPE,
    mock_anthropic_client: mock.MagicMock,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test the run method."""
    mock_content = mocker.MagicMock()
    mock_content.text = "4"
    mock_message = mocker.MagicMock()
    mock_message.content = [mock_content]

    result = await llm.run(
        system_prompt=TEST_SYSTEM_PROMPT,
        user_prompt=TEST_USER_PROMPT,
    )

    assert result == TEST_RUN_RESPONSE


@pytest.mark.parametrize("llm", llms, indirect=True)
@pytest.mark.asyncio
async def test_call_instructor_method(
    llm: LLM_TYPE,
) -> None:
    """Test the call_instructor method."""
    expected_response = _TestResponse(answer="4")
    if isinstance(llm, ollama.OllamaLlm):
        # Ollama doesn't use instructor and therefore requires custom handling.
        class Content(pydantic.BaseModel):
            content: str = json.dumps({"field": _TestResponse(answer="4").model_dump()})

        class Response(pydantic.BaseModel):
            message: Content = Content()

        llm.client.chat.return_value = Response()  # type: ignore[attr-defined]
    else:
        llm._instructor.chat.completions.create.return_value = expected_response  # type: ignore[call-overload, attr-defined]

    result = await llm.call_instructor(
        _TestResponse,
        system_prompt=TEST_SYSTEM_PROMPT,
        user_prompt=TEST_USER_PROMPT,
    )

    assert isinstance(result, _TestResponse), "Invalid type."
    assert result == expected_response, "Invalid response."
