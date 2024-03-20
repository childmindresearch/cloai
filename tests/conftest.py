"""Test configurations."""

import dataclasses
import os
from unittest import mock

import pytest
import pytest_mock

from cloai import openai_api


def pytest_configure() -> None:
    """Configure pytest with the necessary environment variables.

    Args:
        config: The pytest configuration object.

    """
    os.environ["OPENAI_API_KEY"] = "API_KEY"


@dataclasses.dataclass
class EmbeddingData:
    """A mock embedding data."""

    embedding: list[float] = dataclasses.field(default_factory=lambda: [1.0, 2.0, 3.0])


@dataclasses.dataclass
class EmbeddingResponse:
    """A mock embedding response."""

    data: list[EmbeddingData] = dataclasses.field(
        default_factory=lambda: [EmbeddingData()],
    )


@pytest.fixture()
def mock_openai(mocker: pytest_mock.MockFixture) -> mock.MagicMock:
    """Mocks the OpenAI client."""
    mock_audio = mocker.AsyncMock(
        audio=mocker.AsyncMock(
            speech=mocker.MagicMock(create=mocker.MagicMock()),
            transcriptions=mocker.MagicMock(create=mocker.MagicMock()),
        ),
    )
    mock_images = mocker.MagicMock(generate=mocker.AsyncMock())
    mock_chat = mocker.MagicMock(
        completions=mocker.MagicMock(
            create=mocker.AsyncMock(),
        ),
    )
    mock_embedding = mocker.MagicMock(
        create=mocker.AsyncMock(return_value=EmbeddingResponse()),
    )
    mock_client = mocker.AsyncMock(
        spec=openai_api.openai.AsyncOpenAI,
        audio=mock_audio,
        images=mock_images,
        chat=mock_chat,
        embeddings=mock_embedding,
    )
    return mocker.patch(
        "cloai.openai_api.openai.AsyncOpenAI",
        return_value=mock_client,
    )
