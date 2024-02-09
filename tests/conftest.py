"""Test configurations."""
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
    mock_client = mocker.AsyncMock(
        spec=openai_api.openai.AsyncOpenAI,
        audio=mock_audio,
        images=mock_images,
        chat=mock_chat,
    )
    return mocker.patch(
        "cloai.openai_api.openai.AsyncOpenAI",
        return_value=mock_client,
    )
