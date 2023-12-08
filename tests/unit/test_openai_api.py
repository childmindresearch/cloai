"""Unit tests for the OpenAI API module."""
import tempfile
from typing import Any

import pytest
import pytest_mock

from oai import openai_api


@pytest.fixture()
def mock_openai(mocker: pytest_mock.MockerFixture) -> pytest_mock.MockFixture:
    """Mocks the OpenAI client."""
    mock_speech_create = mocker.MagicMock()
    mock_transcriptions_create = mocker.MagicMock()
    mock_audio_speech = mocker.MagicMock(
        speech=mocker.MagicMock(create=mock_speech_create),
        transcriptions=mocker.MagicMock(create=mock_transcriptions_create),
    )
    mock_audio = mocker.MagicMock(audio=mock_audio_speech)
    mock_images = mocker.MagicMock(generate=mocker.MagicMock())
    mock_client = mocker.MagicMock(
        spec=openai_api.openai.OpenAI,
        audio=mock_audio,
        images=mock_images,
    )
    return mocker.patch("oai.openai_api.openai.OpenAI", return_value=mock_client)


class ConcreteOpenAiBaseClass(openai_api.OpenAIBaseClass):
    """A concrete implementation of the OpenAIBaseClass."""

    async def run(self, model: str, *_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
        """Runs the model."""
        return "mocked_response"


@pytest.mark.asyncio()
async def test_openai_base_class(mock_openai: pytest_mock.MockFixture) -> None:
    """Tests the OpenAIBaseClass."""
    concrete_openai_base_class = ConcreteOpenAiBaseClass()

    assert concrete_openai_base_class.client is not None
    assert mock_openai.call_count == 1
    assert await concrete_openai_base_class.run(model="") == "mocked_response"


@pytest.mark.asyncio()
async def test_text_to_speech(mock_openai: pytest_mock.MockFixture) -> None:
    """Tests the TextToSpeech class."""
    text_to_speech = openai_api.TextToSpeech()

    await text_to_speech.run("")

    assert text_to_speech.client is not None
    assert mock_openai.call_count == 1
    assert mock_openai.return_value.audio.speech.create.call_count == 1


@pytest.mark.asyncio()
async def test_speech_to_text(mock_openai: pytest_mock.MockFixture) -> None:
    """Tests the SpeechToText class."""
    speech_to_text = openai_api.SpeechToText()

    with tempfile.NamedTemporaryFile() as audio_file:
        await speech_to_text.run(audio_file.name)

    assert speech_to_text.client is not None
    assert mock_openai.call_count == 1
    assert mock_openai.return_value.audio.transcriptions.create.call_count == 1


@pytest.mark.asyncio()
async def test_image_generation(mock_openai: pytest_mock.MockFixture) -> None:
    """Tests the ImageGeneration class."""
    image_generation = openai_api.ImageGeneration()

    await image_generation.run("")

    assert image_generation.client is not None
    assert mock_openai.call_count == 1
    assert mock_openai.return_value.images.generate.call_count == 1
