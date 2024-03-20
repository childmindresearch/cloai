"""Unit tests for the OpenAI API module."""

import tempfile
from typing import Any
from unittest import mock

import pytest

from cloai import openai_api


class ConcreteOpenAiBaseClass(openai_api.OpenAIBaseClass):
    """A concrete implementation of the OpenAIBaseClass."""

    async def run(self, model: str, *_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
        """Runs the model."""
        return "mocked_response"


@pytest.mark.asyncio()
async def test_openai_base_class(mock_openai: mock.AsyncMock) -> None:
    """Tests the OpenAIBaseClass."""
    concrete_openai_base_class = ConcreteOpenAiBaseClass()

    assert concrete_openai_base_class.client is not None
    assert mock_openai.call_count == 1
    assert await concrete_openai_base_class.run(model="") == "mocked_response"


@pytest.mark.asyncio()
async def test_text_to_speech(mock_openai: mock.AsyncMock) -> None:
    """Tests the TextToSpeech class."""
    text_to_speech = openai_api.TextToSpeech()

    await text_to_speech.run("")

    assert text_to_speech.client is not None
    assert mock_openai.call_count == 1
    assert mock_openai.return_value.audio.speech.create.call_count == 1


@pytest.mark.asyncio()
async def test_speech_to_text(mock_openai: mock.AsyncMock) -> None:
    """Tests the SpeechToText class."""
    speech_to_text = openai_api.SpeechToText()

    with tempfile.NamedTemporaryFile() as audio_file:
        await speech_to_text.run(audio_file.name)

    assert speech_to_text.client is not None
    assert mock_openai.call_count == 1
    assert mock_openai.return_value.audio.transcriptions.create.call_count == 1


@pytest.mark.asyncio()
async def test_image_generation(mock_openai: mock.AsyncMock) -> None:
    """Tests the ImageGeneration class."""
    image_generation = openai_api.ImageGeneration()

    await image_generation.run("")

    assert image_generation.client is not None
    assert mock_openai.call_count == 1
    assert mock_openai.return_value.images.generate.call_count == 1


@pytest.mark.asyncio()
async def test_chat_completion(mock_openai: mock.AsyncMock) -> None:
    """Tests the ChatCompletion class."""
    chat_completion = openai_api.ChatCompletion()

    await chat_completion.run("", "")

    assert chat_completion.client is not None
    assert mock_openai.call_count == 1
    assert mock_openai.return_value.chat.completions.create.call_count == 1


@pytest.mark.asyncio()
async def test_embedding(mock_openai: mock.AsyncMock) -> None:
    """Tests the Embedding class."""
    get_embedding = openai_api.Embedding()

    await get_embedding.run(
        "",
        model="text-embedding-3-large",
    )

    assert get_embedding.client is not None
    assert mock_openai.call_count == 1
    assert mock_openai.return_value.embeddings.create.call_count == 1
