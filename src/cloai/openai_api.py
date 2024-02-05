"""This module contains interactions with OpenAI models."""
from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import aiofiles
import openai

from cloai.core import config, exceptions

if TYPE_CHECKING:
    import pathlib

settings = config.get_settings()
OPENAI_API_KEY = settings.OPENAI_API_KEY
LOGGER_NAME = settings.LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class Message(TypedDict):
    """A message object."""

    role: Literal["assistant", "system", "user"]
    content: str


class OpenAIBaseClass(abc.ABC):
    """An abstract base class for OpenAI models.

    This class initializes the OpenAI client and requires a run method to be
    implemented.

    Attributes:
        client: The OpenAI client used to interact with the model.
    """

    def __init__(self) -> None:
        """Initializes a new instance of the OpenAIBaseClass class."""
        self.client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY.get_secret_value())

    @abc.abstractmethod
    async def run(self, *_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
        """Runs the model."""
        ...


class ChatCompletion(OpenAIBaseClass):
    """A class for running the Chat Completion models."""

    async def run(
        self,
        user_prompt: str,
        system_prompt: str,
        model: Literal["gpt-4", "gpt-3.5-turbo", "gpt-4-1106-preview"] = "gpt-4",
    ) -> str:
        """Runs the Chat Completion model.

        Args:
            user_prompt: The user's prompt.
            system_prompt: The system's prompt.
            model: The name of the Chat Completion model to use.

        Returns:
            The model's response.
        """
        system_message = Message(role="system", content=system_prompt)
        user_message = Message(role="user", content=user_prompt)
        response = await self.client.chat.completions.create(
            model=model,
            messages=[system_message, user_message],  # type: ignore[list-item]
        )
        if not response.choices[0].message.content:
            msg = "No response from OpenAI."
            raise exceptions.OpenAIError(msg)
        return response.choices[0].message.content


class TextToSpeech(OpenAIBaseClass):
    """A class for running the Text-To-Speech models."""

    async def run(
        self,
        text: str,
        output_file: pathlib.Path | str,
        model: str = "tts-1",
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "onyx",
    ) -> None:
        """Runs the Text-To-Speech model.

        Args:
            text: The text to convert to speech.
            output_file: The name of the output file.
            model: The name of the Text-To-Speech model to use.
            voice: The voice to use.

        Returns:
            The model's response.
        """
        response = await self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
        )
        response.stream_to_file(output_file)


class SpeechToText(OpenAIBaseClass):
    """A class for running the Speech-To-Text models."""

    async def run(
        self,
        audio_file: pathlib.Path | str,
        model: str = "whisper-1",
    ) -> str:
        """Runs the Speech-To-Text model.

        Args:
            audio_file: The audio to convert to text.
            model: The name of the Speech-To-Text model to use.

        Returns:
            The model's response.
        """
        async with aiofiles.open(audio_file, "rb") as audio:
            return await self.client.audio.transcriptions.create(
                model=model,
                file=audio,  # type: ignore[arg-type]
                response_format="text",
            )  # type: ignore[return-value] # response_format overrides output type.


class ImageGeneration(OpenAIBaseClass):
    """A class for running the Image Generation models."""

    async def run(  # noqa: PLR0913
        self,
        prompt: str,
        model: str = "dall-e-3",
        size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
        | None = None,
        quality: Literal["standard", "hd"] = "standard",
        n: int | None = None,
    ) -> list[str | None]:
        """Runs the Image Generation model.

        Args:
            prompt: The prompt to generate an image from.
            model: The name of the Image Generation model to use.
            size: The size of the generated image.
            quality: The quality of the generated image.
            n: The number of images to generate.

        Returns:
            str: The image urls.
        """
        response = await self.client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
        )

        return [data.url for data in response.data]
