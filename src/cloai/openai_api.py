"""This module contains interactions with OpenAI models."""
import abc
import logging
import pathlib
from typing import Any, Literal, TypedDict, TypeVar

import instructor
import openai
import pydantic

from cloai.core import config, exceptions

settings = config.get_settings()
OPENAI_API_KEY = settings.OPENAI_API_KEY
LOGGER_NAME = settings.LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

T = TypeVar("T")


class Message(TypedDict):
    """A message object."""

    role: Literal["assistant", "system", "user"]
    content: str


class OpenAIBaseClass(pydantic.BaseModel, abc.ABC):
    """An abstract base class for OpenAI models.

    This class initializes the OpenAI client and requires a run method to be
    implemented.

    Attributes:
        api_key: The OpenAI API key.
        client: The OpenAI client used to interact with the model.
    """

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
    )

    api_key: pydantic.SecretStr | None = OPENAI_API_KEY
    client: openai.AsyncOpenAI = pydantic.Field(init=False, default=None)

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Initializes a new instance of the OpenAIBaseClass class.

        Args:
            api_key: The OpenAI API key.
        """
        if self.api_key is None:
            msg = "No OpenAI API key provided."
            raise exceptions.OpenAIError(msg)
        self.client = openai.AsyncOpenAI(api_key=self.api_key.get_secret_value())

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
        model: str = "gpt-4",
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


class ChatCompletionInstructor(OpenAIBaseClass):
    """A class for running the Chat Completion models using the instructor library.

    This class is intended to only be exposed to the user through the Python interface.
    """

    def __init__(self) -> None:
        """Patches the OpenAI library to use instructor."""
        super().__init__()
        self.client = instructor.patch(self.client)

    async def run(  # noqa: PLR0913
        self,
        user_prompt: str,
        system_prompt: str,
        response_model: T,
        model: str = "gpt-4",
        max_retries: int = 1,
    ) -> T:
        """Runs the Chat Completion model.

        Args:
            user_prompt: The user's prompt.
            system_prompt: The system's prompt.
            response_model: The response model to return.
            model: The name of the Chat Completion model to use.
            max_retries: The maximum number of retries to attempt.

        Returns:
            The model's response.
        """
        system_message = Message(role="system", content=system_prompt)
        user_message = Message(role="user", content=user_prompt)
        return await self.client.chat.completions.create(  # type: ignore[call-overload]
            model=model,
            messages=[system_message, user_message],  # type: ignore[list-item]
            response_model=response_model,
            max_retries=max_retries,
        )


class TextToSpeech(OpenAIBaseClass):
    """A class for running the Text-To-Speech models."""

    async def run(
        self,
        text: str,
        model: Literal["tts-1", "tts-1-hd"] = "tts-1",
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "onyx",
    ) -> bytes:
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
        return response.content


class SpeechToText(OpenAIBaseClass):
    """A class for running the Speech-To-Text models."""

    async def run(
        self,
        audio_file: pathlib.Path | str,
        model: str = "whisper-1",
        language: config.WhisperLanguages | str = config.WhisperLanguages.ENGLISH,
    ) -> str:
        """Runs the Speech-To-Text model.

        Args:
            audio_file: The audio to convert to text.
            model: The name of the Speech-To-Text model to use.
            language: The language of the audio. Can be both provided through the
                config.WhisperLanguages enum, which guarantees support, or as a string.

        Returns:
            The model's response.
        """
        if isinstance(language, config.WhisperLanguages):
            language = language.value

        return await self.client.audio.transcriptions.create(
            model=model,
            file=pathlib.Path(audio_file),
            response_format="text",
            language=language,
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


class Embedding(OpenAIBaseClass):
    """A class for running the Embedding models."""

    async def run(
        self,
        text: str,
        model: Literal[
            "text-embedding-3-small",
            "text-embedding-3-large",
        ] = "text-embedding-3-large",
        *,
        keep_new_lines: bool = False,
    ) -> list[float]:
        """Runs the Embedding model.

        Args:
            text: the string to embed.
            model: the name of the Embedding model to use.
            keep_new_lines: Whether to keep or remove line breaks,
            defaults to False.

        Returns:
            The embedding (list of numbers)
        """
        if keep_new_lines is False:
            text = text.replace("\n", " ")
        response = await self.client.embeddings.create(
            input=text,
            model=model,
        )
        return response.data[0].embedding
