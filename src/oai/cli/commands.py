"""Contains the core business logic of the OpenAI CLI."""
from __future__ import annotations

import asyncio
import logging
import pathlib
import tempfile
from typing import Literal

import ffmpeg

from oai import openai_api
from oai.cli import utils
from oai.core import config

settings = config.get_settings()
logger = logging.getLogger(settings.LOGGER_NAME)
MAX_FILE_SIZE = 24_500_000  # Max size is 25MB, but we leave some room for error.


async def speech_to_text(
    filename: pathlib.Path,
    model: str,
    *,
    clip: bool = False,
) -> str:
    """Transcribes audio files with OpenAI's TTS models.

    Args:
        filename: The file to transcribe. Can be any format that ffmpeg supports.
        model: The transcription model to use.
        voice: The voice to use.
        clip: Whether to clip the file if it is too large, defaults to False.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = pathlib.Path(temp_dir) / "temp.mp3"
        ffmpeg.input(filename).output(str(temp_file)).overwrite_output().run()

        if clip:
            files = list(utils.clip_audio(temp_file, temp_dir, MAX_FILE_SIZE))
        else:
            files = [temp_file]

        stt = openai_api.SpeechToText()
        transcription_promises = [stt.run(filename, model=model) for filename in files]
        transcriptions = await asyncio.gather(*transcription_promises)

        return " ".join(transcriptions)


async def text_to_speech(
    text: str,
    output_file: str,
    model: str,
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
) -> None:
    """Converts text to speech with OpenAI's TTS models.

    Args:
        text: The text to convert to speech.
        output_file: The name of the output file.
        model: The model to use.
        voice: The voice to use.
    """
    tts = openai_api.TextToSpeech()
    await tts.run(text, output_file, model=model, voice=voice)


async def image_generation(  # noqa: PLR0913
    prompt: str,
    output_base_name: str,
    model: str,
    size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] | None,
    quality: Literal["standard", "hd"],
    n: int,
) -> None:
    """Generates an image from text with OpenAI's Image Generation models.

    Args:
        prompt: The text to generate an image from.
        output_base_name: The base name of the output file.
        model: The model to use.
        size: The size of the generated image. Defaults to None.
        quality: The quality of the generated image. Defaults to "standard".
        n: The number of images to generate. Defaults to 1.

    Returns:
        bytes: The generated image as bytes.
    """
    image_generation = openai_api.ImageGeneration()
    urls = await image_generation.run(
        prompt,
        model=model,
        size=size,
        quality=quality,
        n=n,
    )

    for index, url in enumerate(urls):
        if url is None:
            logger.warning("Image %s failed to generate, skipping.", index)
            continue
        file = pathlib.Path(f"{output_base_name}_{index}.png")
        utils.download_file(file, url)
