"""Tests for the parser module."""
import argparse
import pathlib

import pytest

from oai.cli import parser
from oai.core import exceptions


@pytest.mark.parametrize(
    ("value", "will_raise"),
    [
        (1, False),
        (0, True),
        (-1, True),
        (1.0, False),
    ],
)
def test__positive_int(value: float | int, will_raise: bool) -> None:  # noqa: PYI041
    """Tests that the input is a positive integer."""
    if will_raise:
        with pytest.raises(exceptions.InvalidArgumentError):
            parser._positive_int(value)  # type: ignore[arg-type]
    else:
        assert parser._positive_int(value) == int(value)  # type: ignore[arg-type]


def test__add_image_generation_parser() -> None:
    """Tests the _add_image_generation_parser function."""
    subparsers = argparse.ArgumentParser().add_subparsers()
    parser._add_image_generation_parser(subparsers)
    expected_n_arguments = 7

    image_generation_parser = subparsers.choices["dalle"]
    arguments = image_generation_parser._actions

    assert "dalle" in subparsers.choices
    assert (
        image_generation_parser.description == "Generates images with OpenAI's DALL-E."
    )

    assert len(arguments) == expected_n_arguments

    assert arguments[0].dest == "help"
    assert arguments[0].help == "show this help message and exit"

    assert arguments[1].dest == "prompt"
    assert arguments[1].help == "The prompt to generate images from."
    assert arguments[1].type == str

    assert arguments[2].dest == "base_image_name"
    assert arguments[2].help == "The base name for output images."
    assert arguments[2].type == str

    assert arguments[3].dest == "model"
    assert arguments[3].help == (
        "The model to use. Consult OpenAI's documentation for an up-to-date list"
        " of models."
    )
    assert arguments[3].type("dall-e-3") == "dall-e-3"
    assert arguments[3].default == "dall-e-3"

    assert arguments[4].dest == "size"
    assert arguments[4].help == "The size of the generated image."
    assert arguments[4].default == "1024x1024"

    assert arguments[5].dest == "quality"
    assert arguments[5].help == "The quality of the generated image."
    assert arguments[5].type("standard") == "standard"
    assert arguments[5].default == "standard"

    assert arguments[6].dest == "n"
    assert arguments[6].help == "The number of images to generate."
    assert arguments[6].type(1) == 1
    assert arguments[6].default == 1


def test__add_stt_parser() -> None:
    """Tests the _add_stt_parser function."""
    subparsers = argparse.ArgumentParser().add_subparsers()
    parser._add_stt_parser(subparsers)
    expected_n_arguments = 4

    stt_parser = subparsers.choices["whisper"]
    arguments = stt_parser._actions

    assert "whisper" in subparsers.choices
    assert stt_parser.description == "Transcribes audio files with OpenAI's STT models."

    assert len(arguments) == expected_n_arguments

    assert arguments[0].dest == "help"
    assert arguments[0].help == "show this help message and exit"

    assert arguments[1].dest == "filename"
    assert (
        arguments[1].help
        == "The file to transcribe. Can be any format that ffmpeg supports."
    )
    assert arguments[1].type == pathlib.Path

    assert arguments[2].dest == "clip"
    assert arguments[2].help == "Clip the file if it is too large."
    assert isinstance(arguments[2], argparse._StoreTrueAction)  # type: ignore[arg-type]

    assert arguments[3].dest == "model"
    assert arguments[3].help == "The transcription model to use."
    assert arguments[3].default == "whisper-1"


def test__add_tts_parser() -> None:
    """Tests the _add_tts_parser function."""
    subparsers = argparse.ArgumentParser().add_subparsers()
    parser._add_tts_parser(subparsers)
    expected_n_arguments = 5

    tts_parser = subparsers.choices["tts"]
    arguments = tts_parser._actions

    assert "tts" in subparsers.choices
    assert tts_parser.description == "Generates audio files with OpenAI's Jukebox."

    assert len(arguments) == expected_n_arguments

    assert arguments[0].dest == "help"
    assert arguments[0].help == "show this help message and exit"

    assert arguments[1].dest == "text"
    assert arguments[1].help == "The text to generate audio from."
    assert arguments[1].type == str

    assert arguments[2].dest == "output_file"
    assert arguments[2].help == "The name of the output file."
    assert arguments[2].type == pathlib.Path

    assert arguments[3].dest == "model"
    assert arguments[3].help == "The model to use."
    assert arguments[3].default == "tts-1"

    assert arguments[4].dest == "voice"
    assert arguments[4].help == "The voice to use."
    assert arguments[4].default == "onyx"
