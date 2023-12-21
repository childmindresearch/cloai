"""Tests for the parser module."""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import TYPE_CHECKING

import pytest

from cloai.cli import parser
from cloai.core import exceptions

if TYPE_CHECKING:
    import pytest_mock


@pytest.fixture(autouse=True)
def _set_environment() -> None:
    os.environ["OPENAI_API_KEY"] = "test"


@pytest.mark.parametrize(
    ("value", "will_raise"),
    [
        (1, False),
        (0, True),
        (-1, True),
        (1.0, False),
        (1.2, True),
    ],
)
def test__positive_int(value: float | int, will_raise: bool) -> None:  # noqa: PYI041
    """Tests that the input is a positive integer."""
    if will_raise:
        with pytest.raises(exceptions.InvalidArgumentError):
            parser._positive_int(value)  # type: ignore[arg-type]
    else:
        assert parser._positive_int(value) == int(value)  # type: ignore[arg-type]


def test__add_chat_completion_parser() -> None:
    """Tests the _add_chat_completion_parser function."""
    subparsers = argparse.ArgumentParser().add_subparsers()
    parser._add_chat_completion_parser(subparsers)
    expected_n_arguments = 7

    chat_completion_parser = subparsers.choices["gpt"]
    arguments = chat_completion_parser._actions

    assert "gpt" in subparsers.choices

    assert len(arguments) == expected_n_arguments


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

    assert arguments[1].dest == "prompt"
    assert arguments[1].type == str

    assert arguments[2].dest == "base_image_name"
    assert arguments[2].type == str

    assert arguments[3].dest == "model"
    assert arguments[3].help == (
        "The model to use. Consult OpenAI's documentation for an up-to-date list"
        " of models."
    )
    assert arguments[3].default == "dall-e-3"

    assert arguments[4].dest == "size"
    assert arguments[4].default == "1024x1024"

    assert arguments[5].dest == "quality"
    assert arguments[5].default == "standard"

    assert arguments[6].dest == "number"
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

    assert arguments[1].dest == "filename"
    assert (
        arguments[1].help
        == "The file to transcribe. Can be any format that ffmpeg supports."
    )
    assert arguments[1].type == pathlib.Path

    assert arguments[2].dest == "clip"
    assert isinstance(arguments[2], argparse._StoreTrueAction)  # type: ignore[arg-type]

    assert arguments[3].dest == "model"
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

    assert arguments[1].dest == "text"
    assert arguments[1].type == str

    assert arguments[2].dest == "output_file"
    assert arguments[2].type == pathlib.Path

    assert arguments[3].dest == "model"
    assert arguments[3].default == "tts-1"

    assert arguments[4].dest == "voice"
    assert arguments[4].default == "onyx"


@pytest.mark.asyncio()
async def test_run_command_without_arguments() -> None:
    """Tests the run_command function with no arguments."""
    args = argparse.Namespace()

    with pytest.raises(exceptions.InvalidArgumentError):
        await parser.run_command(args)


@pytest.mark.asyncio()
async def test_run_command_with_invalid_command() -> None:
    """Tests the run_command function with an invalid command."""
    args = argparse.Namespace(command="invalid")

    with pytest.raises(exceptions.InvalidArgumentError):
        await parser.run_command(args)


@pytest.mark.asyncio()
async def test_run_command_with_whisper(mocker: pytest_mock.MockFixture) -> None:
    """Tests the run_command function with the 'whisper' command."""
    arg_dict = {
        "command": "whisper",
        "filename": "test.wav",
        "clip": False,
        "model": "whisper-1",
    }
    args = argparse.Namespace(**arg_dict)
    mock = mocker.patch("cloai.cli.commands.speech_to_text")

    await parser.run_command(args)

    mock.assert_called_once_with(
        filename=arg_dict["filename"],
        clip=False,
        model=arg_dict["model"],
    )


@pytest.mark.asyncio()
async def test_run_command_with_dalle(mocker: pytest_mock.MockFixture) -> None:
    """Tests the run_command function with the 'dalle' command."""
    arg_dict = {
        "command": "dalle",
        "prompt": "test",
        "base_image_name": "test",
        "model": "dall-e-3",
        "size": "1024x1024",
        "quality": "standard",
        "number": 1,
    }
    args = argparse.Namespace(**arg_dict)
    mock = mocker.patch("cloai.cli.commands.image_generation")

    await parser.run_command(args)

    mock.assert_called_once_with(
        prompt=arg_dict["prompt"],
        output_base_name=arg_dict["base_image_name"],
        model=arg_dict["model"],
        size=arg_dict["size"],
        quality=arg_dict["quality"],
        n=arg_dict["number"],
    )


@pytest.mark.asyncio()
async def test_run_command_with_tts(mocker: pytest_mock.MockFixture) -> None:
    """Tests the run_command function with the 'tts' command."""
    arg_dict = {
        "command": "tts",
        "text": "test",
        "output_file": "test.wav",
        "model": "tts-1",
        "voice": "onyx",
    }
    args = argparse.Namespace(**arg_dict)
    mock = mocker.patch("cloai.cli.commands.text_to_speech")

    await parser.run_command(args)

    mock.assert_called_once_with(
        text=arg_dict["text"],
        output_file=arg_dict["output_file"],
        model=arg_dict["model"],
        voice=arg_dict["voice"],
    )


@pytest.mark.asyncio()
async def test_parse_args_without_arguments() -> None:
    """Tests the parse_args function with no arguments."""
    sys.argv = ["cloai"]
    expected_error_code = 1

    with pytest.raises(SystemExit) as excinfo:
        await parser.parse_args()
    assert excinfo.value.code == expected_error_code


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "command",
    [
        "whisper",
        "dalle",
        "tts",
    ],
)
async def test_parse_args_with_command_no_other_arguments(
    command: str,
) -> None:
    """Tests the parse_args function with a command but no other arguments."""
    sys.argv = ["cloai", command]
    expected_error_code = 2

    with pytest.raises(SystemExit) as excinfo:
        await parser.parse_args()

    assert excinfo.value.code == expected_error_code


@pytest.mark.asyncio()
async def test_parse_args_from_cli_with_dalle_all_arguments(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Tests the parse_args function with the 'dalle' command and all arguments."""
    command = mocker.patch("cloai.cli.commands.image_generation")
    sys.argv = [
        "cloai",
        "dalle",
        "test",
        "test",
        "--model",
        "dall-e-3",
        "--size",
        "1024x1024",
        "--quality",
        "standard",
        "-n",
        "1",
    ]

    await parser.parse_args()

    command.assert_called_once_with(
        prompt="test",
        output_base_name="test",
        model="dall-e-3",
        size="1024x1024",
        quality="standard",
        n=1,
    )


@pytest.mark.parametrize(
    "size",
    [
        "256x256",
        "512x512",
    ],
)
def test__arg_validation_invalid_dalle_size(size: str) -> None:
    """Tests the _arg_validation function with an invalid dalle size."""
    args = argparse.Namespace(
        command="dalle",
        model="dall-e-3",
        size=size,
    )

    with pytest.raises(exceptions.InvalidArgumentError):
        parser._arg_validation(args)
