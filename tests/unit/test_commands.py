"""Tests for the commands module."""
import pathlib
from unittest import mock

import pytest

from cloai.cli import commands
from cloai.core import exceptions


@pytest.fixture()
def user_prompt_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """Creates a mock file."""
    mock_file = tmp_path / "mock_file.txt"
    mock_file.write_text("User mock file content")
    return mock_file


@pytest.fixture()
def system_prompt_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """Creates a mock file."""
    mock_file = tmp_path / "mock_file.txt"
    mock_file.write_text("System mock file content")
    return mock_file


def test_chat_completion_with_str_arguments() -> None:
    """Tests the chat completion constructor with string arguments."""
    cc = commands.ChatCompletion(user_prompt="Hello", model="gpt-4", system_prompt="Hi")

    assert cc.user_prompt == "Hello"
    assert cc.system_prompt == "Hi"
    assert cc.model == "gpt-4"


def test_chat_completion_with_file_arguments(
    user_prompt_file: pathlib.Path,
    system_prompt_file: pathlib.Path,
) -> None:
    """Tests the chat completion constructor with file arguments."""
    user_expected = user_prompt_file.open().read()
    system_expected = system_prompt_file.open().read()

    cc = commands.ChatCompletion(
        user_prompt_file=user_prompt_file,
        system_prompt_file=system_prompt_file,
        model="gpt-4",
    )

    assert cc.user_prompt == user_expected
    assert cc.system_prompt == system_expected
    assert cc.model == "gpt-4"


def test__validate_initialization_with_correct_prompt() -> None:
    """Tests the _validate_initialization function with correct string prompts."""
    commands.ChatCompletion._validate_initialization(
        user_prompt="Hello",
        system_prompt="Hi",
    )

    # No exception should be raised


def test_validate_initialization_with_two_user_prompts() -> None:
    """Tests the _validate_initialization function with two user prompts."""
    with pytest.raises(exceptions.LoggedValueError):
        commands.ChatCompletion._validate_initialization(
            user_prompt="Hello",
            user_prompt_file=pathlib.Path("mock_file.txt"),
            system_prompt="Hello",
        )


def test_validate_initialization_with_two_system_prompts() -> None:
    """Tests the _validate_initialization function with two system prompts."""
    with pytest.raises(exceptions.LoggedValueError):
        commands.ChatCompletion._validate_initialization(
            user_prompt="Hello",
            system_prompt="Hello",
            system_prompt_file=pathlib.Path("mock_file.txt"),
        )


def test_determine_system_prompt_with_str() -> None:
    """Tests the _detect_prompt_type function with a string."""
    cc = commands.ChatCompletion(
        "gpt-4",
        user_prompt="user",
        system_prompt="system",
    )

    prompt = cc._determine_system_prompt(
        system_prompt="Hello",
    )

    assert prompt == "Hello"


def test_determine_system_prompt_with_file(system_prompt_file: pathlib.Path) -> None:
    """Tests the _determine_system_prompt function with a file."""
    cc = commands.ChatCompletion(
        "gpt-4",
        user_prompt="user",
        system_prompt="system",
    )

    prompt = cc._determine_system_prompt(
        system_prompt_file=system_prompt_file,
    )

    assert prompt == system_prompt_file.open().read()


def test_determine_system_prompt_with_preset() -> None:
    """Tests the _determine_system_prompt function with a preset."""
    cc = commands.ChatCompletion(
        "gpt-4",
        user_prompt="user",
        system_prompt="system",
    )

    prompt = cc._determine_system_prompt(
        system_preset="summary",
    )

    assert isinstance(prompt, str)
    assert len(prompt) > 0


@pytest.mark.asyncio()
async def test_chat_completion_run_method(mock_openai: mock.MagicMock) -> None:
    """Tests the run method."""
    cc = commands.ChatCompletion(
        user_prompt="user",
        system_prompt="system",
        model="gpt-4",
    )

    response = await cc.run()

    assert (
        response
        == mock_openai.return_value.chat.completions.create.return_value.choices[
            0
        ].message.content
    )
