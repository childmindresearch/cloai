"""Tests for the prompts module."""

from collections.abc import Callable
from typing import Any

import pytest

from cloai.llm import prompts


def test_remove_consecutive_whitespace() -> None:
    """Tests removing consecutive whitespace."""
    text = "  Hello,       \n\n\n   world!   "
    expected = "Hello, world!"

    actual = prompts._remove_consecutive_whitespace(text)

    assert actual == expected


def test_substitute_success() -> None:
    """Tests successful substitution."""
    text = "Hello ${var}!"
    expected = "Hello world!"

    actual = prompts._substitute(text, var="world")

    assert actual == expected


def test_substitute_missing() -> None:
    """Tests missing substitution."""
    text = "Hello ${var}!"

    with pytest.raises(KeyError):
        prompts._substitute(text)


def test_substitute_excess_args() -> None:
    """Tests missing substitution."""
    text = "Hello ${var}!"

    with pytest.raises(ValueError, match="Too many keys provided: {'bard'}."):
        prompts._substitute(text, var="world", bard="hi")


@pytest.mark.parametrize(
    ("fun", "args"),
    [
        (prompts.chain_of_density, {"article": "test"}),
        (prompts.chain_of_verification_create_statements, {}),
        (
            prompts.chain_of_verification_verify,
            {"statements": ["a", "b"]},
        ),
        (
            prompts.chain_of_verification_rewrite,
            {"statements": ["a", "b"], "instructions": "a", "source": "b"},
        ),
    ],
)
def test_prompts(fun: Callable, args: dict[str, Any]) -> None:
    """Tests that the prompts aren't missing templates."""
    try:
        fun(**args)
    except KeyError as exc:
        pytest.fail(f"Missing substitution. KeyError: {exc}")
