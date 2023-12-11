"""Tests for the parser module."""

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
