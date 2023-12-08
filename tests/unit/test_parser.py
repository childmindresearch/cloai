"""Tests for the parser module."""
import argparse

import pytest

from oai.cli import parser


@pytest.mark.parametrize(
    ("value", "will_raise"),
    [
        (1, False),
        (0, True),
        (-1, True),
        (1.0, False),
    ],
)
def test__positive_int(value: float, will_raise: bool) -> None:
    """Tests that the input is a positive integer."""
    if will_raise:
        with pytest.raises(argparse.ArgumentTypeError):
            parser._positive_int(value)
    else:
        assert parser._positive_int(value) == int(value)
