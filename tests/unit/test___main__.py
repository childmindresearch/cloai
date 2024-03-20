"""Tests for the entrypoint."""

import os

import pytest

from cloai import __main__ as main


def test_main_no_key() -> None:
    """Test case for the main function when OPENAI_API_KEY is not set.

    It verifies that the main function raises a SystemExit exception with exit code 1.
    """
    del os.environ["OPENAI_API_KEY"]

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 1
