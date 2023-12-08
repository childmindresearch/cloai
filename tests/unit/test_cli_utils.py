"""Tests for the CLI utility functions."""


import pytest_mock

from oai.cli import utils


def test_clip_audio(mocker: pytest_mock.MockerFixture) -> None:
    """Tests that the audio file is clipped."""
    magic = mocker.MagicMock()
    mock_ffmpeg = mocker.patch("oai.cli.utils.ffmpeg", magic)
    mock_ffmpeg.input.return_value = magic
    mock_ffmpeg.output.return_value = magic

    next(utils.clip_audio("mock_filename.wav"))

    mock_ffmpeg.input.assert_called_once_with("mock_filename.wav")
    mock_ffmpeg.output.assert_called_once_with(
        mocker.ANY,
        f="segment",
        segment_time=utils.CLIP_DURATION,
    )


def test_download_audio(mocker: pytest_mock.MockerFixture) -> None:
    """Tests that a file is downloaded."""
    magic = mocker.MagicMock()
    mock_requests = mocker.patch("oai.cli.utils.requests", magic)
    mock_requests.get.return_value = magic
    magic.raise_for_status.return_value = None
    magic.content = "mock_content"

    utils.download_file("mock_filename.wav", "mock_url")

    mock_requests.get.assert_called_once_with("mock_url", timeout=10)
    magic.raise_for_status.assert_called_once_with()
    assert magic.content == "mock_content"
