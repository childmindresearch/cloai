"""Tests for the CLI utility functions."""


import tempfile

import pytest_mock

from oai.cli import utils


def test_clip_audio(mocker: pytest_mock.MockerFixture) -> None:
    """Tests that the audio file is clipped."""
    magic = mocker.MagicMock()
    mock_ffmpeg = mocker.patch("oai.cli.utils.ffmpeg", magic)
    mock_ffmpeg.input.return_value = magic
    mock_ffmpeg.output.return_value = magic
    target_size = 100000

    with tempfile.NamedTemporaryFile() as audio_file:
        list(utils.clip_audio(audio_file.name, "out_dir", target_size))

        mock_ffmpeg.input.assert_called_once_with(audio_file.name)
    mock_ffmpeg.output.assert_called_once_with(
        mocker.ANY,
        f="segment",
        segment_time=1,
    )


def test_download_audio(mocker: pytest_mock.MockerFixture) -> None:
    """Tests that a file is downloaded."""
    magic = mocker.MagicMock()
    mock_requests = mocker.patch("oai.cli.utils.requests")
    mock_requests.get.return_value = magic
    magic.raise_for_status.return_value = None
    magic.content = b"mock_content"

    with tempfile.NamedTemporaryFile() as audio_file:
        utils.download_file(audio_file.name, "mock_url")

        mock_requests.get.assert_called_once_with("mock_url", timeout=10)

    mock_requests.get.assert_called_once_with("mock_url", timeout=10)
    magic.raise_for_status.assert_called_once_with()
    assert magic.content == b"mock_content"
