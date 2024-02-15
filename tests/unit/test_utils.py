"""Tests for the CLI utility functions."""
import pathlib
import tempfile

import aiocsv
import aiofiles
import aioresponses
import pytest
import pytest_mock

from cloai.core import utils


def test_clip_audio(mocker: pytest_mock.MockerFixture) -> None:
    """Tests that the audio file is clipped."""
    magic = mocker.MagicMock()
    mock_ffmpeg = mocker.patch("cloai.core.utils.ffmpeg", magic)
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


@pytest.mark.asyncio()
async def test_download_file(tmp_path: pathlib.Path) -> None:
    """Tests that the file is downloaded successfully."""
    test_url = "http://example.com/testfile"
    test_file_contents = b"This is a test file"
    test_file_path = tmp_path / "downloaded_file"
    with aioresponses.aioresponses() as mock_response:
        mock_response.get(test_url, status=200, body=test_file_contents)

        await utils.download_file(test_file_path, test_url)

        assert test_file_path.read_bytes() == test_file_contents


@pytest.mark.asyncio()
async def test_save_csv(tmp_path: pathlib.Path) -> None:
    """Tests that the content is saved to a csv file asynchronously."""
    test_filename = tmp_path / "test.csv"
    test_content = [1.0, 2.0, 3.0, 4.0, 5.0]

    await utils.save_csv(test_filename, test_content)

    async with aiofiles.open(test_filename, "r") as file:
        async for row in aiocsv.AsyncReader(file):
            content = row

    assert content == ["1", "2", "3", "4", "5"]
