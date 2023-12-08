"""Command line interface for the OpenAI API."""
import argparse
import pathlib
import sys

from oai.cli import commands
from oai.core import config

logger = config.get_logger()


PARSER_DEFAULTS = {
    "epilog": "Please report issues at https://github.com/cmi-dair/cli-oai.",
    "formatter_class": argparse.ArgumentDefaultsHelpFormatter,
}


async def parse_args() -> None:
    """Parse command line arguments and execute the corresponding command."""
    parser = argparse.ArgumentParser(
        prog="oai",
        description="CLI wrapper for OpenAI's API",
        **PARSER_DEFAULTS,
    )
    subparsers = parser.add_subparsers(dest="command")
    _add_stt_parser(subparsers)
    _add_tts_parser(subparsers)
    _add_image_generation_parser(subparsers)

    args = parser.parse_args()
    validated_args = _post_validation(args)
    if validated_args.command is None:
        parser.print_usage()
        sys.exit(1)

    result = await run_command(validated_args)

    if isinstance(result, str):
        sys.stdout.write(result)


async def run_command(args: argparse.ArgumentParser) -> str | bytes | None:
    """Executes the specified command based on the provided arguments.

    Args:
        args: The parsed command-line arguments.

    Returns:
        str, bytes, None: The result of the executed command.
    """
    if args.command == "whisper":
        result = await commands.speech_to_text(
            filename=args.filename,
            model=args.model,
            clip=args.clip,
        )
    elif args.command == "gpt":
        raise NotImplementedError
    elif args.command == "dalle":
        result = await commands.image_generation(
            prompt=args.prompt,
            output_base_name=args.base_image_name,
            model=args.model,
            width=args.width,
            height=args.height,
            quality=args.quality,
            n=args.n,
        )
    elif args.command == "tts":
        result = await commands.text_to_speech(
            text=args.text,
            model=args.model,
            voice=args.voice,
            output_file=args.output_file,
        )
    else:
        msg = f"Unknown command {args.command}."
        logger.error(msg)
        raise ValueError(msg)
    return result


def _add_stt_parser(
    subparsers: argparse._SubParsersAction,  # noqa: SLF001
) -> None:
    """Get the argument parser for the 'whisper' command.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers object to add the
            'whisper' command to.

    Returns:
        argparse.ArgumentParser: The argument parser for the 'whisper' command.
    """
    stt_parser = subparsers.add_parser(
        "whisper",
        description="Transcribes audio files with OpenAI's Whisper.",
        **PARSER_DEFAULTS,
    )
    stt_parser.add_argument(
        "filename",
        help="The file to transcribe. Can be any format that ffmpeg supports.",
        type=pathlib.Path,
    )
    stt_parser.add_argument(
        "--clip",
        help="Clip the file if it is too large.",
        action="store_true",
    )
    stt_parser.add_argument(
        "--model",
        help=(
            "The transcription model to use. Consult OpenAI's documentation for an"
            " up-to-date list of models."
        ),
        default="whisper-1",
    )


def _add_tts_parser(
    subparsers: argparse._SubParsersAction,  # noqa: SLF001
) -> None:
    """Get the argument parser for the 'tts' command.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers object to add the
            'tts' command to.

    Returns:
        argparse.ArgumentParser: The argument parser for the 'tts' command.
    """
    tts_parser = subparsers.add_parser(
        "tts",
        description="Generates audio files with OpenAI's Jukebox.",
        **PARSER_DEFAULTS,
    )
    tts_parser.add_argument(
        "text",
        help="The text to generate audio from.",
        type=str,
    )
    tts_parser.add_argument(
        "output_file",
        help="The name of the output file.",
        type=pathlib.Path,
    )
    tts_parser.add_argument(
        "--model",
        help=(
            "The model to use. Consult OpenAI's documentation for an up-to-date list"
            " of models."
        ),
        type=lambda x: x.lower(),
        default="tts-1",
    )
    tts_parser.add_argument(
        "--voice",
        help="The voice to use.",
        type=lambda x: x.lower(),
        default="onyx",
    )


def _add_image_generation_parser(
    subparsers: argparse._SubParsersAction,  # noqa: SLF001
) -> None:
    image_generation_parser = subparsers.add_parser(
        "dalle",
        description="Generates images with OpenAI's DALL-E.",
        **PARSER_DEFAULTS,
    )
    image_generation_parser.add_argument(
        "prompt",
        help="The prompt to generate images from.",
        type=str,
    )
    image_generation_parser.add_argument(
        "base_image_name",
        help="The base name for output images.",
        type=str,
    )
    image_generation_parser.add_argument(
        "--model",
        help=(
            "The model to use. Consult OpenAI's documentation for an up-to-date list"
            " of models."
        ),
        type=lambda x: x.lower(),
        default="dall-e-3",
    )
    image_generation_parser.add_argument(
        "--width",
        help="The width of the generated image.",
        type=_positive_int_or_none,
        default=None,
    )
    image_generation_parser.add_argument(
        "--height",
        help="The height of the generated image.",
        type=_positive_int_or_none,
        default=None,
    )
    image_generation_parser.add_argument(
        "--quality",
        help="The quality of the generated image.",
        type=lambda x: x.lower(),
        default="standard",
    )
    image_generation_parser.add_argument(
        "--n",
        help="The number of images to generate.",
        type=_positive_int,
        default=1,
    )


def _post_validation(args: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Validate the parsed arguments.

    Validation across arguments is not possible with the built-in argparse
    validation. This function performs validation across arguments.

    Args:
        args: The parsed command-line arguments.

    Returns:
        argparse.ArgumentParser: The validated arguments.
    """
    if args.command == "dalle":
        if args.width is None and args.height is None:
            args.width = 1024
            args.height = 1024
        elif args.width is None:
            args.width = args.height
        elif args.height is None:
            args.height = args.width
    return args


def _positive_int(value: int) -> int:
    if int(value) <= 0:
        msg = f"{value} is not a positive integer."
        raise argparse.ArgumentTypeError(msg)
    return int(value)


def _positive_int_or_none(value: int) -> int | None:
    """Converts the input to a positive integer or None.

    Args:
        value: The input to convert.
    """
    if value is None:
        return None
    return _positive_int(value)
