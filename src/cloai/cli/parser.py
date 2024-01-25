"""Command line interface for the OpenAI API."""
from __future__ import annotations

import argparse
import pathlib
import sys

from cloai.cli import commands
from cloai.core import config, exceptions

logger = config.get_logger()


PARSER_DEFAULTS = {
    "epilog": "Please report issues at https://github.com/childmindresearch/cloai.",
    "formatter_class": argparse.ArgumentDefaultsHelpFormatter,
}


async def parse_args() -> None:
    """Parse command line arguments and execute the corresponding command."""
    parser = create_parser()
    validated_args = get_args(parser)
    result = await run_command(validated_args)
    if isinstance(result, str):
        sys.stdout.write(result)


def get_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse the command line arguments using the provided ArgumentParser object.

    Args:
        parser: The ArgumentParser object to parse the arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.

    Raises:
        SystemExit: If the command is not specified.
    """
    args = parser.parse_args()
    if args.command is None:
        parser.print_usage()
        sys.exit(1)
    return _arg_validation(args)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the CLI.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="cloai",
        description="""
        CLI wrapper for OpenAI's API. All commands require the OPENAI_API_KEY
        environment variable to be set to a valid OpenAI API key.""",
        **PARSER_DEFAULTS,  # type: ignore[arg-type]
    )

    version = config.get_version()
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {version}",
    )

    subparsers = parser.add_subparsers(dest="command")
    _add_chat_completion_parser(subparsers)
    _add_stt_parser(subparsers)
    _add_tts_parser(subparsers)
    _add_image_generation_parser(subparsers)
    return parser


async def run_command(args: argparse.Namespace) -> str | bytes | None:
    """Executes the specified command based on the provided arguments.

    Args:
        args: The parsed command-line arguments.

    Returns:
        str, bytes, None: The result of the executed command.
    """
    if getattr(args, "command", None) is None:
        msg = "No command provided."
        raise exceptions.InvalidArgumentError(msg)

    if args.command == "dalle":
        await commands.image_generation(
            prompt=args.prompt,
            output_base_name=args.base_image_name,
            model=args.model,
            size=args.size,
            quality=args.quality,
            n=args.number,
        )
        return None
    if args.command == "gpt":
        return await commands.ChatCompletion(
            user_prompt=args.user_prompt,
            user_prompt_file=args.user_prompt_file,
            system_prompt=args.system_prompt,
            system_prompt_file=args.system_prompt_file,
            system_preset=args.system_preset,
            model=args.model,
        ).run()
    if args.command == "tts":
        await commands.text_to_speech(
            text=args.text,
            model=args.model,
            voice=args.voice,
            output_file=args.output_file,
        )
        return None
    if args.command == "whisper":
        return await commands.speech_to_text(
            filename=args.filename,
            model=args.model,
            clip=args.clip,
        )
    msg = f"Unknown command {args.command}."
    raise exceptions.InvalidArgumentError(msg)


def _add_chat_completion_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Get the argument parser for the 'gpt' command.

    Args:
        subparsers: The subparsers object to add the 'gpt' command to.

    Returns:
        argparse.ArgumentParser: The argument parser for the 'gpt' command.
    """
    chat_parser = subparsers.add_parser(
        "gpt",
        description="Completes text with OpenAI's GPT models.",
        help="Completes text with OpenAI's GPT models.",
        **PARSER_DEFAULTS,  # type: ignore[arg-type]
    )

    user_group = chat_parser.add_argument_group(
        "User Prompts",
        """The prompts povided by the user. One must be provided and these
        arguments are mutually exclusive.""",
    )
    user_group_exclusive = user_group.add_mutually_exclusive_group(required=True)
    user_group_exclusive.add_argument(
        "-p",
        "--user-prompt",
        help="The user prompt to complete.",
        type=str,
    )
    user_group_exclusive.add_argument(
        "-f",
        "--user-prompt-file",
        help="The file containing the prompt to complete.",
        type=pathlib.Path,
    )

    system_group = chat_parser.add_argument_group(
        "System Prompts",
        """The prompts povided to the system. One must be provided and these
        arguments are mutually exclusive.""",
    )
    system_group_exclusive = system_group.add_mutually_exclusive_group(required=True)
    system_group_exclusive.add_argument(
        "-s",
        "--system-prompt",
        help="The system prompt.",
    )
    system_group_exclusive.add_argument(
        "-t",
        "--system-prompt-file",
        help="The file containing the system prompt.",
        type=pathlib.Path,
    )
    system_group_exclusive.add_argument(
        "-r",
        "--system-preset",
        help="""The preset system prompt. The following prompts are available:
        1) "summary": Summarizes the user prompt; 2) "transcript": Edits a
         audio transcript for coherence; 3) "simplify": Simplifies the reading
         level of the user prompt.""",
        choices=("summary", "transcript", "simplify"),
    )

    chat_parser.add_argument(
        "-m",
        "--model",
        help="The model to use.",
        choices=("gpt-4", "gpt-3.5-turbo", "gpt-4-1106-preview"),
        default="gpt-4",
    )


def _add_stt_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Get the argument parser for the "whisper" command.

    Args:
        subparsers: The subparsers object to add the
            "whisper" command to.

    Returns:
        argparse.ArgumentParser: The argument parser for the "whisper" command.
    """
    stt_parser = subparsers.add_parser(
        "whisper",
        description="Transcribes audio files with OpenAI's STT models.",
        help="Transcribes audio files with OpenAI's STT models.",
        **PARSER_DEFAULTS,  # type: ignore[arg-type]
    )
    stt_parser.add_argument(
        "filename",
        help="The file to transcribe. Can be any format that ffmpeg supports.",
        type=pathlib.Path,
    )
    stt_parser.add_argument(
        "-c",
        "--clip",
        help="Clip the file if it is too large.",
        action="store_true",
    )
    stt_parser.add_argument(
        "-m",
        "--model",
        help=("The transcription model to use."),
        type=lambda x: x.lower(),
        choices=["whisper-1"],
        default="whisper-1",
    )


def _add_tts_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Get the argument parser for the "tts" command.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers object to add the
            "tts" command to.

    Returns:
        argparse.ArgumentParser: The argument parser for the "tts" command.
    """
    tts_parser = subparsers.add_parser(
        "tts",
        description="Generates audio files with OpenAI's Jukebox.",
        help="Generates audio files with OpenAI's TTS models.",
        **PARSER_DEFAULTS,  # type: ignore[arg-type]
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
        "-m",
        "--model",
        help=("The model to use."),
        choices=["tts-1"],
        default="tts-1",
    )
    tts_parser.add_argument(
        "-v",
        "--voice",
        help="The voice to use.",
        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        default="onyx",
    )


def _add_image_generation_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    image_generation_parser = subparsers.add_parser(
        "dalle",
        description="Generates images with OpenAI's DALL-E.",
        help="Generates images with OpenAI's image generation models.",
        **PARSER_DEFAULTS,  # type: ignore[arg-type]
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
        "-m",
        "--model",
        help=(
            "The model to use. Consult OpenAI's documentation for an up-to-date list"
            " of models."
        ),
        type=lambda x: x.lower(),
        default="dall-e-3",
    )
    image_generation_parser.add_argument(
        "-s",
        "--size",
        help="The size of the generated image.",
        type=lambda x: x.lower(),
        choices=["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
        default="1024x1024",
    )
    image_generation_parser.add_argument(
        "-q",
        "--quality",
        help="The quality of the generated image.",
        type=lambda x: x.lower(),
        choices=["standard", "hd"],
        default="standard",
    )
    image_generation_parser.add_argument(
        "-n",
        "--number",
        help="The number of images to generate.",
        type=_positive_int,
        default=1,
    )


def _arg_validation(args: argparse.Namespace) -> argparse.Namespace:
    """Validate the parsed arguments.

    Validation across arguments is not possible with the built-in argparse
    validation. This function performs validation across arguments.

    Args:
        args: The parsed command-line arguments.

    Returns:
        argparse.ArgumentParser: The validated arguments.
    """
    if args.command == "dalle":  # noqa: SIM102 # Allows logical expansion across commands.
        if args.model == "dall-e-3" and args.size in ["256x256", "512x512"]:
            msg = "The dall-e-3 model does not support 256x256 or 512x512 images."
            raise exceptions.InvalidArgumentError(msg)

    return args


def _positive_int(value: int | str) -> int:
    """Ensures the value is a positive integer.

    Args:
        value: The value.

    Returns:
        int: The positive integer.

    Raises:
        exceptions.InvalidArgumentError: If the value is not an integer or not a
        positive integer.
    """
    if int(value) != float(value):
        msg = f"{value} is not an integer."
        raise exceptions.InvalidArgumentError(msg)
    if int(value) <= 0:
        msg = f"{value} is not a positive integer."
        raise exceptions.InvalidArgumentError(msg)
    return int(value)
