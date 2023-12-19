"""Entry point of the cloai package."""
import asyncio
import os
import sys


def main() -> None:
    """Entry point for the CLI."""
    if "OPENAI_API_KEY" not in os.environ:
        sys.stderr.write("Error: Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    from cloai.cli import parser

    asyncio.run(parser.parse_args())


if __name__ == "__main__":
    main()
