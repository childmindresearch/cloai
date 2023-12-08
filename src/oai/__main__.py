"""Entry point of the OAI package."""
import asyncio

from oai.cli import parser


def main() -> None:
    """Entry point for the CLI."""
    asyncio.run(parser.parse_args())


if __name__ == "__main__":
    main()
