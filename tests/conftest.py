"""Test configurations."""
import os


def pytest_configure() -> None:
    """Configure pytest with the necessary environment variables.

    Args:
        config: The pytest configuration object.

    """
    os.environ["OPENAI_API_KEY"] = "API_KEY"
