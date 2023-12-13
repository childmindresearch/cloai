"""Configuration for the oai module."""
import functools
import logging
from typing import Literal

import pydantic
import pydantic_settings


class Settings(pydantic_settings.BaseSettings):
    """Represents the settings for the oai module."""

    LOGGER_NAME: str = "oai"
    LOGGER_VERBOSITY: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    OPENAI_API_KEY: pydantic.SecretStr = pydantic.Field(
        ...,
        json_schema_extra={
            "env": "OPENAI_API_KEY",
            "description": "The API key for OpenAI.",
        },
    )


@functools.lru_cache
def get_settings() -> Settings:
    """Cached fetcher for the API settings.

    Returns:
        The settings for the API.
    """
    return Settings()  # type: ignore[call-arg]


def initialize_logger() -> logging.Logger:
    """Initializes the logger for the API."""
    settings = get_settings()
    logger = logging.getLogger(settings.LOGGER_NAME)
    if settings.LOGGER_VERBOSITY is not None:
        logger.setLevel(settings.LOGGER_VERBOSITY)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s - %(message)s",  # noqa: E501
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    """Gets the logger for the API.

    Returns:
        The logger for the API.
    """
    settings = get_settings()
    logger = logging.getLogger(settings.LOGGER_NAME)
    if logger.hasHandlers():
        return logger
    return initialize_logger()
