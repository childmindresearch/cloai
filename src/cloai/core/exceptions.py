"""Custom exceptions."""
from cloai.core import config

logger = config.get_logger()


class LoggedException(BaseException):
    """Base class for exceptions that log their message."""

    def __init__(self, message: str) -> None:
        """Initializes a new instance of the LoggedException class.

        Args:
            message: The message to display.
        """
        logger.exception(message)
        super().__init__(message)


class FileSizeError(LoggedException):
    """Raised when the file is too large."""


class InvalidArgumentError(LoggedException):
    """Raised when the arguments are invalid."""


class LoggedValueError(LoggedException):
    """Raised when a value is invalid."""


class OpenAIError(LoggedException):
    """Raised when an OpenAI error occurs."""
