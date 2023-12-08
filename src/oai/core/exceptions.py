"""Custom exceptions."""
from oai.core import config

logger = config.get_logger()


class LoggedException(BaseException):
    """Base class for exceptions that log their message."""

    def __init__(self, message: str) -> None:
        """Initializes a new instance of the LoggedException class.

        Args:
            message: The message to display.
        """
        logger.error(message)
        super().__init__(message)


class FileSizeError(LoggedException):
    """Raised when the file is too large."""
