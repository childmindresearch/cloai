"""Tools for logging."""

import logging


def get_logger(level: int | None = None) -> logging.Logger:
    """Gets the ctk-functions logger."""
    logger = logging.getLogger("cloai")
    if logger.hasHandlers():
        return logger

    if level is not None:
        logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s - %(message)s",  # noqa: E501
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
