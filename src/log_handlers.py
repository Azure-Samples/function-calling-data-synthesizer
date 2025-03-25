import logging
from logging.handlers import RotatingFileHandler


def get_handlers() -> tuple[RotatingFileHandler, logging.StreamHandler]:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        filename="app.log", maxBytes=10 * 1024 * 1024, backupCount=20, encoding="utf8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    return (file_handler, console_handler)
