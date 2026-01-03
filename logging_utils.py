import logging
import sys

def setup_logger(debug: bool) -> logging.Logger:
    logger = logging.getLogger("dk")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if debug else logging.INFO)

    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger
