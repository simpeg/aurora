__version__ = "0.6.0"

import sys
from loguru import logger


# =============================================================================
# Initiate loggers
# =============================================================================
config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "level": "INFO",
            "colorize": True,
            "format": "<level>{time} | {level: <3} | {name} | {function} | line: {line} | {message}</level>",
        },
    ],
    "extra": {"user": "someone"},
}
logger.configure(**config)
# logger.disable("mt_metadata")
