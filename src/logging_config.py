"""
logging_config.py

Central logging setup for the whole project. Every module gets its logger via
`get_logger(__name__)`, which produces consistent, timestamped output like:

    2026-07-31 14:22:01 | INFO | src.models.predict | Model loaded successfully.

Using logging instead of print() lets us see WHEN something happened, from
WHICH module, and at what severity — and lets us silence or redirect output
without touching the code that emits it.
"""

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configure the root logger only once (idempotent — safe to import many times).
_configured = False


def _configure_root(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    _configured = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger for the given module name."""
    _configure_root(level)
    return logging.getLogger(name)
