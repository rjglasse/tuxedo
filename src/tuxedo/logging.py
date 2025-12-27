"""Logging configuration for Tuxedo."""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Module-level logger
_logger: logging.Logger | None = None
_log_file: Path | None = None


def setup_logging(
    log_dir: Path | None = None,
    level: int = logging.INFO,
    console: bool = False,
) -> logging.Logger:
    """Set up logging for the application.

    Args:
        log_dir: Directory for log files. If None, uses ~/.tuxedo/logs
        level: Logging level (default: INFO)
        console: If True, also log to stderr

    Returns:
        Configured logger instance
    """
    global _logger, _log_file

    if _logger is not None:
        return _logger

    # Determine log directory
    if log_dir is None:
        log_dir = Path.home() / ".tuxedo" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file = log_dir / f"tuxedo_{timestamp}.log"

    # Create logger
    _logger = logging.getLogger("tuxedo")
    _logger.setLevel(level)
    _logger.handlers.clear()  # Remove any existing handlers

    # File handler
    file_handler = logging.FileHandler(_log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    _logger.addHandler(file_handler)

    # Console handler (optional, for debugging)
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_format = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_format)
        _logger.addHandler(console_handler)

    _logger.info(f"Logging initialized. Log file: {_log_file}")

    return _logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Optional sub-logger name (e.g., 'clustering', 'grobid')

    Returns:
        Logger instance
    """
    global _logger

    if _logger is None:
        setup_logging()

    if name:
        return _logger.getChild(name)
    return _logger


def get_log_file() -> Path | None:
    """Get the current log file path."""
    return _log_file


def cleanup_old_logs(log_dir: Path | None = None, keep_days: int = 7) -> int:
    """Remove log files older than keep_days.

    Args:
        log_dir: Directory containing log files
        keep_days: Number of days to keep logs

    Returns:
        Number of files removed
    """
    if log_dir is None:
        log_dir = Path.home() / ".tuxedo" / "logs"

    if not log_dir.exists():
        return 0

    cutoff = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
    removed = 0

    for log_file in log_dir.glob("tuxedo_*.log"):
        if log_file.stat().st_mtime < cutoff:
            log_file.unlink()
            removed += 1

    return removed
