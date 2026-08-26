"""Console + file logging, one log per results directory."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(run_dir: Path | str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the root ``tzsad`` logger; tee to ``run_dir/run.log`` when given."""
    logger = logging.getLogger("tzsad")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(stream)

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(run_dir / "run.log")
        fh.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(fh)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Child logger under the ``tzsad`` root."""
    return logging.getLogger(f"tzsad.{name}")
