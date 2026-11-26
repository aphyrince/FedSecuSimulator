"""Persist lifecycle and request diagnostics for the FedSecuSimulator processes."""

from datetime import datetime, timezone
import faulthandler
import logging
import os
from pathlib import Path
import time


runtime_logger = logging.getLogger("fedsec.runtime")


def configure_runtime_logging(role: str) -> None:
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s pid=%(process)d thread=%(threadName)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    formatter.converter = time.gmtime
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    runtime_logger.addHandler(console)
    runtime_logger.setLevel(logging.INFO)
    runtime_logger.propagate = False

    directory = os.environ.get("FEDSEC_RUNTIME_LOG_DIR")
    if directory is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        directory = str(Path(__file__).resolve().parents[1] / "results" / "runtime_logs" / f"{stamp}-{os.getpid()}")
    path = Path(directory) / f"{role}-{os.getpid()}.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
    except OSError as error:
        runtime_logger.warning("runtime log file unavailable role=%s path=%s error=%s", role, path, error)
        faulthandler.enable(all_threads=True)
    else:
        os.environ["FEDSEC_RUNTIME_LOG_DIR"] = str(path.parent)
        file_handler.setFormatter(formatter)
        runtime_logger.addHandler(file_handler)
        faulthandler.enable(file=file_handler.stream, all_threads=True)
        runtime_logger.info("runtime log started role=%s path=%s", role, path)
