"""Human-readable and machine-readable logging setup."""

import json
import logging
from datetime import datetime, timezone
from typing import Any


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    logger.info(json.dumps(payload, sort_keys=True, default=str))
