from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class NoOpLogger:
    def add_scalar(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


def build_logger(backend: str, log_dir: str):
    if backend == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=log_dir)
    return NoOpLogger()


def log_scalars(logger: Any, scalars: Dict[str, float], step: int) -> None:
    for k, v in scalars.items():
        logger.add_scalar(k, float(v), step)
