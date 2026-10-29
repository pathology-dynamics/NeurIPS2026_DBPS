"""Optimizer factory + lightweight CSV/stdout logger for training."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

import torch


PathLike = Union[str, Path]


def make_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Adam with configurable beta1 and weight decay."""
    return torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
        betas=(float(beta1), float(beta2)),
        eps=float(eps),
        weight_decay=float(weight_decay),
    )


class TrainLogger:
    """Buffered CSV + stdout logger.

    Schema: epoch, step, kind, loss   (kind in {"train", "val"}).
    Training writes "train" rows only.

    Writes are flushed on `close()`. The constructor truncates any
    existing file at `path` so reruns don't accumulate stale rows.
    """

    def __init__(self, path: PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="")
        self._writer = csv.writer(self._f)
        self._writer.writerow(["epoch", "step", "kind", "loss"])
        self._closed = False

    def log_train(self, epoch: int, step: int, loss: float) -> None:
        self._writer.writerow([int(epoch), int(step), "train", f"{float(loss):.6e}"])

    def log_val(self, epoch: int, step: int, loss: float) -> None:
        self._writer.writerow([int(epoch), int(step), "val", f"{float(loss):.6e}"])

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        if not self._closed:
            self._f.flush()
            self._f.close()
            self._closed = True

    def __enter__(self) -> "TrainLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
