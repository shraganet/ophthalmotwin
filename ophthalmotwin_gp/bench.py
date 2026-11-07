from __future__ import annotations

import logging
import os
import random
import tempfile
from typing import Any

import numpy as np
import torch

_LOGGER_NAME = "ophthalmotwin_gp"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name or _LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unknown device preference {prefer!r}")
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" or torch.cuda.is_available():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "cuda":
            raise RuntimeError("cuda requested but not available")
    return torch.device("cpu")


def atomic_save(payload: dict[str, Any], path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(handle)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_load(path: str, map_location: str = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint at {path} is not a state mapping")
    return payload


def count_parameters(module: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())
