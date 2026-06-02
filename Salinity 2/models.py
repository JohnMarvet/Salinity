from __future__ import annotations

from typing import Any, Dict

from algo_params import build_model, default_params
from config import ALGORITHM_CHOICES


def list_algorithms() -> Dict[str, str]:
    return dict(ALGORITHM_CHOICES)


def get_model(
    algorithm: str,
    random_state: int = 42,
    params: Dict[str, Any] | None = None,
) -> Any:
    return build_model(algorithm, params or default_params(algorithm), random_state)


def get_default_params(algorithm: str) -> Dict[str, Any]:
    return default_params(algorithm)
