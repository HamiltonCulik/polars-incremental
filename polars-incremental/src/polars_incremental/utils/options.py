from __future__ import annotations

from typing import Any


def get_option(options: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in options:
            return options[key]
    return default
