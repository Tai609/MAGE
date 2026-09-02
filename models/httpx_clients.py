"""
Helpers for constructing httpx clients used by LLM providers.
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import httpx


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_httpx_clients(timeout: Any = None) -> Tuple[httpx.Client, httpx.AsyncClient]:
    """
    Build sync/async httpx clients for OpenAI-compatible SDK calls.

    By default, we ignore environment proxy variables to avoid accidental
    proxy hijacking from shell/session-level settings.
    Set MAGE_HTTPX_TRUST_ENV=1 to re-enable trust_env behavior.
    """
    trust_env = _env_flag("MAGE_HTTPX_TRUST_ENV", default=False)
    kwargs: dict[str, Any] = {"trust_env": trust_env}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return httpx.Client(**kwargs), httpx.AsyncClient(**kwargs)

