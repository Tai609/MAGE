"""Safe endpoint and credential selection for supported model providers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import urlparse


OFFICIAL_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "deepseek": "https://api.deepseek.com/v1",
}

PROVIDER_KEY_VARIABLES = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

PROVIDER_BASE_VARIABLES = {
    "openai": "OPENAI_API_BASE",
    "google": "GOOGLE_API_BASE",
    "deepseek": "DEEPSEEK_API_BASE",
}


def _clean(value: object) -> str:
    return str(value or "").strip()


def _hostname(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _is_apiyi_endpoint(base_url: str, configured_proxy_url: str) -> bool:
    if _hostname(base_url) == "api.apiyi.com":
        return True
    return bool(configured_proxy_url) and base_url.rstrip("/") == configured_proxy_url.rstrip("/")


def resolve_provider_connection(
    provider: str,
    *,
    explicit_api_key: str = "",
    explicit_base_url: str = "",
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve a provider key and endpoint without crossing credential domains.

    Official provider endpoints are the default. APIYi is used only when
    ``APIYI_BASE_URL`` (or an explicit APIYi URL) is supplied, and it receives
    only ``APIYI_API_KEY`` unless the caller explicitly passes a key.
    """

    provider_name = _clean(provider).lower()
    if provider_name not in OFFICIAL_BASE_URLS:
        raise ValueError(f"Unsupported provider: {provider!r}")

    env = os.environ if environ is None else environ
    provider_base = _clean(env.get(PROVIDER_BASE_VARIABLES[provider_name], ""))
    proxy_base = _clean(env.get("APIYI_BASE_URL", ""))
    base_url = (
        _clean(explicit_base_url)
        or provider_base
        or proxy_base
        or OFFICIAL_BASE_URLS[provider_name]
    ).rstrip("/")

    api_key = _clean(explicit_api_key)
    if not api_key:
        if _is_apiyi_endpoint(base_url, proxy_base):
            api_key = _clean(env.get("APIYI_API_KEY", ""))
        else:
            api_key = _clean(env.get(PROVIDER_KEY_VARIABLES[provider_name], ""))

    return api_key, base_url
