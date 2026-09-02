"""
OpenAI model configuration module.
This module defaults to the official OpenAI endpoint and supports explicitly
configured OpenAI-compatible endpoints.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
try:
    from .httpx_clients import build_httpx_clients
    from .provider_config import resolve_provider_connection
except ImportError:
    from models.httpx_clients import build_httpx_clients
    from models.provider_config import resolve_provider_connection

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _resolve_api_key(default: str = "", base_url: str = "") -> str:
    api_key, _ = resolve_provider_connection(
        "openai",
        explicit_api_key=default,
        explicit_base_url=base_url,
    )
    return api_key


def _resolve_base_url(default: str = "") -> str:
    _, base_url = resolve_provider_connection("openai", explicit_base_url=default)
    return base_url


@dataclass
class _OpenAIInvokeResult:
    content: str
    raw_response: Any | None = None


class OpenAIProxyChatModel:
    """Small invoke-compatible wrapper backed by the official OpenAI SDK."""

    def __init__(self, model_name: str, temperature: float = 0, **kwargs: Any):
        explicit_api_key = kwargs.pop("api_key", "")
        explicit_base_url = kwargs.pop("base_url", "")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = kwargs.pop("max_tokens", int(os.getenv("LLM_MAX_TOKENS", "8192")))
        self.timeout = kwargs.pop(
            "timeout",
            kwargs.pop("request_timeout", float(os.getenv("LLM_TIMEOUT_SECONDS", "300"))),
        )
        self.max_retries = kwargs.pop("max_retries", int(os.getenv("LLM_MAX_RETRIES", "0")))
        self.request_kwargs = dict(kwargs)
        self.http_client, _ = build_httpx_clients(timeout=self.timeout)
        resolved_api_key, resolved_base_url = resolve_provider_connection(
            "openai",
            explicit_api_key=explicit_api_key,
            explicit_base_url=explicit_base_url,
        )
        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client=self.http_client,
        )

    def invoke(self, prompt: str) -> _OpenAIInvokeResult:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.request_kwargs,
        )
        message = response.choices[0].message
        content = message.content or ""
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                else:
                    text_value = getattr(part, "text", None)
                    if text_value:
                        text_parts.append(text_value)
            content = "".join(text_parts)
        return _OpenAIInvokeResult(content=str(content), raw_response=response)


def get_openai_model(model_name: str, temperature=0, **kwargs):
    """Get a configured OpenAI or explicitly routed compatible chat model."""
    explicit_api_key = kwargs.pop("openai_api_key", kwargs.pop("api_key", ""))
    explicit_base_url = kwargs.pop("openai_api_base", kwargs.pop("base_url", ""))
    resolved_api_key, resolved_base_url = resolve_provider_connection(
        "openai",
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )

    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS", "8192"))

    if "timeout" not in kwargs and "request_timeout" not in kwargs:
        kwargs["timeout"] = float(os.getenv("LLM_TIMEOUT_SECONDS", "300"))

    if "max_retries" not in kwargs:
        kwargs["max_retries"] = int(os.getenv("LLM_MAX_RETRIES", "0"))

    timeout_for_client = kwargs.get("timeout")
    if "http_client" not in kwargs or kwargs.get("http_client") is None:
        http_client, http_async_client = build_httpx_clients(timeout=timeout_for_client)
        kwargs["http_client"] = http_client
        kwargs["http_async_client"] = http_async_client

    # Force explicit value so ChatOpenAI does not implicitly read OPENAI_PROXY.
    if "openai_proxy" not in kwargs:
        kwargs["openai_proxy"] = None

    if ChatOpenAI is not None:
        model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_base=resolved_base_url,
            openai_api_key=resolved_api_key,
            verbose=True,
            **kwargs,
        )
        return model

    kwargs.pop("http_client", None)
    kwargs.pop("http_async_client", None)
    kwargs.pop("openai_proxy", None)
    return OpenAIProxyChatModel(
        model_name=model_name,
        temperature=temperature,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        **kwargs,
    )


def get_available_models():
    """Get a list of models from the resolved OpenAI-compatible endpoint."""
    try:
        http_client, _ = build_httpx_clients()
        api_key, base_url = resolve_provider_connection("openai")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )
        models = client.models.list()
        return [m.id for m in models.data]
    except Exception as e:
        print(f"Error fetching models from configured API: {e}")
        return []


if __name__ == "__main__":
    try:
        test_model_name = "gpt-4o-mini"
        llm_model = get_openai_model(model_name=test_model_name, temperature=0.1)
        print(f"Successfully created model: {test_model_name}")
        response = llm_model.invoke("What is the capital of France?")
        print(response.content)
    except Exception as e:
        print(f"Error during testing: {e}")
