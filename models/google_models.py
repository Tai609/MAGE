"""
Module for Gemini models through Google's official OpenAI-compatible endpoint
or an explicitly configured gateway.
"""

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks.usage import get_usage_metadata_callback
from langchain_openai import ChatOpenAI
from models.httpx_clients import build_httpx_clients
from models.provider_config import resolve_provider_connection

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


logger = logging.getLogger(__name__)


def get_google_model(model_name: str, **kwargs: Any) -> ChatOpenAI:
    """
    Initializes and returns a Gemini model instance through a safe endpoint.

    Args:
        model_name: Model name (e.g. 'gemini-3-pro-preview-thinking').
        **kwargs: Extra args forwarded to ChatOpenAI.
    """
    try:
        explicit_api_key = kwargs.pop("openai_api_key", kwargs.pop("api_key", ""))
        explicit_base_url = kwargs.pop("openai_api_base", kwargs.pop("base_url", ""))
        resolved_api_key, resolved_base_url = resolve_provider_connection(
            "google",
            explicit_api_key=explicit_api_key,
            explicit_base_url=explicit_base_url,
        )

        if "model" not in kwargs:
            kwargs["model"] = model_name
        elif kwargs["model"] != model_name:
            logger.warning(
                "Overriding 'model' kwarg ('%s') with model_name ('%s').",
                kwargs["model"],
                model_name,
            )
            kwargs["model"] = model_name

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

        if not resolved_api_key:
            logger.warning(
                "No Google credential found for the resolved endpoint; model calls may fail."
            )

        logger.info("Initializing Gemini through the resolved endpoint: %s", model_name)
        llm = ChatOpenAI(
            openai_api_base=resolved_base_url,
            openai_api_key=resolved_api_key,
            **kwargs,
        )
        return llm
    except ImportError as e:
        logger.error("Failed to import ChatOpenAI. Ensure 'langchain-openai' is installed.")
        raise ImportError(
            "The 'langchain-openai' package is required for Gemini calls."
        ) from e
    except Exception as e:
        logger.error("Error initializing Google model '%s': %s", model_name, e)
        raise


if __name__ == "__main__":
    model = get_google_model("gemini-3-pro-preview-thinking")
    with get_usage_metadata_callback() as usage_callback:
        response = model.invoke("What is the capital of France?")
        print(response)
        print(usage_callback.usage_metadata)
