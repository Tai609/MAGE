"""
Deepseek model configuration module.
This module provides functions to initialize and configure Deepseek chat models.
"""

import os
import logging
from pathlib import Path
from langchain_openai import ChatOpenAI
from models.utils import check_model
from models.httpx_clients import build_httpx_clients
from dotenv import load_dotenv  

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
logger = logging.getLogger(__name__)

PROXY_BASE_URL = os.getenv("APIYI_BASE_URL", "https://api.apiyi.com/v1")
PROXY_API_KEY = (
    os.getenv("APIYI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("DEEPSEEK_API_KEY")
    or ""
)
temp = 0

def get_deepseek_model(model_name: str, temperature=temp, api_key: str = "", **kwargs):
    """Get a configured Deepseek chat model instance based on the provided model name.
    
    Args:
        model_name (str): The specific Deepseek model to use (e.g., 'deepseek-chat', 'deepseek-v3').
        temperature (float): Sampling temperature for the model
        api_key (str): API key for authentication. If empty, falls back to PROXY_API_KEY.
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        ChatOpenAI: Configured Deepseek chat model instance
    """
    
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

    resolved_api_key = api_key or PROXY_API_KEY
    if not resolved_api_key:
        logger.warning(
            "No API key found in APIYI_API_KEY/OPENAI_API_KEY/GOOGLE_API_KEY/DEEPSEEK_API_KEY; "
            "deepseek calls may fail."
        )

    model = ChatOpenAI(
        model=model_name,
        openai_api_base=PROXY_BASE_URL,
        openai_api_key=resolved_api_key,
        temperature=temperature,
        verbose=True,
        **kwargs
    )
    return model

if __name__ == "__main__":
    try:
        llm_model = get_deepseek_model(model_name="deepseek-chat")
        check_model(llm_model)
    except Exception as e:
        print(f"Error during testing: {e}")
