# Copyright Sierra

"""
Provider configuration and setup utilities for model integration.
Handles setup of different LLM providers (DashScope, OpenRouter, local, etc.)
"""

import os
from typing import Optional, Tuple


def setup_provider(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    api_key_override: Optional[str] = None,
    dashscope_region: str = "singapore",
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Set up provider configuration and return (api_base_url, api_key, formatted_model).

    Args:
        provider: Provider name (e.g., "dashscope", "openrouter", "openai", "local")
        model: Model name (e.g., "qwen3-8b", "openai/gpt-oss-20b", "gpt-4o")
        base_url: Optional custom base URL (for local servers or overrides)
        api_key_override: Optional API key override
        dashscope_region: DashScope region ("singapore" or "us")

    Returns:
        Tuple of (api_base_url, api_key, formatted_model) where formatted_model
        has the correct litellm provider prefix (e.g., "openrouter/qwen/qwen3-4b")
    """
    from .dashscope import DashScopeConfig
    from .openrouter import OpenRouterConfig

    # DashScope models - use OpenAI-compatible API via litellm
    if model in DashScopeConfig.MODELS or provider == "dashscope":
        api_key = api_key_override or DashScopeConfig.get_api_key()
        endpoint = base_url or DashScopeConfig.get_endpoint(dashscope_region)
        # DashScope uses OpenAI-compatible API, prefix with openai/ for litellm
        formatted_model = f"openai/{model}" if not model.startswith("openai/") else model
        return (endpoint, api_key, formatted_model)

    # OpenRouter models - prefix with openrouter/ for litellm
    if model in OpenRouterConfig.MODELS or provider == "openrouter":
        api_key = api_key_override or OpenRouterConfig.get_api_key()
        endpoint = base_url or OpenRouterConfig.BASE_URL
        # Prefix with openrouter/ for litellm to recognize the provider
        formatted_model = f"openrouter/{model}" if not model.startswith("openrouter/") else model
        return (endpoint, api_key, formatted_model)

    # Local models (base_url provided but not standard provider)
    if provider == "local" and base_url:
        api_key = api_key_override or "local-model-key"  # Placeholder for local
        # Local servers typically use OpenAI-compatible API
        formatted_model = f"openai/{model}" if not model.startswith("openai/") else model
        return (base_url, api_key, formatted_model)

    # Default providers (OpenAI, Anthropic, etc. - use litellm defaults)
    # If custom base_url provided, use it (for custom endpoints)
    if base_url:
        return (base_url, api_key_override, model)

    return (None, None, model)
