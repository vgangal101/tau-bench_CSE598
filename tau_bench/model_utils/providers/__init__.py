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
) -> Tuple[Optional[str], Optional[str]]:
    """
    Set up provider configuration and return (api_base_url, api_key).

    Args:
        provider: Provider name (e.g., "dashscope", "openrouter", "openai", "local")
        model: Model name (e.g., "qwen3-8b", "openai/gpt-oss-20b", "gpt-4o")
        base_url: Optional custom base URL (for local servers or overrides)
        api_key_override: Optional API key override
        dashscope_region: DashScope region ("singapore" or "us")

    Returns:
        Tuple of (api_base_url, api_key) - both can be None for default providers
    """
    from .dashscope import DashScopeConfig
    from .openrouter import OpenRouterConfig

    # DashScope models
    if model in DashScopeConfig.MODELS or provider == "dashscope":
        api_key = api_key_override or DashScopeConfig.get_api_key()
        endpoint = base_url or DashScopeConfig.get_endpoint(dashscope_region)
        return (endpoint, api_key)

    # OpenRouter models
    if model in OpenRouterConfig.MODELS or provider == "openrouter":
        api_key = api_key_override or OpenRouterConfig.get_api_key()
        endpoint = base_url or OpenRouterConfig.BASE_URL
        return (endpoint, api_key)

    # Local models (base_url provided but not standard provider)
    if provider == "local" and base_url:
        api_key = api_key_override or "local-model-key"  # Placeholder for local
        return (base_url, api_key)

    # Default providers (OpenAI, Anthropic, etc. - use litellm defaults)
    # If custom base_url provided, use it (for custom endpoints)
    if base_url:
        return (base_url, api_key_override)

    return (None, None)
