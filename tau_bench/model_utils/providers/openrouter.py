# Copyright Sierra

"""
OpenRouter API configuration for Qwen and other models.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class OpenRouterConfig:
    """Configuration for OpenRouter API"""

    BASE_URL = "https://openrouter.ai/api/v1"

    MODELS = [
        "qwen/qwen3-8b",
        "qwen/qwen3-14b",
        "qwen/qwen3-32b",
        "openai/gpt-oss-20b",
    ]

    @staticmethod
    def get_api_key() -> str:
        """Load API key from environment"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Please set it in your .env file or environment."
            )
        return api_key
