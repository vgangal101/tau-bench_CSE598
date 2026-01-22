# Copyright Sierra

"""
DashScope (Alibaba Cloud) API configuration for Qwen models.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DashScopeConfig:
    """Configuration for DashScope API (Alibaba Qwen models)"""

    ENDPOINTS = {
        "singapore": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        # China endpoint excluded - not accessible for this project
    }

    MODELS = [
        "qwen3-4b",
        "qwen3-8b",
        "qwen3-14b",
        "qwen3-32b",
    ]

    @staticmethod
    def get_api_key() -> str:
        """Load API key from environment"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY environment variable not set. "
                "Please set it in your .env file or environment."
            )
        return api_key

    @staticmethod
    def get_endpoint(region: str = "singapore") -> str:
        """Get endpoint URL for region"""
        if region not in DashScopeConfig.ENDPOINTS:
            raise ValueError(
                f"Invalid region: {region}. Must be 'singapore' or 'us'"
            )
        return DashScopeConfig.ENDPOINTS[region]
