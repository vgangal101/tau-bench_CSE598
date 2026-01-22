# Copyright Sierra

"""
Response format normalization for different LLM providers.
Handles thinking/reasoning modes and extracts clean content.
"""

import re
from typing import Dict, Any, Tuple, Optional


def normalize_response(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize response from different providers to consistent format.

    Extracts clean content (without thinking/reasoning) and optionally
    preserves thinking for logging/debugging.

    Args:
        message: Raw message dict from litellm completion

    Returns:
        Normalized message dict with:
        - content: Clean response (thinking removed)
        - thinking: Thinking/reasoning content (if present)
        - tool_calls: Tool calls (unchanged if present)
    """
    # If tool call, return as-is (no content parsing needed)
    if message.get("tool_calls"):
        return message

    content = message.get("content", "")
    thinking = None

    # Strategy 1: Check for separate 'reasoning' field (OpenRouter format)
    if "reasoning" in message and message["reasoning"]:
        thinking = message["reasoning"]
        # Content is the actual answer (already separate)
        clean_content = content or ""

    # Strategy 2: Check for <think> tags in content
    elif "<think>" in content or "<thinking>" in content:
        thinking, clean_content = extract_thinking_tags(content)

    # Strategy 3: Check for other common thinking patterns
    elif "Let me think" in content or "Thought:" in content:
        thinking, clean_content = extract_reasoning_patterns(content)

    # No thinking found - return content as-is
    else:
        clean_content = content

    # Create normalized message
    normalized = message.copy()
    normalized["content"] = clean_content.strip()

    # Add thinking to metadata (for logging/debugging)
    if thinking:
        if "metadata" not in normalized:
            normalized["metadata"] = {}
        normalized["metadata"]["thinking"] = thinking.strip()

    return normalized


def extract_thinking_tags(content: str) -> Tuple[Optional[str], str]:
    """
    Extract thinking from <think> or <thinking> tags.

    Returns:
        (thinking_content, answer_content)
    """
    # Try <think> tags first
    think_pattern = r"<think>(.*?)</think>"
    match = re.search(think_pattern, content, re.DOTALL | re.IGNORECASE)

    if match:
        thinking = match.group(1).strip()
        answer = re.sub(
            think_pattern, "", content, flags=re.DOTALL | re.IGNORECASE
        ).strip()
        return thinking, answer

    # Try <thinking> tags
    thinking_pattern = r"<thinking>(.*?)</thinking>"
    match = re.search(thinking_pattern, content, re.DOTALL | re.IGNORECASE)

    if match:
        thinking = match.group(1).strip()
        answer = re.sub(
            thinking_pattern, "", content, flags=re.DOTALL | re.IGNORECASE
        ).strip()
        return thinking, answer

    return None, content


def extract_reasoning_patterns(content: str) -> Tuple[Optional[str], str]:
    """
    Extract thinking from common reasoning patterns.

    Patterns:
    - "Let me think... [thinking] ... Answer: [answer]"
    - "Thought: [thinking]\nAction: [answer]"
    - etc.

    Returns:
        (thinking_content, answer_content)
    """
    # Pattern 1: "Answer:" marker
    if "Answer:" in content:
        parts = content.split("Answer:", 1)
        return parts[0].strip(), parts[1].strip()

    # Pattern 2: "Response:" marker
    if "Response:" in content:
        parts = content.split("Response:", 1)
        return parts[0].strip(), parts[1].strip()

    # Pattern 3: Thought/Action (ReAct format)
    if "Thought:" in content and ("Action:" in content or "Response:" in content):
        # Extract everything before Action/Response as thinking
        for marker in ["Action:", "Response:"]:
            if marker in content:
                parts = content.split(marker, 1)
                return parts[0].strip(), parts[1].strip()

    # No clear pattern - return as-is
    return None, content
