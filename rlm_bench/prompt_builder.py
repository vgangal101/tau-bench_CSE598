import json
from typing import List, Dict, Any


class PromptBuilder:
    def __init__(self, wiki: str, tools_info: List[Dict[str, Any]]):
        self.wiki = wiki
        self.tools_description = self._format_tools(tools_info)

    def _format_tools(self, tools_info: List[Dict[str, Any]]) -> str:
        lines = []
        for tool in tools_info:
            func = tool.get("function", tool)
            name = func["name"]
            desc = func.get("description", "")
            params = func.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])

            param_lines = []
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                param_lines.append(f"    - {pname} ({ptype}{req}): {pdesc}")

            lines.append(f"- {name}: {desc}")
            if param_lines:
                lines.extend(param_lines)
        return "\n".join(lines)

    def build_prompt(self, conversation_history: List[Dict[str, str]]) -> str:
        sections = []

        sections.append("# Policy and Domain Knowledge")
        sections.append(self.wiki)

        sections.append("\n# Available Tools")
        sections.append(self.tools_description)

        sections.append("\n# Conversation History")
        for entry in conversation_history:
            role = entry["role"]
            content = entry["content"]
            sections.append(f"[{role}]: {content}")

        sections.append("\n# Instructions")
        sections.append(
            'You are a customer service agent. Your job is to EXECUTE actions, not just discuss them.\n'
            '\n'
            'Output a JSON array of actions. RULES:\n'
            '\n'
            '1. BATCH independent tool calls in one turn:\n'
            '   [{"name": "tool_1", "kwargs": {...}}, {"name": "tool_2", "kwargs": {...}}]\n'
            '   Example: if you have 5 order IDs, look up ALL of them in one turn.\n'
            '\n'
            '2. Single tool call:\n'
            '   [{"name": "tool_name", "kwargs": {...}}]\n'
            '\n'
            '3. Respond to customer (ONLY when you have all info or need input):\n'
            '   [{"name": "respond", "kwargs": {"content": "your message"}}]\n'
            '   Never batch respond with tool calls.\n'
            '\n'
            '4. EXECUTE mutation tools (exchange, return, modify, cancel) when you have the\n'
            '   required parameters. Do NOT just describe what you would do — actually call the tool.\n'
            '   If the customer confirmed an action and you have all the IDs, call the tool NOW.\n'
            '\n'
            '5. When the customer provides their name/zip/email, immediately look them up.\n'
            '   When you get a user with order IDs, batch-fetch ALL order details in one turn.\n'
            '\n'
            'Output ONLY the JSON array. No explanations, no markdown, no commentary.'
        )

        return "\n".join(sections)
