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
            'Decide your next action(s). You can BATCH multiple tool calls in a single turn to '
            'minimize round trips. Output a JSON array of actions:\n'
            '\n'
            'For multiple tool calls (PREFERRED when you need several lookups):\n'
            '[{"name": "<tool_1>", "kwargs": {<args>}}, {"name": "<tool_2>", "kwargs": {<args>}}, ...]\n'
            '\n'
            'For a single action:\n'
            '[{"name": "<tool_name>", "kwargs": {<arguments>}}]\n'
            '\n'
            'To respond to the customer (must be the ONLY action, never batched with tool calls):\n'
            '[{"name": "respond", "kwargs": {"content": "<your message>"}}]\n'
            '\n'
            'IMPORTANT: Batch as many independent tool calls as possible in one turn. For example, '
            'if you need to look up a user AND check an order, do BOTH in one turn instead of '
            'separate turns. Only respond to the customer when you have all the information needed.\n'
            '\n'
            'Output the JSON array and nothing else.'
        )

        return "\n".join(sections)
