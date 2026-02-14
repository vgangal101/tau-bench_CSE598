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
            'Decide your next action. Output ONLY a JSON object in one of these formats:\n'
            '- To call a tool: {"name": "<tool_name>", "kwargs": {<arguments>}}\n'
            '- To respond to the customer: {"name": "respond", "kwargs": {"content": "<your message>"}}\n'
            "\n"
            "Output the JSON and nothing else."
        )

        return "\n".join(sections)
