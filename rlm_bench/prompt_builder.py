import json
from typing import List, Dict, Any


# Maximum number of conversation history entries to include in the prompt.
# Each turn is 2 entries (agent + customer/tool_result), so 30 entries ≈ 15 turns.
MAX_HISTORY_ENTRIES = 30


class PromptBuilder:
    def __init__(self, wiki: str, tools_info: List[Dict[str, Any]]):
        self.wiki = wiki
        self.tools_description = self._format_tools(tools_info)
        # Extract tool names for use in instruction examples (prevents placeholder copying)
        self.tool_names = [
            tool.get("function", tool)["name"] for tool in tools_info
        ]

    def _format_tools(self, tools_info: List[Dict[str, Any]]) -> str:
        lines = []
        for tool in tools_info:
            func = tool.get("function", tool)
            name = func["name"]
            desc = func.get("description", "")

            # Strengthen transfer tool description to discourage premature use
            if name == "transfer_to_human_agents":
                desc = (
                    "LAST RESORT ONLY. Transfer to a human agent. "
                    "You must NEVER call this unless: (a) the customer explicitly asks "
                    "for a human, OR (b) you have already looked up the account and "
                    "confirmed the issue cannot be resolved with available tools."
                )
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

        # Role header at the top — first thing the model sees when it reads context
        sections.append("# YOUR TASK")
        sections.append(
            "You are a customer service agent. Read the policy, tools, and conversation below, "
            "then output a JSON action array. Do NOT summarize the policy. Do NOT output prose."
        )

        sections.append("\n# Policy and Domain Knowledge")
        sections.append(self.wiki)

        sections.append("\n# Available Tools")
        sections.append(self.tools_description)

        sections.append("\n# Conversation History")
        # Keep the first entry (original customer request) and the most recent turns
        if len(conversation_history) > MAX_HISTORY_ENTRIES:
            sections.append(f"[{conversation_history[0]['role']}]: {conversation_history[0]['content']}")
            sections.append(f"[... {len(conversation_history) - MAX_HISTORY_ENTRIES} earlier messages omitted ...]")
            trimmed = conversation_history[-MAX_HISTORY_ENTRIES + 1:]
        else:
            trimmed = conversation_history
        for entry in trimmed:
            role = entry["role"]
            content = entry["content"]
            sections.append(f"[{role}]: {content}")

        # First-turn guidance: prevent the model from hallucinating prior interaction
        if len(conversation_history) <= 2:
            sections.append("\n# IMPORTANT: This is the START of the conversation")
            sections.append(
                "The customer just reached out. You have NOT asked them anything yet.\n"
                "Your action: respond and ask for their user ID so you can look up their account.\n"
                "Do NOT call any tools yet (you have no user ID to look up).\n"
                "Do NOT transfer to a human agent."
            )

        sections.append("\n# Instructions")

        # Use real tool names in examples to prevent placeholder copying
        ex1 = self.tool_names[0] if self.tool_names else "tool_name"
        ex2 = self.tool_names[1] if len(self.tool_names) > 1 else "tool_name_2"

        sections.append(
            'You are a customer service agent. Your job is to EXECUTE actions using tools, not discuss policies.\n'
            '\n'
            'Output ONLY a valid JSON array. No text before or after. No markdown. No commentary.\n'
            '\n'
            '## Output Format\n'
            '\n'
            'Tool calls:\n'
            f'  [{{"name": "{ex1}", "kwargs": {{"param": "value"}}}}]\n'
            '\n'
            'Batch independent tool calls:\n'
            f'  [{{"name": "{ex1}", "kwargs": {{...}}}}, {{"name": "{ex2}", "kwargs": {{...}}}}]\n'
            '\n'
            'Respond to customer:\n'
            '  [{"name": "respond", "kwargs": {"content": "your message"}}]\n'
            '\n'
            '## Action Rules\n'
            '\n'
            '1. ALWAYS prefer tool calls over responding. Only respond when you genuinely need\n'
            '   information from the customer or are delivering a final answer.\n'
            '2. When the customer provides identifying info (name, email, user ID), immediately\n'
            '   call the lookup tool. When you get order/reservation IDs, batch-fetch ALL details.\n'
            '3. EXECUTE mutation tools (modify, cancel, exchange, return) when you have the\n'
            '   required parameters. Do NOT just describe what you would do — call the tool.\n'
            '4. Never batch a respond action with tool calls.\n'
            '\n'
            '## Critical Prohibitions\n'
            '\n'
            '- Do NOT summarize or recite the policy. The customer does not need policy text.\n'
            '- Do NOT fabricate data (reservation IDs, amounts, dates) that the customer did not provide.\n'
            '- Do NOT repeat the same question the customer just asked you.\n'
            '- Do NOT output anything other than a JSON array. No prose, no bullet points, no analysis.\n'
            '- Keep responses SHORT and actionable (1-2 sentences max).\n'
            '- Do NOT call transfer_to_human_agents unless the customer explicitly requests a human agent.\n'
            '  You MUST first: ask for their user ID, look up their account, and attempt to resolve the issue.'
        )

        return "\n".join(sections)
