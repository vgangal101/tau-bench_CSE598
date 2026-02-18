import json
import re
from typing import List, Optional, Dict, Any

from rlm import RLM

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import Action, SolveResult, RESPOND_ACTION_NAME

from rlm_bench.prompt_builder import PromptBuilder

# Patterns that indicate RLM REPL internal errors (should never reach the customer)
_REPL_ERROR_PATTERNS = [
    "Error: Variable",
    "not found. Available variables:",
    "You must create and assign a variable BEFORE calling FINAL_VAR",
]

# Litellm/backend artifacts that leak into responses and corrupt conversation history
_LITELLM_ARTIFACTS = [
    "Provider List: https://docs.litellm.ai/docs/providers",
]

MAX_RLM_RETRIES = 2
# Max consecutive respond-only turns before injecting a refocus nudge
MAX_CONSECUTIVE_RESPONDS = 4

# Custom system prompt that replaces the RLM's default 90-line document-analysis prompt.
# Orients the model as an agent while preserving REPL mechanics (```repl``` blocks, FINAL()).
AGENT_SYSTEM_PROMPT = (
    "You are a customer service agent. Your task instructions are in the `context` variable.\n\n"
    "The context contains: domain policy, available tools, conversation history, and output format instructions.\n\n"
    "Workflow:\n"
    "1. Read the context: ```repl\nprint(context)\n```\n"
    "2. Decide your action based on the conversation history and available tools.\n"
    "3. Output FINAL([json_array]) with your action.\n\n"
    "Action examples:\n"
    '  FINAL([{"name": "get_user_details", "kwargs": {"user_id": "sara_doe_496"}}])\n'
    '  FINAL([{"name": "respond", "kwargs": {"content": "your message"}}])\n\n'
    "CRITICAL RULES:\n"
    "- ALWAYS read context first in a ```repl``` block before outputting FINAL.\n"
    "- Output ONLY a JSON array inside FINAL(). No prose, no policy summaries, no commentary.\n"
    "- Prefer tool calls over responding to the customer.\n"
    "- Do NOT use llm_query or llm_query_batched. Everything you need is in context.\n"
)

# Root prompt inserted at every REPL iteration to keep the model focused on JSON output.
ROOT_PROMPT = (
    "Read the context and output FINAL([json_array]) with your action. "
    "The json_array must contain objects with 'name' and 'kwargs' keys. No prose."
)

# Corrective suffix appended to the prompt when the model outputs prose instead of JSON.
CORRECTIVE_SUFFIX = (
    "\n\n# CORRECTIVE INSTRUCTION\n"
    "Your previous response was not valid JSON actions. "
    'Output FINAL([{"name": "tool_or_respond", "kwargs": {...}}]) now. No prose.'
)


class RLMAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        max_depth: int = 1,
        environment: str = "local",
        model_base_url: Optional[str] = None,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.prompt_builder = PromptBuilder(wiki=wiki, tools_info=tools_info)

        backend_kwargs = {"model_name": model}
        if temperature > 0:
            backend_kwargs["temperature"] = temperature
        if model_base_url:
            backend_kwargs["api_base"] = model_base_url

        self.rlm = RLM(
            backend=provider,
            backend_kwargs=backend_kwargs,
            environment=environment,
            max_depth=max_depth,
            custom_system_prompt=AGENT_SYSTEM_PROMPT,
        )

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Strip litellm/backend artifacts that leak into responses."""
        for artifact in _LITELLM_ARTIFACTS:
            text = text.replace(artifact, "")
        # Collapse runs of blank lines left behind
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        return text.strip()

    @staticmethod
    def _is_repl_error(response_text: str) -> bool:
        """Check if the RLM response is an internal REPL error, not a real answer."""
        return any(pattern in response_text for pattern in _REPL_ERROR_PATTERNS)

    @staticmethod
    def _is_valid_action_response(response_text: str) -> bool:
        """Check if the response contains at least one parseable JSON action."""
        text = response_text.strip()
        if '"name"' not in text or '"kwargs"' not in text:
            return False
        # Try direct parse as JSON array/object with action structure
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and any(
                isinstance(item, dict) and "name" in item and "kwargs" in item
                for item in parsed
            ):
                return True
            if isinstance(parsed, dict) and "name" in parsed and "kwargs" in parsed:
                return True
        except (json.JSONDecodeError, KeyError):
            pass
        # Try to find an embedded JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list) and any(
                    isinstance(item, dict) and "name" in item and "kwargs" in item
                    for item in parsed
                ):
                    return True
            except (json.JSONDecodeError, KeyError):
                pass
        return False

    def _parse_actions(self, response_text: str) -> List[Action]:
        """Parse one or more actions from the RLM response.

        Supports both batched (JSON array) and single (JSON object) formats.
        """
        text = response_text.strip()

        # Try 1: parse as JSON array of actions
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) > 0:
                actions = []
                for item in parsed:
                    if isinstance(item, dict) and "name" in item and "kwargs" in item:
                        actions.append(Action(name=item["name"], kwargs=item["kwargs"]))
                if actions:
                    return actions
        except (json.JSONDecodeError, KeyError):
            pass

        # Try 2: find a JSON array in the response
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    actions = []
                    for item in parsed:
                        if isinstance(item, dict) and "name" in item and "kwargs" in item:
                            actions.append(Action(name=item["name"], kwargs=item["kwargs"]))
                    if actions:
                        return actions
            except (json.JSONDecodeError, KeyError):
                pass

        # Try 3: find a single JSON object with name+kwargs
        match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"kwargs"\s*:\s*\{[^}]*\}[^}]*\}', text)
        if match:
            try:
                parsed = json.loads(match.group())
                return [Action(name=parsed["name"], kwargs=parsed["kwargs"])]
            except (json.JSONDecodeError, KeyError):
                pass

        # Try 4: find any nested JSON object with name+kwargs
        for match in re.finditer(r'\{[^{}]*\{[^{}]*\}[^{}]*\}', text):
            try:
                parsed = json.loads(match.group())
                if "name" in parsed and "kwargs" in parsed:
                    return [Action(name=parsed["name"], kwargs=parsed["kwargs"])]
            except (json.JSONDecodeError, KeyError):
                continue

        # Try 5: parse the whole response as a single JSON object
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "name" in parsed and "kwargs" in parsed:
                return [Action(name=parsed["name"], kwargs=parsed["kwargs"])]
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: treat entire response as a customer-facing message,
        # but cap length to avoid sending policy dumps or off-topic essays.
        if len(text) > 500:
            # Likely a policy summary or off-topic content; truncate to first sentence
            first_sentence_end = min(
                (text.find(". ") + 1) if ". " in text else len(text),
                (text.find(".\n") + 1) if ".\n" in text else len(text),
                500,
            )
            text = text[:first_sentence_end].strip()
            if not text.endswith("."):
                text += "."
            text = text + " How can I help you with your specific request?"

        return [Action(
            name=RESPOND_ACTION_NAME,
            kwargs={"content": text},
        )]

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = self._sanitize_text(env_reset_res.observation)
        info = env_reset_res.info.model_dump()
        reward = 0.0
        steps_used = 0
        consecutive_responds = 0  # track respond-only turns for loop detection

        conversation_history: List[Dict[str, str]] = [
            {"role": "customer", "content": obs},
        ]
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]

        done = False
        rlm_call_num = 0
        while steps_used < max_num_steps and not done:
            rlm_call_num += 1
            prompt = self.prompt_builder.build_prompt(conversation_history)

            print(f"\n{'='*80}")
            print(f"RLM CALL #{rlm_call_num} (steps used: {steps_used}/{max_num_steps})")
            print(f"{'='*80}")
            print(f"\n--- PROMPT SENT TO RLM ({len(prompt)} chars) ---")
            # Show last 1500 chars to avoid flooding terminal with the full wiki
            if len(prompt) > 2000:
                print(f"  [... first {len(prompt)-1500} chars truncated ...]")
                print(prompt[-1500:])
            else:
                print(prompt)

            # Retry loop: if RLM returns a REPL error, retry up to MAX_RLM_RETRIES times
            response_text = None
            for attempt in range(1 + MAX_RLM_RETRIES):
                result = self.rlm.completion(prompt, root_prompt=ROOT_PROMPT)
                response_text = self._sanitize_text(result.response)

                if self._is_repl_error(response_text):
                    print(f"\n--- RLM REPL ERROR (attempt {attempt+1}/{1+MAX_RLM_RETRIES}) ---")
                    print(f"  {response_text[:200]}")
                    if attempt < MAX_RLM_RETRIES:
                        print("  Retrying...")
                        continue
                    else:
                        print("  Max retries reached, skipping this turn.")
                        break
                else:
                    break

            # If all attempts returned REPL errors, skip this turn entirely
            if self._is_repl_error(response_text):
                print("  [SKIPPED — REPL error not sent to customer]")
                continue

            # Prose detection: if response is not valid JSON actions, retry once with correction
            if not self._is_valid_action_response(response_text):
                print(f"\n--- PROSE DETECTED (retrying with corrective prompt) ---")
                print(f"  First 200 chars: {response_text[:200]}")
                corrective_prompt = prompt + CORRECTIVE_SUFFIX
                result = self.rlm.completion(corrective_prompt, root_prompt=ROOT_PROMPT)
                response_text = self._sanitize_text(result.response)
                print(f"\n--- CORRECTIVE RLM RESPONSE ---")
                print(response_text[:500])

            print(f"\n--- RLM RESPONSE ---")
            print(response_text)

            actions = self._parse_actions(response_text)
            print(f"\n--- PARSED ACTIONS ({len(actions)}) ---")
            for i, a in enumerate(actions):
                print(f"  [{i+1}] {a.name}({json.dumps(a.kwargs)})")

            # --- Loop detection: all actions are respond-only? ---
            all_respond = all(a.name == RESPOND_ACTION_NAME for a in actions)
            if all_respond:
                consecutive_responds += 1
            else:
                consecutive_responds = 0

            if consecutive_responds >= MAX_CONSECUTIVE_RESPONDS:
                print(f"\n  [LOOP DETECTED] {consecutive_responds} consecutive respond-only turns — "
                      f"aborting to avoid wasting remaining steps.")
                # Send a final concise message so the env can score
                final_action = Action(
                    name=RESPOND_ACTION_NAME,
                    kwargs={"content": "I apologize, but I'm unable to assist further with this request. "
                            "Let me transfer you to a human agent who can help."},
                )
                env_response = env.step(final_action)
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}
                steps_used += 1
                messages.append({"role": "assistant", "content": final_action.kwargs["content"]})
                break

            messages.append({"role": "assistant", "content": response_text})

            # Execute all actions from this RLM call
            for action in actions:
                if steps_used >= max_num_steps:
                    break

                env_response = env.step(action)
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}
                steps_used += 1

                # Sanitize env observations
                env_obs = self._sanitize_text(env_response.observation)

                print(f"\n--- ENV STEP {steps_used}: {action.name} ---")
                if action.name != RESPOND_ACTION_NAME:
                    print(f"  Tool result: {env_obs[:500]}")
                    conversation_history.append(
                        {"role": "agent", "content": f"Tool call: {action.name}({json.dumps(action.kwargs)})"}
                    )
                    conversation_history.append(
                        {"role": "tool_result", "content": env_obs}
                    )
                    messages.append(
                        {"role": "user", "content": f"Tool result ({action.name}): {env_obs}"}
                    )
                else:
                    print(f"  Agent says: {action.kwargs.get('content', response_text)[:500]}")
                    print(f"  Customer says: {env_obs[:500]}")
                    conversation_history.append(
                        {"role": "agent", "content": action.kwargs.get("content", response_text)}
                    )
                    conversation_history.append(
                        {"role": "customer", "content": env_obs}
                    )
                    messages.append(
                        {"role": "user", "content": env_obs}
                    )

                print(f"  done={env_response.done}, reward={env_response.reward}")

                if env_response.done:
                    done = True
                    break

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )
