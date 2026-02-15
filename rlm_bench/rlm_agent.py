import json
import re
from typing import List, Optional, Dict, Any

from rlm import RLM

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import Action, SolveResult, RESPOND_ACTION_NAME

from rlm_bench.prompt_builder import PromptBuilder


class RLMAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        max_depth: int = 2,
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
        )

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

        # Fallback: treat entire response as a customer-facing message
        return [Action(
            name=RESPOND_ACTION_NAME,
            kwargs={"content": text},
        )]

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        steps_used = 0

        conversation_history: List[Dict[str, str]] = [
            {"role": "customer", "content": obs},
        ]
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]

        done = False
        while steps_used < max_num_steps and not done:
            prompt = self.prompt_builder.build_prompt(conversation_history)
            result = self.rlm.completion(prompt)
            response_text = result.response

            actions = self._parse_actions(response_text)
            messages.append({"role": "assistant", "content": response_text})

            # Execute all actions from this RLM call
            for action in actions:
                if steps_used >= max_num_steps:
                    break

                env_response = env.step(action)
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}
                steps_used += 1

                if action.name != RESPOND_ACTION_NAME:
                    conversation_history.append(
                        {"role": "agent", "content": f"Tool call: {action.name}({json.dumps(action.kwargs)})"}
                    )
                    conversation_history.append(
                        {"role": "tool_result", "content": env_response.observation}
                    )
                    messages.append(
                        {"role": "user", "content": f"Tool result ({action.name}): {env_response.observation}"}
                    )
                else:
                    conversation_history.append(
                        {"role": "agent", "content": action.kwargs.get("content", response_text)}
                    )
                    conversation_history.append(
                        {"role": "customer", "content": env_response.observation}
                    )
                    messages.append(
                        {"role": "user", "content": env_response.observation}
                    )

                if env_response.done:
                    done = True
                    break

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )
