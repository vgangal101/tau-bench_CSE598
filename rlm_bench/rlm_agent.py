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

    def _parse_action(self, response_text: str) -> Action:
        # Try to find a JSON object in the response
        # First try: find JSON between braces
        match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"kwargs"\s*:\s*\{[^}]*\}[^}]*\}', response_text)
        if match:
            try:
                parsed = json.loads(match.group())
                return Action(name=parsed["name"], kwargs=parsed["kwargs"])
            except (json.JSONDecodeError, KeyError):
                pass

        # Second try: find any JSON object with "name" key
        for match in re.finditer(r'\{[^{}]*\{[^{}]*\}[^{}]*\}', response_text):
            try:
                parsed = json.loads(match.group())
                if "name" in parsed and "kwargs" in parsed:
                    return Action(name=parsed["name"], kwargs=parsed["kwargs"])
            except (json.JSONDecodeError, KeyError):
                continue

        # Third try: parse the whole response as JSON
        try:
            parsed = json.loads(response_text.strip())
            if "name" in parsed and "kwargs" in parsed:
                return Action(name=parsed["name"], kwargs=parsed["kwargs"])
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: treat entire response as a customer-facing message
        return Action(
            name=RESPOND_ACTION_NAME,
            kwargs={"content": response_text.strip()},
        )

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0

        conversation_history: List[Dict[str, str]] = [
            {"role": "customer", "content": obs},
        ]
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]

        for _ in range(max_num_steps):
            prompt = self.prompt_builder.build_prompt(conversation_history)
            result = self.rlm.completion(prompt)
            response_text = result.response

            action = self._parse_action(response_text)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            if action.name != RESPOND_ACTION_NAME:
                # Tool call
                conversation_history.append(
                    {"role": "agent", "content": f"Tool call: {action.name}({json.dumps(action.kwargs)})"}
                )
                conversation_history.append(
                    {"role": "tool_result", "content": env_response.observation}
                )
                messages.extend([
                    {"role": "assistant", "content": response_text},
                    {"role": "user", "content": f"Tool result ({action.name}): {env_response.observation}"},
                ])
            else:
                # Response to customer
                conversation_history.append(
                    {"role": "agent", "content": action.kwargs.get("content", response_text)}
                )
                conversation_history.append(
                    {"role": "customer", "content": env_response.observation}
                )
                messages.extend([
                    {"role": "assistant", "content": action.kwargs.get("content", response_text)},
                    {"role": "user", "content": env_response.observation},
                ])

            if env_response.done:
                break

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )
