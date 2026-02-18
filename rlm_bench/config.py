from tau_bench.types import RunConfig
from typing import Optional


class RLMRunConfig(RunConfig):
    agent_strategy: str = "rlm"
    rlm_max_depth: int = 1
    rlm_environment: str = "local"
