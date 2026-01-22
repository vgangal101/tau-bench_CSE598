# Copyright Sierra

import argparse
from tau_bench.types import RunConfig
from tau_bench.run import run
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        help="The model provider for the agent (openai, anthropic, dashscope, openrouter, local, etc.)",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="openai/gpt-oss-20b",
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="tool-calling",
        choices=["tool-calling", "act", "react", "few-shot"],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The sampling temperature for the action model",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="The split of tasks to run (only applies to the retail domain for now",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument("--task-ids", type=int, nargs="+", help="(Optional) run only the tasks with the given IDs")
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])
    parser.add_argument("--few-shot-displays-path", type=str, help="Path to a jsonlines file containing few shot displays")
    # New arguments for custom provider endpoints
    parser.add_argument(
        "--model-base-url",
        type=str,
        default=None,
        help="Custom base URL for agent model API (for DashScope, OpenRouter, or local servers)",
    )
    parser.add_argument(
        "--model-api-key",
        type=str,
        default=None,
        help="Override API key for agent model (useful for local deployments)",
    )
    parser.add_argument(
        "--user-model-base-url",
        type=str,
        default=None,
        help="Custom base URL for user model API",
    )
    parser.add_argument(
        "--user-model-api-key",
        type=str,
        default=None,
        help="Override API key for user model (useful for local deployments)",
    )
    parser.add_argument(
        "--dashscope-region",
        type=str,
        default="singapore",
        choices=["singapore", "us"],
        help="DashScope region (singapore or us)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens for agent model responses (default: 1000)",
    )
    parser.add_argument(
        "--user-max-tokens",
        type=int,
        default=500,
        help="Maximum tokens for user simulator responses (default: 500)",
    )
    args = parser.parse_args()
    print(args)
    return RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
        model_base_url=args.model_base_url,
        model_api_key=args.model_api_key,
        user_model_base_url=args.user_model_base_url,
        user_model_api_key=args.user_model_api_key,
        dashscope_region=args.dashscope_region,
        max_tokens=args.max_tokens,
        user_max_tokens=args.user_max_tokens,
    )


def main():
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
