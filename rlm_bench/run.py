import os
import json
import random
import argparse
import traceback
import multiprocessing
from typing import List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tau_bench.envs import get_env
from tau_bench.types import EnvRunResult
from tau_bench.run import display_metrics
from tau_bench.envs.user import UserStrategy

from rlm_bench.config import RLMRunConfig
from rlm_bench.rlm_agent import RLMAgent


def run(config: RLMRunConfig) -> List[EnvRunResult]:
    assert config.env in ["retail", "airline"], "Only retail and airline envs are supported"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"
    assert config.user_strategy in [item.value for item in UserStrategy], "Invalid user strategy"

    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    user_model_safe = config.user_model.replace("/", "-")
    ckpt_path = (
        f"{config.log_dir}/rlm-{config.model.split('/')[-1]}-{config.temperature}"
        f"_range_{config.start_index}-{config.end_index}"
        f"_user-{user_model_safe}-{config.user_strategy}_{time_str}.json"
    )
    os.makedirs(config.log_dir, exist_ok=True)

    print(f"Loading user with strategy: {config.user_strategy}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
        user_api_base=config.user_model_base_url,
    )

    agent = RLMAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=config.model,
        provider=config.model_provider,
        temperature=config.temperature,
        max_depth=config.rlm_max_depth,
        environment=config.rlm_environment,
        model_base_url=config.model_base_url,
    )

    end_index = (
        len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
    )
    results: List[EnvRunResult] = []
    lock = multiprocessing.Lock()

    if config.task_ids and len(config.task_ids) > 0:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})")

    for i in range(config.num_trials):
        if config.task_ids and len(config.task_ids) > 0:
            idxs = config.task_ids
        else:
            idxs = list(range(config.start_index, end_index))
        if config.shuffle:
            random.shuffle(idxs)

        def _run(idx: int) -> EnvRunResult:
            isolated_env = get_env(
                config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
                user_provider=config.user_model_provider,
                task_index=idx,
                user_api_base=config.user_model_base_url,
            )

            print(f"Running task {idx}")
            try:
                res = agent.solve(env=isolated_env, task_index=idx)
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                )
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                )
            print(
                "\u2705" if result.reward == 1 else "\u274c",
                f"task_id={idx}",
                result.info,
            )
            print("-----")
            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r") as f:
                        data = json.load(f)
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result.model_dump()], f, indent=2)
            return result

        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            res = list(executor.map(_run, idxs))
            results.extend(res)

    display_metrics(results)

    with open(ckpt_path, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n\U0001f4c4 Results saved to {ckpt_path}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run tau-bench with RLM agent")

    # Standard tau-bench args
    parser.add_argument("--env", type=str, default="retail", choices=["retail", "airline"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-provider", type=str, required=True)
    parser.add_argument("--user-model", type=str, default="gpt-4o")
    parser.add_argument("--user-model-provider", type=str, default="openai")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--task-split", type=str, default="test", choices=["train", "test", "dev"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--task-ids", nargs="+", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--user-strategy", type=str, default="llm")
    parser.add_argument("--model-base-url", type=str, default=None)
    parser.add_argument("--user-model-base-url", type=str, default=None)

    # RLM-specific args
    parser.add_argument("--rlm-max-depth", type=int, default=2)
    parser.add_argument("--rlm-environment", type=str, default="local", choices=["local", "docker", "modal"])

    args = parser.parse_args()

    config = RLMRunConfig(
        model=args.model,
        model_provider=args.model_provider,
        user_model=args.user_model,
        user_model_provider=args.user_model_provider,
        temperature=args.temperature,
        num_trials=args.num_trials,
        env=args.env,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        model_base_url=args.model_base_url,
        user_model_base_url=args.user_model_base_url,
        rlm_max_depth=args.rlm_max_depth,
        rlm_environment=args.rlm_environment,
    )

    run(config)


if __name__ == "__main__":
    main()
