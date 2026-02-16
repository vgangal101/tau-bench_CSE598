# RLM Agent Data Flow — Batched Tool Calling

## RLM Agent Full Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          rlm_bench/run.py                                   │
│                                                                             │
│  CLI Args ──► RLMRunConfig ──► run(config)                                  │
│                                                                             │
│  ┌─────────────────────┐    ┌──────────────────────┐                        │
│  │ get_env("retail")   │    │ RLMAgent(            │                        │
│  │  ├─ 16 tools        │    │   backend=openrouter │                        │
│  │  ├─ wiki (policies) │    │   model=qwen3-8b     │                        │
│  │  └─ user simulator  │    │   max_depth=2        │                        │
│  │    (Qwen3-32B)      │    │ )                    │                        │
│  └─────────┬───────────┘    └──────────┬───────────┘                        │
│            │                           │                                    │
│            ▼                           ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    agent.solve(env, task_index)                     │    │
│  │                                                                     │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │ Step 0: env.reset(task_index)                                 │   │    │
│  │  │                                                               │   │    │
│  │  │  Task Instruction ──► User Simulator LLM ──► Initial Message  │   │    │
│  │  │        (hidden)        (Qwen3-32B)         "Hi, I need to     │   │    │
│  │  │                                             change my order"  │   │    │
│  │  └──────────────────────────┬───────────────────────────────────┘   │    │
│  │                             │                                       │    │
│  │                             ▼                                       │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │     │
│  │  │ MAIN LOOP (while steps < 30 and not done)                    │   │     │
│  │  │                                                               │   │     │
│  │  │  ┌────────────────────────────────────────────────────────┐   │   │     │
│  │  │  │ 1. PromptBuilder.build_prompt(conversation_history)    │   │   │     │
│  │  │  │                                                        │   │   │     │
│  │  │  │  ┌──────────────────────────────────────────────────┐  │   │   │     │
│  │  │  │  │ # Policy and Domain Knowledge                    │  │   │   │     │
│  │  │  │  │ <retail wiki: return policies, order rules...>   │  │   │   │     │
│  │  │  │  │                                                  │  │   │   │     │
│  │  │  │  │ # Available Tools                                │  │   │   │     │
│  │  │  │  │ - get_user_details(user_id)                      │  │   │   │     │
│  │  │  │  │ - get_order_details(order_id)                    │  │   │   │     │
│  │  │  │  │ - modify_pending_order(...)                      │  │   │   │     │
│  │  │  │  │ - ... (16 tools as text descriptions)            │  │   │   │     │
│  │  │  │  │                                                  │  │   │   │     │
│  │  │  │  │ # Conversation History                           │  │   │   │     │
│  │  │  │  │ [customer]: Hi, I need to change my order        │  │   │   │     │
│  │  │  │  │ [agent]: Tool call: get_user(...)                │  │   │   │     │
│  │  │  │  │ [tool_result]: {user data}                       │  │   │   │     │
│  │  │  │  │ ...                                              │  │   │   │     │
│  │  │  │  │                                                  │  │   │   │     │
│  │  │  │  │ # Instructions                                   │  │   │   │     │
│  │  │  │  │ ★ BATCH multiple tool calls in one turn! ★       │  │   │   │     │
│  │  │  │  │ Output a JSON array of actions.                  │  │   │   │     │
│  │  │  │  └──────────────────────────────────────────────────┘  │   │   │     │
│  │  │  └──────────────────────────┬─────────────────────────────┘   │   │     │
│  │  │                             │ prompt (single string)          │   │     │
│  │  │                             ▼                                 │   │     │
│  │  │  ┌────────────────────────────────────────────────────────┐   │   │     │
│  │  │  │ 2. rlm.completion(prompt)                              │   │   │     │
│  │  │  │                                                        │   │   │     │
│  │  │  │  ┌──────────── RLM REPL Sandbox ────────────────────┐  │   │   │     │
│  │  │  │  │                                                  │  │   │   │     │
│  │  │  │  │  Iteration 1:                                    │  │   │   │     │
│  │  │  │  │    LLM generates Python code ◄── Qwen3-8B        │  │   │   │     │
│  │  │  │  │    ```python                     (OpenRouter)    │  │   │   │     │
│  │  │  │  │    # I need user + order info                    │  │   │   │     │
│  │  │  │  │    user = llm_query("extract user_id")           │  │   │   │     │
│  │  │  │  │    order = llm_query("extract order_id")         │  │   │   │     │
│  │  │  │  │    FINAL_VAR(json.dumps([                        │  │   │   │     │
│  │  │  │  │      {"name":"get_user","kwargs":{...}},         │  │   │   │     │
│  │  │  │  │      {"name":"get_order","kwargs":{...}}         │  │   │   │     │
│  │  │  │  │    ]))                                           │  │   │   │     │
│  │  │  │  │    ```                                           │  │   │   │     │
│  │  │  │  │         │                                        │  │   │   │     │
│  │  │  │  │         ▼                                        │  │   │   │     │
│  │  │  │  │    Code executes in sandboxed REPL               │  │   │   │     │
│  │  │  │  │    llm_query() = recursive LLM sub-calls         │  │   │   │     │
│  │  │  │  │    (up to max_depth=2 levels deep)               │  │   │   │     │
│  │  │  │  │         │                                        │  │   │   │     │
│  │  │  │  │         ▼                                        │  │   │   │     │
│  │  │  │  │    FINAL_VAR() ──► result.response               │  │   │   │     │
│  │  │  │  │                                                  │  │   │   │     │
│  │  │  │  └──────────────────────┬───────────────────────────┘  │   │   │     │
│  │  │  │                         │                              │   │   │     │
│  │  │  │  result.response = '[{"name":"get_user",...},          │   │   │     │
│  │  │  │                       {"name":"get_order",...}]'       │   │   │     │
│  │  │  └─────────────────────────┬──────────────────────────────┘   │   │     │
│  │  │                            │                                  │   │     │
│  │  │                            ▼                                  │   │     │
│  │  │  ┌────────────────────────────────────────────────────────┐   │   │     │
│  │  │  │ 3. _parse_actions(response_text)                       │   │   │     │
│  │  │  │                                                        │   │   │     │
│  │  │  │  JSON array ──► [Action(get_user), Action(get_order)]  │   │   │     │
│  │  │  │                  ▲                                     │   │   │     │
│  │  │  │                  │ Parsing priority:                   │   │   │     │
│  │  │  │                  │ 1. Full JSON array                  │   │   │     │
│  │  │  │                  │ 2. Embedded JSON array (regex)      │   │   │     │
│  │  │  │                  │ 3. Single JSON object               │   │   │     │
│  │  │  │                  │ 4. Fallback → respond action        │   │   │     │
│  │  │  └─────────────────────────┬──────────────────────────────┘   │   │     │
│  │  │                            │                                  │   │     │
│  │  │                            ▼                                  │   │     │
│  │  │  ┌────────────────────────────────────────────────────────┐   │   │     │
│  │  │  │ 4. Execute ALL actions from this batch                 │   │   │     │
│  │  │  │                                                        │   │   │     │
│  │  │  │  Action 1: get_user_details(user_id="u123")            │   │   │     │
│  │  │  │       │                                                │   │   │     │
│  │  │  │       ▼                                                │   │   │     │
│  │  │  │  env.step(action) ──► Mock Retail DB ──► user data     │   │   │     │
│  │  │  │       │                                                │   │   │     │
│  │  │  │       ▼ append to history                              │   │   │     │
│  │  │  │                                                        │   │   │     │
│  │  │  │  Action 2: get_order_details(order_id="o456")          │   │   │     │
│  │  │  │       │                                                │   │   │     │
│  │  │  │       ▼                                                │   │   │     │
│  │  │  │  env.step(action) ──► Mock Retail DB ──► order data    │   │   │     │
│  │  │  │       │                                                │   │   │     │
│  │  │  │       ▼ append to history                              │   │   │     │
│  │  │  │                                                        │   │   │     │
│  │  │  │  (both results now in conversation_history)            │   │   │     │
│  │  │  └─────────────────────────┬──────────────────────────────┘   │   │     │
│  │  │                            │                                  │   │     │
│  │  │                            ▼                                  │   │     │
│  │  │                   Loop back to step 1                         │   │     │
│  │  │                   (next RLM call sees ALL results)            │   │     │
│  │  │                            .                                  │   │     │
│  │  │                            .                                  │   │     │
│  │  │                            .                                  │   │     │
│  │  │  ┌────────────────────────────────────────────────────────┐   │   │     │
│  │  │  │ Final turn: RLM returns respond action                 │   │   │     │
│  │  │  │                                                        │   │   │     │
│  │  │  │  [{"name":"respond","kwargs":{"content":"Done!..."}}]  │   │   │     │
│  │  │  │       │                                                │   │   │     │
│  │  │  │       ▼                                                │   │   │     │
│  │  │  │  env.step(respond) ──► User Simulator ──► "###STOP###" │   │   │     │
│  │  │  │                        (Qwen3-32B)        done=True    │   │   │     │
│  │  │  └────────────────────────────────────────────────────────┘   │   │     │
│  │  │                                                               │   │     │
│  │  └───────────────────────────────────────────────────────────────┘   │     │
│  │                             │                                       │     │
│  │                             ▼                                       │     │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │ calculate_reward()                                           │   │    │
│  │  │  Compare DB state after agent actions vs ground truth        │   │    │
│  │  │  reward = 1.0 (correct) or 0.0 (incorrect)                   │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Results ──► checkpoint JSON ──► display_metrics() ──► Pass^k score         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Batched vs Non-Batched Comparison

```
NON-BATCHED (standard agents):          BATCHED (RLM agent):
─────────────────────────────           ─────────────────────────────

RLM call 1 ──► get_user                RLM call 1 ──► get_user
   └─ result                                        ├─ get_order    ← same turn
                                                     └─ results
RLM call 2 ──► get_order
   └─ result
                                        RLM call 2 ──► modify_order
RLM call 3 ──► modify_order                          └─ result
   └─ result
                                        RLM call 3 ──► respond
RLM call 4 ──► respond                     └─ done
   └─ done

4 RLM calls                             3 RLM calls
(4 REPL sessions x N sub-LLM calls)    (3 REPL sessions x N sub-LLM calls)
```

## How Standard Tau-Bench Differs from RLM

```
STANDARD (tool-calling agent):

  prompt ──► litellm.completion() ──► response with tool_calls
              (1 LLM call per step)     (native function calling)


RLM (batched):

  prompt ──► rlm.completion() ──────► response string (JSON array)
              │                         (parsed via _parse_actions)
              │
              └── internally runs a REPL:
                  LLM writes Python code
                  code can call llm_query() recursively
                  loops until FINAL_VAR() produces answer
                  (1+ LLM calls per step, up to depth=2)
```

**What stays identical**: env, tools, user simulator, grading, task definitions, results format

**What changes**: The single `litellm.completion()` call is replaced by `rlm.completion()`, which lets the model reason in a Python REPL with recursive sub-calls before producing its final action(s). Actions are parsed from text (since RLM returns a string, not structured tool_calls), and multiple actions can be batched in a single turn.

## Component Ownership

```
┌──────────────────────────────────────────────────────────────────┐
│                        rlm_bench/ (NEW)                          │
│  ┌────────────┐  ┌───────────────┐  ┌────────────────────────┐   │
│  │ run.py     │  │ prompt_builder│  │ rlm_agent.py           │   │
│  │ CLI entry  │  │ .py           │  │ _parse_actions()       │   │
│  │ point      │  │ builds text   │  │ solve() with batching  │   │
│  │            │  │ prompts with  │  │                        │   │
│  │            │  │ batch instrs  │  │ Uses: rlm.completion() │   │
│  └─────┬──────┘  └───────────────┘  └───────────┬────────────┘   │
│        │                                        │                │
├────────┼────────────────────────────────────────┼────────────────┤
│        │          tau_bench/ (UNTOUCHED)        │              │
│        ▼                                         ▼               │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌────────┐          │
│  │ get_env()│  │ user.py  │  │ tools     │  │ types  │          │
│  │ envs/    │  │ simulator│  │ (retail   │  │ Action │          │
│  │          │  │ (litellm)│  │  DB ops)  │  │ Solve  │          │
│  └──────────┘  └──────────┘  └───────────┘  │ Result │          │
│                                             └────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                     External Services                           │
│  ┌──────────────────────┐    ┌──────────────────────┐           │
│  │ OpenRouter           │    │ OpenRouter           │           │
│  │ Qwen3-8B (agent)     │    │ Qwen3-32B (user sim) │           │
│  │ via RLM's openai     │    │ via litellm          │           │
│  │ client               │    │                      │           │
│  └──────────────────────┘    └──────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Changelog

### v2 — REPL Error Filtering + Prompt Overhaul (2026-02-15)

**Problem**: First test run (3 tasks, 0/3 passed) revealed two critical issues:

1. **RLM REPL errors leaked to the customer** — When the model's Python code in the
   REPL referenced undefined variables, `FINAL_VAR()` failed and the error string
   (e.g., `"Error: Variable 'exchange_initiated' not found"`) became `result.response`,
   which got parsed as a `respond` action and sent directly to the user simulator.

2. **Agent never executed mutation tools** — The agent gathered all necessary data
   (user details, order details, product variants) but never called
   `exchange_delivered_order_items` or `return_delivered_order_items`. It would
   describe what it *would* do instead of actually doing it.

**Fix 1 — REPL error retry** (`rlm_agent.py`):
- Added `_is_repl_error()` that detects known REPL error patterns:
  - `"Error: Variable"`
  - `"not found. Available variables:"`
  - `"You must create and assign a variable BEFORE calling FINAL_VAR"`
- Added retry loop: on REPL error, retry `rlm.completion()` up to 2 more times
- If all 3 attempts fail, the turn is **skipped** (no action sent to env)

```
rlm.completion(prompt)
       │
       ▼
  REPL error? ──yes──► retry (up to 2x) ──still error──► skip turn
       │
       no
       ▼
  _parse_actions() ──► env.step()
```

**Fix 2 — Stronger prompt** (`prompt_builder.py`):
- Rewritten instructions section with numbered rules:
  - Rule 1: Batch independent tool calls (with concrete example)
  - Rule 2: Single tool call format
  - Rule 3: Respond only when you have all info
  - Rule 4: **"EXECUTE mutation tools when you have the parameters.
    Do NOT just describe what you would do — actually call the tool."**
  - Rule 5: **"When you get order IDs, batch-fetch ALL order details in one turn."**
- Ends with: "Output ONLY the JSON array. No explanations, no markdown, no commentary."

### v1 — Batched Tool Calling (2026-02-15)

**Changes from initial single-action implementation:**

- `_parse_action()` renamed to `_parse_actions()`, now returns `List[Action]`
- Parses JSON arrays first (batched), falls back to single JSON objects
- `solve()` loop executes all actions from one RLM call before making the next
- Prompt instructs model to batch independent tool calls in one turn
- Added verbose logging: prints prompt, RLM response, parsed actions, env results

### v0 — Initial Implementation (2026-02-14)

- `rlm_bench/__init__.py` — Package init, exports RLMAgent
- `rlm_bench/config.py` — RLMRunConfig extending RunConfig (adds rlm_max_depth, rlm_environment)
- `rlm_bench/prompt_builder.py` — Formats wiki + tools + history into text prompt
- `rlm_bench/rlm_agent.py` — RLMAgent implementing Agent.solve() with rlm.completion()
- `rlm_bench/run.py` — Standalone CLI entry point mirroring tau_bench/run.py
- Zero modifications to existing tau_bench/ files
