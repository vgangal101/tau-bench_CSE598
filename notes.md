# RLM Agent Fix Notes — Feb 18, 2026

## Problem

Running Qwen3-32B (via OpenRouter) on the airline benchmark produced **0% reward on all 3 test tasks**. The agent called `transfer_to_human_agents` on the very first turn every time, ending the conversation instantly.

Log file: `results/rlm-qwen3-32b-0.0_range_0--1_user-qwen-qwen3-32b-llm_0218103350.log`

### What the model was doing wrong

On every task, the model hallucinated prior failed interaction — e.g., "Customer has not provided their user ID despite multiple requests" — even though there was only 1 customer message. It then pattern-matched the stall-escape instruction in the prompt and called `transfer_to_human_agents`, which is a **terminate tool** (`done=True`, `reward=0`).

### Root causes identified

1. **Stall-escape instruction** in `prompt_builder.py` gave the model a template for giving up:
   ```
   If the conversation has stalled ... respond with:
   [{"name": "respond", "kwargs": {"content": "Let me transfer you to a specialist."}}]
   ```
2. **No guardrails** on `transfer_to_human_agents` — the model could call it freely with no preconditions.
3. **No mechanism** to prevent premature termination on turn 1.

### Important constraint

Some tasks genuinely require `transfer_to_human_agents` as the correct action (e.g., partially-used reservations, insurance refunds) — but only **after** looking up the user/reservation first. So we delay, not block.

---

## Fixes Implemented (6 changes across 3 files)

### Two-level defense strategy

- **Prompt-level** (soft) — discourage the behavior through instructions. The model *can* still ignore these.
- **Code-level** (hard) — intercept and replace terminate calls if `steps_used < 3`. The model *cannot* bypass this.

---

### File 1: `rlm_bench/rlm_agent.py`

#### Change 1: `MIN_STEPS_BEFORE_TERMINATE` constant
```python
MIN_STEPS_BEFORE_TERMINATE = 3
```
Minimum number of `env.step()` calls before allowing terminate tools. Ensures the agent must do some actual work (ask for ID, look up account) before it can escalate.

#### Change 2: `terminate_tools` parameter in `__init__`
```python
def __init__(self, ..., terminate_tools: Optional[List[str]] = None):
    self.terminate_tools = terminate_tools or []
```
Accepts the list of terminate tool names from the environment (passed in from `run.py`).

#### Change 3: `_intercept_premature_terminate()` method
```python
def _intercept_premature_terminate(self, actions: List[Action], steps_used: int) -> List[Action]:
    if steps_used >= MIN_STEPS_BEFORE_TERMINATE:
        return actions
    intercepted = []
    for action in actions:
        if action.name in self.terminate_tools:
            intercepted.append(Action(
                name=RESPOND_ACTION_NAME,
                kwargs={"content": "I'd be happy to help you with that. "
                        "Could you please provide me with your user ID "
                        "so I can look into this for you?"},
            ))
        else:
            intercepted.append(action)
    return intercepted
```
Replaces any terminate-tool call with a polite "ask for user ID" response when `steps_used < 3`.

#### Change 4: Wired into `solve()` loop
```python
actions = self._parse_actions(response_text)
actions = self._intercept_premature_terminate(actions, steps_used)  # <-- new line
```
Called right after parsing, before loop detection and execution.

---

### File 2: `rlm_bench/prompt_builder.py`

#### Change 5a: Strengthened `transfer_to_human_agents` tool description
In `_format_tools()`, override the tool description:
```python
if name == "transfer_to_human_agents":
    desc = (
        "LAST RESORT ONLY. Transfer to a human agent. "
        "You must NEVER call this unless: (a) the customer explicitly asks "
        "for a human, OR (b) you have already looked up the account and "
        "confirmed the issue cannot be resolved with available tools."
    )
```
Makes the tool itself carry its usage constraints in the tool list.

#### Change 5b: First-turn guidance
In `build_prompt()`, when `len(conversation_history) <= 2`:
```python
sections.append("\n# IMPORTANT: This is the START of the conversation")
sections.append(
    "The customer just reached out. You have NOT asked them anything yet.\n"
    "Your action: respond and ask for their user ID so you can look up their account.\n"
    "Do NOT call any tools yet (you have no user ID to look up).\n"
    "Do NOT transfer to a human agent."
)
```
Directly combats the hallucinated-history problem by explicitly stating no prior interaction exists.

#### Change 5c: Removed stall-escape instruction
Deleted these two lines from `## Critical Prohibitions`:
```
- If the conversation has stalled and you cannot make progress, respond with:
  [{"name": "respond", "kwargs": {"content": "Let me transfer you to a specialist."}}]
```
This was actively harmful — the model pattern-matched it as permission to give up. The existing `MAX_CONSECUTIVE_RESPONDS=4` loop detection already handles real stalls.

#### Change 5d: Added anti-transfer prohibition
Added to `## Critical Prohibitions`:
```
- Do NOT call transfer_to_human_agents unless the customer explicitly requests a human agent.
  You MUST first: ask for their user ID, look up their account, and attempt to resolve the issue.
```

---

### File 3: `rlm_bench/run.py`

#### Change 6: Pass `terminate_tools` to agent
```python
agent = RLMAgent(
    ...
    terminate_tools=getattr(env, "terminate_tools", []),
)
```
Connects the environment's knowledge of terminal actions to the agent's runtime guardrails. Without this, `self.terminate_tools` would always be `[]` and the interception guard would never fire.

---

## Key tau-bench Architecture Notes

- **`env.terminate_tools`** = `["transfer_to_human_agents"]` for both airline and retail environments
- Calling a terminate tool triggers `done=True` in `env.step()`, ending the episode immediately
- **Reward calculation** compares DB state changes and expected output strings — `transfer_to_human_agents` on turn 1 means no DB changes were made, so reward = 0.0
- The RLM library wraps the LLM in a Python REPL sandbox. The prompt becomes a `context` variable; the model reads it via `print(context)` and returns actions via `FINAL([json_array])`

## Pre-existing defenses (added in earlier sessions)

These were already in the code before this fix session:

| Defense | Location | Purpose |
|---------|----------|---------|
| `AGENT_SYSTEM_PROMPT` | `rlm_agent.py` | Replaces RLM's default document-analysis prompt with agent-oriented instructions |
| `ROOT_PROMPT` | `rlm_agent.py` | Re-injected every REPL iteration to keep model focused on JSON output |
| `CORRECTIVE_SUFFIX` | `rlm_agent.py` | Appended when model outputs prose instead of JSON, triggers retry |
| `_sanitize_text()` | `rlm_agent.py` | Strips litellm/backend artifacts from responses |
| `_is_repl_error()` | `rlm_agent.py` | Detects REPL internal errors, retries up to `MAX_RLM_RETRIES=2` |
| `MAX_CONSECUTIVE_RESPONDS=4` | `rlm_agent.py` | Loop detection — aborts after 4 consecutive respond-only turns |
| `MAX_HISTORY_ENTRIES=30` | `prompt_builder.py` | Trims conversation history to prevent prompt bloat |
| `Tee` class | `run.py` | Mirrors stdout to `.log` file for debugging |

## Verification

Re-run the same 3 airline tasks:
```bash
python -m rlm_bench.run \
  --model qwen/qwen3-32b \
  --model-provider openrouter \
  --env airline \
  --task-ids 0 1 2
```

**Expected behavior after fix:**
- Task 0: Agent asks for user ID → looks up account → attempts booking
- Task 1: Agent asks for user ID → looks up reservations → attempts modification
- Task 2: Agent asks for user ID → looks up reservations → attempts resolution

(No more instant `transfer_to_human_agents` on turn 1.)
