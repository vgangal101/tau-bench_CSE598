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

---

## Round 2: Post-Escalation Fix Issues

After Round 1 fixes, the agent does real work but still scores 0/3. New failure modes identified from log `rlm-qwen3-32b-0.0_range_0--1_user-qwen-qwen3-32b-llm_0218111635.log`:

### New problems found

**Task 0** (book flight JFK→SEA) — Agent completed the booking but reward=0 because:
- Used `flight_type: "one way"` (space) instead of `"one_way"` (underscore) → different DB state hash
- Included extra fields in `flights[]` (origin, destination, times) — tool only needs `flight_number` and `date`
- Included extra `source` field in `payment_methods[]` — tool only needs `payment_id` and `amount`
- Wasted 2 steps on hallucinated tools: `check_order_status` (doesn't exist) and `list_all_airports` (unnecessary)
- Batched 2 respond actions in one turn (each consumed a step)

**Task 1** (cancel reservation Z7GOZK) — Agent correctly cancelled but reward=0 because:
- Called `send_certificate(user_id, amount=100)` after cancelling — this is unnecessary since `cancel_reservation` handles refunds automatically
- The extra certificate modified DB state beyond ground truth

**Task 2** (downgrade 5 reservations) — User simulator (Qwen3-32B) said `###STOP###` immediately instead of providing user ID — not an agent bug

**Prose detection false positives** — Valid JSON like `[{"name": "respond", ...}]` got flagged as prose (due to ANSI codes or markdown fences), triggering corrective prompts that produced worse responses

### Key architecture insight: reward calculation

tau-bench reward uses **SHA256 hash comparison of entire DB state** (not parameter matching). After the agent acts, it hashes the DB. Then it replays ground-truth actions on a fresh DB and hashes that. `r_actions = (agent_hash == ground_truth_hash)`. Any extra mutation (like `send_certificate`) or wrong parameter value (like `"one way"`) that changes DB state differently → reward=0.

---

## Round 2 Fixes (4 changes across 2 files)

### File 1: `rlm_bench/rlm_agent.py`

#### Change 7: Tool name validation (`valid_tool_names` + `_validate_actions`)

**Problem**: Model hallucinated tools (`check_order_status`, `reset_password`) wasting steps.

Extract valid tool names in `__init__`:
```python
self.valid_tool_names = {
    tool.get("function", tool)["name"] for tool in tools_info
}
```

New `_validate_actions()` method that filters invalid tools AND enforces single-respond-per-turn:
```python
def _validate_actions(self, actions: List[Action]) -> List[Action]:
    validated = []
    for action in actions:
        if action.name == RESPOND_ACTION_NAME or action.name in self.valid_tool_names:
            validated.append(action)
        else:
            print(f"  [INVALID TOOL] '{action.name}' not in valid tools, skipping")
    if not validated:
        validated.append(Action(name=RESPOND_ACTION_NAME,
            kwargs={"content": "Let me look into that for you."}))
    # If any action is a respond, keep only up to the first respond
    first_respond_idx = next(
        (i for i, a in enumerate(validated) if a.name == RESPOND_ACTION_NAME), None)
    if first_respond_idx is not None:
        validated = validated[:first_respond_idx + 1]
    return validated
```

Wired into `solve()` as the third step in the action pipeline:
```python
actions = self._parse_actions(response_text)
actions = self._intercept_premature_terminate(actions, steps_used)
actions = self._validate_actions(actions)  # NEW
```

This prevents:
- Wasting steps on nonexistent tools
- Batching multiple respond actions (only first respond kept)
- Mixing respond with tool calls (respond truncates the list)

#### Change 8: Fix prose detection false positives

**Problem**: `_is_valid_action_response` returned False for valid JSON wrapped in ANSI codes or markdown fences, triggering unnecessary corrective prompts that produced hallucinated tool calls.

Added pre-processing at the top of `_is_valid_action_response`:
```python
# Strip ANSI escape codes that may leak from litellm
text = re.sub(r'\x1b\[[0-9;]*m', '', text)
# Strip markdown code fences
if text.startswith("```"):
    lines = text.split("\n")
    if lines[-1].strip() == "```":
        lines = lines[1:-1]
    else:
        lines = lines[1:]
    text = "\n".join(lines).strip()
```

### File 2: `rlm_bench/prompt_builder.py`

#### Change 9: Parameter formatting rules

**Problem**: Model used wrong enum values (`"one way"` vs `"one_way"`) and included extra fields in tool kwargs.

Added `## Parameter Rules` section before `## Critical Prohibitions`:
```
## Parameter Rules

- Use ONLY the parameters listed in each tool description. Do not add extra fields.
- Use exact enum values from tool descriptions (e.g., "one_way" not "one way",
  "round_trip" not "round trip", "basic_economy" not "basic economy").
- For flights arrays, pass ONLY flight_number and date. The tool looks up the rest.
- For payment_methods, pass ONLY payment_id and amount.
- After cancelling a reservation, do NOT call send_certificate — refunds are automatic.
```

---

## Round 2 Verification

Same command as Round 1:
```bash
python -m rlm_bench.run \
  --model qwen/qwen3-32b \
  --model-provider openrouter \
  --env airline \
  --task-ids 0 1 2
```

**Expected improvements:**
- Task 0: Correct `flight_type: "one_way"`, minimal kwargs in flights/payments, no hallucinated tools → DB hash should match
- Task 1: No extra `send_certificate` after cancel → DB hash should match
- Task 2: Still depends on user simulator quality (outside our control)

---

## Round 3: Results and Analysis

Log file: `results/rlm-qwen3-32b-0.0_range_0--1_user-qwen-qwen3-32b-llm_0218123556.log`

### Score: 1/3 (33%) — up from 0/3 (0%)

**Task 0** (book flight JFK→SEA for passenger Mia Li) — reward=0.0
- Round 2 formatting fixes confirmed working: `flight_type: "one_way"`, minimal kwargs
- But model made **reasoning errors**:
  - Hallucinated passenger name "Amelia Ahmed" instead of using "Mia Li" from `get_user_details` response
  - Selected flight HAT218 ($285) instead of cheapest option HAT136
  - Used credit card instead of travel certificates (customer had certificates available)
- These are comprehension failures — the correct data was in the tool results

**Task 1** (cancel reservation Z7GOZK) — reward=1.0 ✅ FIRST WIN
- Round 2 fix directly caused this: no spurious `send_certificate` after cancel
- Agent correctly: asked for user ID → looked up user → found reservation → cancelled it
- DB state hash matched ground truth exactly

**Task 2** (downgrade 5 reservations from business to economy) — reward=0.0
- Agent performed all 5 `update_reservation_baggages` mutations correctly
- Failed on **output value**: reported savings of "$5,640" but ground truth expected "$23,553"
- Root cause: model pre-computed the savings amount in a respond action *before* executing the update tools, using wrong arithmetic
- The correct calculation requires subtracting new economy prices from original business prices — but the model used post-downgrade prices for both

### What the Round 2 fixes accomplished

All 4 fixes were confirmed working in the logs:

| Fix | Evidence |
|-----|----------|
| Tool name validation | `[INVALID TOOL] 'get_order_status'` caught and filtered |
| Single respond per turn | No more batched respond actions |
| ANSI/markdown stripping | No false positive prose detections |
| Parameter Rules | Correct enum values, minimal kwargs |

### Conclusion: Pipeline vs Model Capability

The remaining failures are **model reasoning problems**, not pipeline problems:

| Category | Example | Fixable with code? |
|----------|---------|-------------------|
| Data hallucination | "Amelia Ahmed" when API returned "Mia Li" | No — model had the data |
| Wrong selection | Picked non-cheapest flight | No — requires domain reasoning |
| Wrong payment method | Credit card instead of certificates | No — requires policy understanding |
| Pre-computed outputs | Calculated savings before tool execution | No — requires execution planning |

**Pipeline fixes** (Rounds 1-2) addressed structural issues: wrong formats, missing guards, extra mutations. These are problems where the model *would* do the right thing with better scaffolding.

**Model capability limits** (Round 3) are problems where the model has all the data but makes wrong reasoning choices. Fixing these in code would mean hardcoding domain logic that only works for specific tasks and wouldn't generalize to the full benchmark.

**Decision**: Accept current pipeline as complete. Run the full benchmark to get a statistically meaningful score. Compare across model sizes (4B, 8B, 14B, 32B) to see if reasoning improves with scale.

---

## Round 4: RLM Library Compatibility Fix

### Problem

After the RLM library (`rlm_cse598`) was updated upstream, every task fails with:
```
KeyError: '"name"'
```

Traceback:
```
rlm/utils/prompts.py, line 156, in build_rlm_system_prompt
    final_system_prompt = system_prompt.format(custom_tools_section=custom_tools_section)
```

### Root cause

The updated RLM library now calls `system_prompt.format(custom_tools_section=...)` on the custom system prompt. Python's `str.format()` treats `{` and `}` as format placeholders. Our `AGENT_SYSTEM_PROMPT` contained JSON examples with literal curly braces:
```python
'  FINAL([{"name": "get_user_details", "kwargs": {"user_id": "sara_doe_496"}}])\n'
```

Python interprets `{"name"` as a format field lookup for key `"name"` → `KeyError`.

### Fix (Change 10): Escape curly braces in `AGENT_SYSTEM_PROMPT`

**File**: `rlm_bench/rlm_agent.py`

Doubled all literal curly braces in the JSON example lines (lines 41-42):
```python
# Before:
'  FINAL([{"name": "get_user_details", "kwargs": {"user_id": "sara_doe_496"}}])\n'
'  FINAL([{"name": "respond", "kwargs": {"content": "your message"}}])\n\n'

# After:
'  FINAL([{{"name": "get_user_details", "kwargs": {{"user_id": "sara_doe_496"}}}}])\n'
'  FINAL([{{"name": "respond", "kwargs": {{"content": "your message"}}}}])\n\n'
```

`{{` and `}}` are Python's escape sequences for literal `{` and `}` inside `.format()` strings. After `.format()` processes the string, the model sees the correct single braces.

Note: `CORRECTIVE_SUFFIX` and `ROOT_PROMPT` were NOT changed — they are part of the `prompt` (context), not the `system_prompt`, so they don't go through `.format()`.

### Fix (Change 11): Remove `sara_doe_496` from `AGENT_SYSTEM_PROMPT` examples

**Problem**: In Round 4 run, the model called `get_user_details("sara_doe_496")` as its FIRST action on 2/3 tasks — copying the example user ID from `AGENT_SYSTEM_PROMPT` instead of waiting for the customer to provide their ID. This wasted a step and sometimes triggered `transfer_to_human_agents` after the "user not found" error.

**File**: `rlm_bench/rlm_agent.py`

Replaced concrete `get_user_details` example with generic placeholders:
```python
# Before:
'  FINAL([{{"name": "get_user_details", "kwargs": {{"user_id": "sara_doe_496"}}}}])\n'
'  FINAL([{{"name": "respond", "kwargs": {{"content": "your message"}}}}])\n\n'

# After:
'  FINAL([{{"name": "respond", "kwargs": {{"content": "How can I help you today?"}}}}])\n'
'  FINAL([{{"name": "TOOL_NAME", "kwargs": {{"param": "value"}}}}])\n\n'
```

The `respond` example now leads with asking the customer (the correct first action). The tool call example uses a generic placeholder so the model won't copy a specific tool name or user ID.

---

# Size-Aware RLM Agent with `llm_query()` Chunking — Feb 19, 2026

## Problem

The RLM agent wastes the framework's recursive capabilities. Every turn, the model does `print(context)` to read the entire prompt, then outputs `FINAL([...])`. As conversations progress (turn 10+), the context can exceed 20K chars. The RLM library's `format_iteration()` (`parsing.py:67`) truncates REPL output to 20K chars in message history, meaning the model can't see the full context on later turns.

Additionally, `llm_query()` was explicitly prohibited in `AGENT_SYSTEM_PROMPT` despite being fully functional in the REPL sandbox (`local_repl.py:165`). Using it to extract relevant policy rules from long contexts should produce better tool-calling decisions.

## Changes (3 files)

### Change 12: Add section markers to `prompt_builder.py`

**File**: `rlm_bench/prompt_builder.py`

Wrapped each section of the built prompt in `<<<SECTION_NAME>>>` / `<<<END_SECTION_NAME>>>` delimiters so the model can extract sections programmatically with `str.find()` in the REPL.

Markers added: `<<<TASK>>>`, `<<<WIKI>>>`, `<<<TOOLS>>>`, `<<<HISTORY>>>`, `<<<INSTRUCTIONS>>>` (and corresponding `<<<END_*>>>` markers).

```python
# Before:
sections.append("# YOUR TASK")
sections.append(...)
sections.append("\n# Policy and Domain Knowledge")
sections.append(self.wiki)

# After:
sections.append("<<<TASK>>>")
sections.append("# YOUR TASK")
sections.append(...)
sections.append("<<<END_TASK>>>")
sections.append("\n<<<WIKI>>>")
sections.append("# Policy and Domain Knowledge")
sections.append(self.wiki)
sections.append("<<<END_WIKI>>>")
```

This is backward-compatible — markers are just extra text in the string. Existing prompts work identically.

### Change 13: Two-workflow `AGENT_SYSTEM_PROMPT` in `rlm_agent.py`

**File**: `rlm_bench/rlm_agent.py`

Added `CONTEXT_SIZE_THRESHOLD = 20000` constant.

Replaced the single-workflow system prompt with a two-workflow design:

- **Step 1**: Always start by checking `len(context)` in a REPL block
- **Workflow A** (under 20K chars): `print(context)` then `FINAL()` — same as before
- **Workflow B** (20K+ chars): Extract sections via `str.find()`, use `llm_query()` to summarize wiki + older history, keep tools + recent 5 turns visible, then `FINAL()`

Key changes:
- Removed `llm_query` prohibition (was: "Do NOT use llm_query or llm_query_batched")
- Added explicit size check as the first required step
- Added "Do NOT use llm_query when context is under 20000 chars" to prevent unnecessary sub-calls on small contexts
- Used `.replace("SIZE_THRESHOLD", _THRESHOLD_STR)` instead of `.format()` to avoid conflicts with the RLM library's own `.format()` pass on the system prompt

### Change 14: Updated `ROOT_PROMPT` in `rlm_agent.py`

**File**: `rlm_bench/rlm_agent.py`

Updated from:
```python
ROOT_PROMPT = "Read the context and output FINAL([json_array]) with your action. ..."
```

To:
```python
ROOT_PROMPT = "Check the context size, follow the appropriate workflow (A for small, B for large), and output FINAL([json_array]) with your action. ..."
```

This reinforces the size-check-first behavior at every REPL iteration.

## Why `.replace()` instead of `.format()`

The RLM library's `build_rlm_system_prompt()` concatenates our system prompt into a template and calls `.format()` on it. Using our own `.format(threshold=...)` would strip the double braces `{{` needed for the RLM library's pass (e.g., `{{"name": "respond"}}` → `{"name": "respond"}` → KeyError). Using `.replace("SIZE_THRESHOLD", "20000")` only touches our placeholder and leaves all braces intact.

## How `llm_query()` works

- Already registered in the REPL sandbox (`local_repl.py:165`)
- `max_depth=1` means `llm_query()` makes a direct LLM call (no nested REPL) — exactly right for summarization
- The sub-LLM call input is ~11K chars (6K wiki + 5K older history) — trivial for any model
- Cost: ~$0.0002 per chunked turn at OpenRouter Qwen3-8B pricing
- Already has try/except returning error string on failure; model can fall back to `print(context)`

## Verification

- `py_compile.compile()` passes for both files
- Section markers verified with `str.find()` extraction on test prompts
- `.replace()` confirmed to substitute threshold while preserving `{{` double braces
- With realistic wiki (~6K) and 14 tools, contexts reach 12-23K range (observation #2038), crossing the 20K threshold at turn 10+ with long tool results

---

### Fix (Change 15): Simplify AGENT_SYSTEM_PROMPT after regression — model copying examples

**Problem**: After Change 13's two-workflow system prompt, the model regressed to 0/2 reward on airline tasks 3 and 4. Two issues:
1. Model copied the action example `"How can I help you today?"` verbatim as its default response on nearly every turn (same class of bug as Change 11 / `sara_doe_496`)
2. Model skipped all REPL steps (no `print(context)`, no size check) — the two-workflow structure (Step 1 → Step 2 → Workflow A/B) was too complex; model jumped straight to FINAL with the example text

**Log**: `results/rlm-qwen3-32b-0.0_range_0--1_user-qwen-qwen3-32b-llm_0219143733.log`

**File**: `rlm_bench/rlm_agent.py`

**Changes**:
1. Restored the simple workflow structure from the original prompt: `1. Read context → 2. Decide → 3. FINAL`. The size-aware chunking is now an inline `if/else` inside the single REPL block (Step 1), not two separate named workflows.
2. Removed copyable example text (`"How can I help you today?"`). Format examples now use `"..."` placeholder instead of realistic text.
3. Moved CRITICAL RULES above the workflow (first thing model sees after the intro).
4. Restored the original simpler ROOT_PROMPT: `"Read the context and output FINAL..."`.

**Key insight**: The model treats action examples as defaults. Any realistic text in examples becomes the model's fallback response. Use `"..."` placeholders in examples to force the model to generate its own responses from context.

### Fix (Change 16): Revert AGENT_SYSTEM_PROMPT — if/else chunking caused total context-reading failure

**Problem**: After Change 15's inline `if/else` system prompt, the model regressed further to 0/3 on airline tasks 0, 1, 2. TWO new failure modes:
1. Model NEVER executed `print(context)` — it jumped straight to FINAL() on every call, meaning it never saw tool descriptions or conversation history
2. Model hallucinated non-existent tool names (`retrieve_account_issue`, `process_refund`, `process_payment`) — because it never read the tool list from context
3. Model output literal `[tool_call]` as text — framework syntax confusion

**Log**: `results/rlm-qwen3-32b-0.0_range_0--1_user-qwen-qwen3-32b-llm_0219150534.log`

**File**: `rlm_bench/rlm_agent.py`

**Changes**: Reverted `AGENT_SYSTEM_PROMPT` to the original proven structure:
```python
"Workflow:\n"
"1. Read the context: ```repl\nprint(context)\n```\n"
"2. Decide your action based on the conversation history and available tools.\n"
"3. Output FINAL([json_array]) with your action.\n\n"
```

Key differences from original:
- Action examples use `"..."` and `"TOOL_NAME"` instead of `"How can I help you today?"` (prevents example copying)
- Removed `llm_query` prohibition (no longer needed since chunking isn't in the prompt)
- Section markers in `prompt_builder.py` remain (harmless extra text, useful for future work)

**Lesson learned**: Multi-line Python blocks (if/else, str.find(), llm_query) in the system prompt are ignored by the model. The original one-liner `print(context)` was the simplest possible instruction and still only worked *some* of the time. The chunking optimization is premature — the model must first reliably read context before we can optimize *how* it reads context.
