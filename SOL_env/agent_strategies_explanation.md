# Agent Strategies in Tau-Bench

This document explains the fundamental low-level differences between the three agent strategies used in Tau-Bench (`tool-calling`, `act`, and `react`).

## Overview

The strategies differ primarily in **how they structure the prompt** sent to the LLM and **how they parse the action** from the LLM's output.

| Strategy | Prompt content | Output Format | Processing Logic |
| :--- | :--- | :--- | :--- |
| **Tool-Calling** | Domain Wiki only | Native API Structure | Uses `message.tool_calls` API field |
| **Act** | Wiki + "Output JSON" instr. | Raw Text (Action only) | String split on "Action:" -> JSON parse |
| **ReAct** | Wiki + "Think then Act" instr. | Raw Text (Thought + Action) | String split on "Action:" -> JSON parse |

---

## 1. Tool-Calling (`tool-calling`)

*   **Source Code:** `tau_bench/agents/tool_calling_agent.py`
*   **Mechanism:** Uses the **native function-calling API** of the model (e.g., OpenAI's `tools` parameter, or similar for open weights models compatible with tool use).
*   **Prompting:**
    *   The system prompt is clean. It typically only contains the domain knowledge (e.g., the retail policy wiki).
    *   It does *not* need explicit instructions on *how* to call tools because the model is trained/fine-tuned to understand the schema passed in the API call.
*   **Execution Loop:**
    1.  The agent sends the conversation history + a JSON schema of available tools to the LLM's API.
    2.  The LLM returns a specialized structured response (the `tool_calls` object).
    3.  The agent reads this structured object directly to execute the tool.

## 2. Act (`act`)

*   **Source Code:** `tau_bench/agents/chat_react_agent.py` (initialized with `use_reasoning=False`)
*   **Mechanism:** Uses a standard **pure text chat** completion. It forces the model to act via string parsing.
*   **Prompting:**
    *   Appends a specific `ACT_INSTRUCTION` to the prompt.
    *   **Instruction:** "At each step, your generation should have exactly the following format: Action: {name: ..., arguments: ...}".
*   **Execution Loop:**
    1.  The agent sends the prompt.
    2.  The LLM outputs a raw string like: `Action: {"name": "search", "arguments": { ... }}`.
    3.  The agent parses the string: it looks for "Action:", slices the text after it, and attempts to parse it as JSON.
*   **Key Characteristic:** It asks the model to jump **straight to the action** without explaining why. It treats the model as a direct input-output mapping machine.

## 3. ReAct (`react`)

*   **Source Code:** `tau_bench/agents/chat_react_agent.py` (initialized with `use_reasoning=True`)
*   **Mechanism:** Uses the same class as `act`, but enables **Chain-of-Thought** reasoning.
*   **Prompting:**
    *   Appends `REACT_INSTRUCTION` to the prompt.
    *   **Instruction:** "At each step... format: Thought: <reasoning> Action: <json>".
*   **Execution Loop:**
    1.  The agent sends the prompt.
    2.  The LLM outputs a string like:
        > Thought: The user wants to find a flight. I should check the schedule first to see what is available.
        > Action: {"name": "check_schedule", ...}
    3.  The agent ignores the "Thought" part for the immediate programmatic execution but keeps it in the conversation history. It parses the "Action" part to run the tool.
*   **Key Characteristic:** It forces the model to "think out loud" before acting. This explicit reasoning step often helps the model make better decisions (improving performance on complex tasks) but consumes more tokens and time.
