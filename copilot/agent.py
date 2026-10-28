"""Tool-calling agent loop on Nemotron (Nebius Token Factory)."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from .llm import clean
from .tools import ToolBox

MAX_TOOL_RESULT_CHARS = 12000

SYSTEM_PROMPT = """You are Datasheet Copilot, a design assistant for electronics engineers and students who work from component datasheets.

Rules:
1. Ground every datasheet fact in a search_datasheets result and cite it inline as [<document> p.<page>]. If the loaded datasheets do not state a value, say so plainly. Never guess or recall a spec from memory.
2. Do every numerical calculation with the calculate tool, passing inputs with units. Never do arithmetic yourself. Quote the tool's result.
3. State assumptions (operating conditions, temperature, load). For design margins use worst-case (min/max) values rather than typical, and say which you used.
4. If a datasheet you need is not loaded, find the manufacturer's PDF with web_search (if available) and load it with fetch_datasheet_pdf, then search it. Prefer manufacturer domains.
5. Structure answers as: the answer first; then "Working" listing each value used with its citation, the calculation and the result; then caveats. Be concise.
6. Use British English."""

FINAL_NUDGE = (
    "Stop calling tools. Give your best answer now from the results above, "
    "and say clearly which parts are unverified."
)


@dataclass
class Step:
    tool: str
    arguments: dict
    result: dict


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)


def _library_context(toolbox: ToolBox) -> str:
    counts = toolbox.library.page_counts
    if not counts:
        return "No datasheets are loaded yet."
    return "Loaded datasheets: " + "; ".join(f"{doc} ({pages} pages)" for doc, pages in counts.items())


def _parse_arguments(raw: str | None) -> dict:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_unparsed": raw}
    return value if isinstance(value, dict) else {"_unparsed": raw}


def run_agent(
    client,
    model: str,
    toolbox: ToolBox,
    question: str,
    history: list[dict] | None = None,
    max_steps: int = 8,
    on_step: Callable[[Step], None] | None = None,
) -> AgentResult:
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{_library_context(toolbox)}"},
        *(history or []),
        {"role": "user", "content": question},
    ]
    steps: list[Step] = []
    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=toolbox.schemas(),
            tool_choice="auto",
            temperature=0.2,
            max_tokens=4096,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return AgentResult(clean(message.content), steps)
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in message.tool_calls
                ],
            }
        )
        for call in message.tool_calls:
            result = toolbox.call(call.function.name, call.function.arguments)
            step = Step(call.function.name, _parse_arguments(call.function.arguments), result)
            steps.append(step)
            if on_step:
                on_step(step)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result)[:MAX_TOOL_RESULT_CHARS],
                }
            )
    response = client.chat.completions.create(
        model=model,
        messages=[*messages, {"role": "user", "content": FINAL_NUDGE}],
        temperature=0.2,
        max_tokens=4096,
    )
    return AgentResult(clean(response.choices[0].message.content), steps)
