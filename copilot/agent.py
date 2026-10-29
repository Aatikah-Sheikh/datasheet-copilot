"""Tool-calling agent loop on Nemotron (Nebius Token Factory)."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from openai import BadRequestError

from .llm import NO_THINKING, clean
from .tools import ToolBox

MAX_TOOL_RESULT_CHARS = 12000
# A citation is a bracket like [doc p.4], [doc p.1 & p.4], [doc p.1, p.4] or [doc pp. 1-4],
# not followed by "(" (that would be a Markdown link).
CITATION = re.compile(r"\[([^\[\]\n]{3,160}?)\](?!\()")
_CITATION_DOC = re.compile(r"(.+?)\s+pp?\.\s*\d")
# A label before the document name, e.g. "read_graph: lm358.pdf p.23" or "Source: lm358.pdf p.4"
_CITATION_LABEL = re.compile(r"^[A-Za-z_ ]{2,30}:\s*")
_NUMBER = re.compile(r"\d{1,4}")


def parse_citation(inner: str) -> tuple[str, list[int]] | None:
    """('lm358.pdf', [1, 4]) for 'lm358.pdf p.1 & p.4'; None if the bracket is not a page citation."""
    inner = _CITATION_LABEL.sub("", inner.strip())
    match = _CITATION_DOC.match(inner)
    if not match:
        return None
    return match.group(1).strip(), [int(n) for n in _NUMBER.findall(inner[match.end(1):])]


def iter_citations(text: str):
    """Yield (regex match, document, pages) for every page citation in `text`."""
    for match in CITATION.finditer(text):
        parsed = parse_citation(match.group(1))
        if parsed:
            yield match, *parsed

SYSTEM_PROMPT = """You are Datasheet Copilot, a design assistant for electronics engineers and students who work from component datasheets.

Rules:
1. Ground every datasheet fact in a search_datasheets or read_graph result and cite it inline as [<document> p.<page>], e.g. [lm358.pdf p.23]: square brackets, the document name only (no tool name), including inside tables. If the loaded datasheets do not state a value, say so plainly. Never guess or recall a spec from memory.
2. Do every numerical calculation with the calculate tool, passing inputs with units. Never do arithmetic yourself: any number you derive (a ratio, unit conversion, dB value or margin) must come from a calculate call in this turn. Quote the tool's result, and never say you used the calculator unless you called it.
3. State assumptions (operating conditions, temperature, load). For design margins use worst-case (min/max) values rather than typical, and say which you used.
4. If a datasheet you need is not loaded, find the manufacturer's PDF with web_search (if available) and load it with fetch_datasheet_pdf, then search it. Prefer manufacturer domains.
5. Some values appear only in graphs (typical-characteristics curves). If read_graph is available, find the graph's page with search_datasheets (figure captions are in the page text), then call read_graph with that page and a question naming the figure, the x value and the condition. Give graph readings as approximate (≈), say they come from a typical curve rather than a guaranteed limit, and cite the page.
6. Structure answers as: the answer first; then "Working" listing each value used with its citation, the calculation and the result; then caveats. Be concise.
7. Search in every turn. Results from earlier turns in the conversation are not evidence for this answer. Never describe a search you did not run: the Working section must match the tool calls you actually made.
8. Be economical. When you need several values, request all the searches together in one step (several tool calls at once), and stop searching once you have what you need. For comparisons, cover the five or six specs that matter most, not every row of the datasheet.
9. Use British English."""

FINAL_NUDGE = (
    "Stop calling tools. Give your best answer now from the results above. Cite every value as "
    "[document p.N] using only pages those results returned, and say clearly which parts are unverified."
)
MISSING_PAGES_NOTE = (
    "Your answer cites pages you did not retrieve in this turn: {pages}. Earlier turns are not evidence. "
    "Use search_datasheets now to check each of those values, citing only passages you retrieved in this "
    "turn. Remove any value you cannot find."
)
FALSE_CALC_NOTE = (
    "Your answer says it used the calculator, but you made no calculate call in this turn. "
    "Call calculate now for every number you derived (ratios, conversions, dB values, margins) and quote its results."
)
_CALCULATOR_CLAIM = re.compile(r"calculate tool|calculator|\[calculate\]", re.IGNORECASE)


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


def retrieved_pages(steps: list[Step]) -> set[tuple[str, int]]:
    """Pages this turn actually looked at: search hits and pages whose graphs were read."""
    pages = {
        (hit["doc"], int(hit["page"]))
        for step in steps
        if step.tool == "search_datasheets"
        for hit in step.result.get("results", [])
    }
    return pages | {
        (step.result["doc"], int(step.result["page"]))
        for step in steps
        if step.tool == "read_graph" and "error" not in step.result
    }


def for_model(result: dict) -> str:
    """A tool result as sent to the model: JSON without the UI-only "_" keys, truncated."""
    return json.dumps({k: v for k, v in result.items() if not k.startswith("_")}, ensure_ascii=False)[:MAX_TOOL_RESULT_CHARS]


def unsupported_citations(answer: str, steps: list[Step]) -> list[tuple[str, int]]:
    """Pages cited as [doc p.N] that no search in this turn actually returned."""
    retrieved, missing = retrieved_pages(steps), []
    for _, doc, pages in iter_citations(answer):
        for page in pages:
            key = (doc, page)
            if key not in retrieved and key not in missing:
                missing.append(key)
    return missing


def false_calculator_claim(answer: str, steps: list[Step]) -> bool:
    """True if the answer credits the calculator but no calculate call succeeded this turn."""
    calculated = any(s.tool == "calculate" and "error" not in s.result for s in steps)
    return bool(_CALCULATOR_CLAIM.search(answer)) and not calculated


def _create(client, tool_choice: str, **kwargs):
    """Chat completion; falls back to tool_choice='auto' if the server rejects 'required'."""
    try:
        return client.chat.completions.create(tool_choice=tool_choice, **kwargs)
    except BadRequestError:
        if tool_choice == "auto":
            raise
        return client.chat.completions.create(tool_choice="auto", **kwargs)


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
    rechecked = False
    for index in range(max_steps):
        # With datasheets loaded, the first move must be a tool call, so the model cannot
        # answer from memory (it did exactly that on a follow-up comparison question).
        response = _create(
            client,
            "required" if index == 0 and toolbox.library.docs() else "auto",
            model=model,
            messages=messages,
            tools=toolbox.schemas(),
            temperature=0.2,
            max_tokens=4096,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            answer = clean(message.content)
            if not answer:
                break  # empty reply (reasoning used up the budget): go to the forced answer below
            # One recheck if the answer claims evidence it does not have: pages it never
            # retrieved this turn, or calculator results it never computed.
            notes = []
            missing = unsupported_citations(answer, steps)
            if missing:
                notes.append(MISSING_PAGES_NOTE.format(pages=", ".join(f"{d} p.{p}" for d, p in missing)))
            if false_calculator_claim(answer, steps):
                notes.append(FALSE_CALC_NOTE)
            if notes and not rechecked and index + 1 < max_steps:
                rechecked = True
                messages.append({"role": "assistant", "content": message.content or ""})
                messages.append({"role": "user", "content": " ".join(notes) + " Then give the corrected final answer."})
                continue
            return AgentResult(answer, steps)
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
                    "content": for_model(result),
                }
            )
    # Reasoning off for the forced answer: with it on, Nemotron can spend the whole output
    # budget thinking and return empty content (seen after a long Tavily lookup).
    response = client.chat.completions.create(
        model=model,
        messages=[*messages, {"role": "user", "content": FINAL_NUDGE}],
        temperature=0.2,
        max_tokens=4096,
        extra_body=NO_THINKING,
    )
    return AgentResult(clean(response.choices[0].message.content), steps)
