"""Presentation helpers for the Streamlit app: styling, the hero band, citation chips, the evidence
panel, spec cards and the token meter. Free of API calls so the layout can be previewed offline.

Colour encodes provenance throughout: pink = taken from a datasheet, amber = computed by the
calculator, lime = the app's own chrome and status.

Every colour is derived from `currentColor` (the active Streamlit theme's text colour) with
color-mix, so panels and accents follow whichever theme is actually rendered, light or charcoal
dark, with no server-side theme detection. Mixing an accent towards currentColor darkens it on
light backgrounds and lightens it on dark ones, which keeps it readable in both.
"""

import base64
import html
import re

import streamlit as st

from .agent import CITATION, Step, iter_citations, parse_citation

CSS = """
<style>
:root {
  --mono: "IBM Plex Mono", ui-monospace, monospace;
  --ink: currentColor;
  --muted: color-mix(in srgb, currentColor 62%, transparent);
  --surface: color-mix(in srgb, currentColor 4%, transparent);
  --raised: color-mix(in srgb, currentColor 6%, transparent);
  --user: color-mix(in srgb, currentColor 8%, transparent);
  --line: color-mix(in srgb, currentColor 14%, transparent);
  --src: color-mix(in srgb, #FF3D8B 76%, currentColor);
  --src-tint: color-mix(in srgb, #FF3D8B 16%, transparent);
  --src-bg: color-mix(in srgb, #FF3D8B 7%, transparent);
  --calc: color-mix(in srgb, #F5A000 68%, currentColor);
  --calc-tint: color-mix(in srgb, #F5A000 18%, transparent);
  --calc-bg: color-mix(in srgb, #F5A000 8%, transparent);
  --ok: color-mix(in srgb, #A6E01F 66%, currentColor);
  --ok-tint: color-mix(in srgb, #A6E01F 20%, transparent);
  --err: color-mix(in srgb, #FF4D4D 74%, currentColor);
  --err-tint: color-mix(in srgb, #FF4D4D 13%, transparent);
  --hero: linear-gradient(120deg, #1A1A1A 0%, #232323 55%, #2B2230 100%);
  --hero-line: #2E2E2E;
}
/* Primary buttons are lime in both themes, so they take charcoal text, not Streamlit's default white */
[data-testid="stBaseButton-primary"]:not(:disabled), [data-testid="stBaseButton-primary"]:not(:disabled) p { color: #141414 !important; font-weight: 600; }
.block-container { padding-top: 3.6rem; padding-bottom: 2rem; max-width: 1360px; }  /* clears Streamlit's top bar */

/* Hero band: charcoal in both themes, lit by the three provenance colours */
.dc-hero { position: relative; overflow: hidden; background: var(--hero); border: 1px solid var(--hero-line); border-radius: 14px;
  padding: 1.35rem 1.6rem; display: flex; justify-content: space-between; align-items: center; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1.3rem; }
.dc-hero::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 3px;
  background: linear-gradient(90deg, #C6F135 0%, #C6F135 33%, #FF5FA2 33%, #FF5FA2 66%, #FFB547 66%, #FFB547 100%); }
.dc-hero .eyebrow { font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.14em; color: #C6F135; text-transform: uppercase; }
.dc-hero h1 { color: #FFFFFF; font-size: 2rem; line-height: 1.1; margin: 0.25rem 0 0.3rem 0; padding: 0; letter-spacing: -0.015em; }
.dc-hero p { color: #BDBDBD; margin: 0; font-size: 0.95rem; }
.dc-guarantees { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.dc-g { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12); border-radius: 8px; padding: 0.5rem 0.75rem; min-width: 9.5rem; }
.dc-g .k { font-family: var(--mono); font-size: 0.8rem; font-weight: 600; }
.dc-g .v { color: #D6D6D6; font-size: 0.8rem; margin-top: 0.1rem; }
.dc-g.src .k { color: #FF5FA2; } .dc-g.calc .k { color: #FFB547; } .dc-g.trace .k { color: #C6F135; }

/* Panels */
.dc-panel-title { font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.12em; color: var(--muted); text-transform: uppercase; margin: 0 0 0.55rem 0; }
.dc-steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.7rem; margin-bottom: 1.1rem; }
.dc-howto { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 0.85rem 0.95rem; }
.dc-howto .n { display: inline-block; font-family: var(--mono); font-size: 0.72rem; font-weight: 600; color: var(--ok); background: var(--ok-tint); border-radius: 4px; padding: 0.05rem 0.4rem; }
.dc-howto .t { font-weight: 600; margin: 0.45rem 0 0.2rem 0; }
.dc-howto .d { color: var(--muted); font-size: 0.85rem; line-height: 1.4; }
@media (max-width: 900px) { .dc-steps { grid-template-columns: 1fr; } }

/* Chat. Columns must be allowed to shrink below their content (flex min-width: auto
   otherwise lets a wide table push the conversation under the evidence panel). */
[data-testid="stColumn"], [data-testid="stChatMessageContent"] { min-width: 0; }
[data-testid="stChatMessage"] { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 0.9rem 1rem; }
[data-testid="stChatMessageContent"] table { display: block; max-width: 100%; overflow-x: auto; }
[data-testid="stChatMessageContent"] td, [data-testid="stChatMessageContent"] th { overflow-wrap: anywhere; min-width: 6rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: var(--user); }
/* Grounding badge under each answer */
.dc-verify { display: flex; align-items: flex-start; gap: 0.55rem; font-size: 0.8rem; line-height: 1.45; margin-top: 0.5rem;
  padding: 0.45rem 0.65rem; border-radius: 7px; border: 1px solid var(--line); background: var(--raised); }
.dc-verify .mark { flex: 0 0 auto; display: inline-flex; align-items: center; justify-content: center; width: 1.1rem; height: 1.1rem;
  border-radius: 50%; font-size: 0.7rem; font-weight: 700; background: var(--line); margin-top: 0.08rem; }
.dc-verify.ok { border-color: var(--ok-tint); background: color-mix(in srgb, #A6E01F 7%, transparent); }
.dc-verify.ok .mark { background: var(--ok-tint); color: var(--ok); } .dc-verify.ok b { color: var(--ok); }
.dc-verify.warn { border-color: var(--err-tint); background: var(--err-tint); }
.dc-verify.warn .mark { background: var(--err-tint); color: var(--err); } .dc-verify.warn b { color: var(--err); }

/* Chips */
.dc-cite { display: inline-block; font-family: var(--mono); font-size: 0.76em; line-height: 1.55; background: var(--src-tint);
  color: var(--src); border-radius: 4px; padding: 0 0.42em; margin: 0 0.08em; white-space: nowrap; }
.dc-cite.calc { background: var(--calc-tint); color: var(--calc); }

/* Workspace: conversation | evidence. Stack below ~1200px, where the sidebar leaves too little room. */
@media (max-width: 1200px) {
  [data-testid="stHorizontalBlock"]:has(.dc-ev) { flex-direction: column; gap: 1.2rem; }
  [data-testid="stHorizontalBlock"]:has(.dc-ev) > [data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important; min-width: 100%; }
}

/* Evidence panel */
.dc-ev { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.05rem; }
.dc-ev .for { font-size: 0.84rem; color: var(--muted); margin: -0.2rem 0 0.8rem 0; line-height: 1.4; }
.dc-ev .sec { display: flex; align-items: center; gap: 0.45rem; font-size: 0.78rem; font-weight: 600; margin: 0.9rem 0 0.45rem 0; }
.dc-ev .sec .dot { width: 0.5rem; height: 0.5rem; border-radius: 50%; }
.dc-ev .sec.src .dot { background: var(--src); } .dc-ev .sec.calc .dot { background: var(--calc); } .dc-ev .sec.trace .dot { background: var(--ok); }
.dc-ev .count { font-family: var(--mono); color: var(--muted); font-weight: 400; }
.dc-quote { border-left: 3px solid var(--src); background: var(--src-bg); border-radius: 0 6px 6px 0; padding: 0.5rem 0.65rem; margin-bottom: 0.5rem; }
.dc-quote .ref { font-family: var(--mono); font-size: 0.74rem; color: var(--src); font-weight: 600; }
.dc-quote .txt { font-size: 0.82rem; color: var(--ink); opacity: 0.88; line-height: 1.45; margin-top: 0.2rem; word-break: break-word; }
.dc-quote.missing { border-left-color: var(--err); background: var(--err-tint); }
.dc-quote.missing .ref { color: var(--err); }
.dc-graph { border-left: 3px solid var(--src); background: var(--src-bg); border-radius: 0 6px 6px 0; padding: 0.5rem 0.65rem; margin-bottom: 0.5rem; }
.dc-graph .ref { font-family: var(--mono); font-size: 0.74rem; color: var(--src); font-weight: 600; }
.dc-graph .val { font-family: var(--mono); font-size: 1rem; font-weight: 600; color: var(--src); margin-top: 0.2rem; }
.dc-graph .txt { font-size: 0.8rem; color: var(--ink); opacity: 0.88; line-height: 1.45; margin-top: 0.2rem; }
.dc-graph summary { cursor: pointer; font-size: 0.76rem; color: var(--muted); margin-top: 0.35rem; }
.dc-graph img { width: 100%; border-radius: 4px; margin-top: 0.4rem; background: #FFFFFF; }
.dc-calc { border-left: 3px solid var(--calc); background: var(--calc-bg); border-radius: 0 6px 6px 0; padding: 0.5rem 0.65rem; margin-bottom: 0.5rem; }
.dc-calc .expr { font-family: var(--mono); font-size: 0.84rem; color: var(--ink); word-break: break-word; }
.dc-calc .in { font-family: var(--mono); font-size: 0.74rem; color: var(--muted); margin-top: 0.15rem; }
.dc-calc .out { font-family: var(--mono); font-size: 1rem; font-weight: 600; color: var(--calc); margin-top: 0.2rem; }
.dc-timeline { border-left: 2px solid var(--line); margin: 0.1rem 0 0 0.3rem; padding-left: 0.85rem; }
.dc-step { position: relative; margin: 0 0 0.5rem 0; font-size: 0.8rem; line-height: 1.45; color: var(--ink); word-break: break-word; }
.dc-step::before { content: ""; position: absolute; left: -1.2rem; top: 0.38rem; width: 0.5rem; height: 0.5rem; border-radius: 50%; background: var(--ok); }
.dc-step.error::before { background: var(--err); }
.dc-tag { font-family: var(--mono); font-size: 0.64rem; font-weight: 600; letter-spacing: 0.06em; padding: 0.03rem 0.35rem; border-radius: 3px; background: var(--ok-tint); color: var(--ok); margin-right: 0.35rem; }
.dc-step.error .dc-tag { background: var(--err-tint); color: var(--err); }
.dc-muted { color: var(--muted); }
.dc-empty-ev { text-align: left; color: var(--muted); font-size: 0.86rem; line-height: 1.5; }
.dc-empty-ev .row { display: flex; gap: 0.55rem; align-items: flex-start; margin-top: 0.6rem; }
.dc-empty-ev .sw { flex: 0 0 auto; width: 0.7rem; height: 0.7rem; border-radius: 3px; margin-top: 0.3rem; }

/* Sidebar */
.dc-side-title { font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.12em; color: var(--muted); text-transform: uppercase; margin-top: 0.2rem; }
.dc-brand { display: flex; align-items: center; gap: 0.55rem; margin: -0.4rem 0 0.4rem 0; }
.dc-brand .logo { width: 1.9rem; height: 1.9rem; border-radius: 7px; background: #161616; border: 1px solid #333333;
  color: #C6F135; font-family: var(--mono); font-weight: 600; font-size: 0.8rem; display: flex; align-items: center; justify-content: center; }
.dc-brand .name { font-weight: 700; font-size: 1rem; }
.dc-card { background: var(--raised); border: 1px solid var(--line); border-radius: 10px; padding: 0.7rem 0.75rem; margin: 0.15rem 0 0.5rem 0; }
.dc-card .pn { font-family: var(--mono); font-weight: 600; font-size: 1rem; color: var(--ok); }
.dc-card .mf { font-size: 0.76rem; color: var(--muted); }
.dc-card .ds { font-size: 0.8rem; color: var(--ink); opacity: 0.88; margin: 0.3rem 0 0.45rem 0; line-height: 1.35; }
.dc-card table { width: 100%; border-collapse: collapse; font-size: 0.76rem; }
.dc-card td { border-top: 1px solid var(--line); padding: 0.3rem 0.1rem; vertical-align: top; }
.dc-card td.v { font-family: var(--mono); white-space: nowrap; padding-left: 0.4rem; }
.dc-card td.p { text-align: right; white-space: nowrap; }
.dc-card .pg { font-family: var(--mono); font-size: 0.68rem; color: var(--src); }
.dc-card .cond { font-size: 0.68rem; color: var(--muted); }
.dc-card .note { font-size: 0.68rem; color: var(--muted); margin-top: 0.45rem; }
.dc-meter { margin: 0.3rem 0 0.5rem 0; }
.dc-meter .row { display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--muted); font-family: var(--mono); }
.dc-meter .bar { height: 5px; background: var(--line); border-radius: 3px; overflow: hidden; margin-top: 0.18rem; }
.dc-meter .fill { height: 100%; background: var(--ok); }
.dc-meter .fill.warn { background: var(--calc); }
.dc-models { font-family: var(--mono); font-size: 0.7rem; color: var(--muted); line-height: 1.7; margin-top: 0.5rem; }
.dc-models b { color: var(--ink); font-weight: 500; }
</style>
"""

HOW_IT_WORKS = [
    ("01", "Load datasheets", "Upload PDFs, paste a URL, or name a part and it fetches the manufacturer's datasheet."),
    ("02", "Ask a design question", "Bandwidth, margins, power, noise: in plain language, with your operating conditions."),
    ("03", "Check the evidence", "Each answer lists the exact passages it relied on and every calculation it ran."),
]

_CALC_CITATION = re.compile(r"\[(?:calculate|calculator)(?: (?:tool|output|result))?\]", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]{3,}")


def inject_css() -> None:
    st.html(CSS)


def render_hero() -> None:
    st.html(
        """
<div class="dc-hero">
  <div>
    <div class="eyebrow">NVIDIA Nemotron · Nebius Token Factory</div>
    <h1>Datasheet Copilot</h1>
    <p>Design answers from your component datasheets that you can check line by line.</p>
  </div>
  <div class="dc-guarantees">
    <div class="dc-g src"><div class="k">[doc p.N]</div><div class="v">Every spec cited</div></div>
    <div class="dc-g calc"><div class="k">f(x) = 35 kHz</div><div class="v">Every number computed</div></div>
    <div class="dc-g trace"><div class="k">search &rarr; calc</div><div class="v">Every step shown</div></div>
  </div>
</div>"""
    )


def render_brand() -> None:
    st.html('<div class="dc-brand"><div class="logo">DC</div><div class="name">Datasheet Copilot</div></div>')


def panel_title(text: str) -> None:
    st.html(f'<div class="dc-panel-title">{html.escape(text)}</div>')


def render_how_it_works() -> None:
    cards = "".join(
        f'<div class="dc-howto"><span class="n">{n}</span><div class="t">{t}</div><div class="d">{d}</div></div>'
        for n, t, d in HOW_IT_WORKS
    )
    st.html(f'<div class="dc-steps">{cards}</div>')


def format_answer(text: str) -> str:
    """Turn [doc p.N] and [calculate] citations into styled chips (rendered with unsafe_allow_html)."""

    def chip(match: re.Match) -> str:
        parsed = parse_citation(match.group(1))
        if not parsed:
            return match.group(0)  # an ordinary bracket, left as written
        doc, pages = parsed
        return "".join(f'<span class="dc-cite">{html.escape(doc)} · p.{page}</span>' for page in pages)

    return _CALC_CITATION.sub('<span class="dc-cite calc">calculator</span>', CITATION.sub(chip, text))


def render_turn(turn: dict) -> None:
    avatar = ":material/person:" if turn["role"] == "user" else ":material/memory:"
    with st.chat_message(turn["role"], avatar=avatar):
        if turn["role"] == "assistant":
            st.markdown(format_answer(turn["content"]), unsafe_allow_html=True)
            steps = turn.get("steps", [])
            if steps:
                st.html(verification_html(turn["content"], steps))
        else:
            st.markdown(turn["content"])


def _snippet(text: str, query: str, width: int = 300) -> str:
    """A window of `text` around the first query word it contains, so the quote shows the value."""
    lowered = text.lower()
    positions = [lowered.find(w) for w in _WORD.findall(query.lower()) if lowered.find(w) >= 0]
    start = max(0, min(positions) - 60) if positions else 0
    snippet = text[start : start + width].strip()
    return ("… " if start else "") + snippet + (" …" if start + width < len(text) else "")


def _cited_passages(answer: str, steps: list[Step]) -> list[tuple[str, int, str | None]]:
    """(doc, page, quote) for each page cited in the answer; quote is None if no retrieved passage backs it."""
    retrieved: dict[tuple[str, int], str] = {}
    for step in steps:
        if step.tool == "search_datasheets":
            query = str(step.arguments.get("query", ""))
            for hit in step.result.get("results", []):
                retrieved.setdefault((hit["doc"], int(hit["page"])), _snippet(hit["text"], query))
    for reading in _graph_readings(steps):  # a graph reading says more about its page than a text snippet
        retrieved[(reading["doc"], int(reading["page"]))] = f'Graph reading: {reading["value"]}. {reading.get("reading") or ""}'
    seen, passages = set(), []
    for _, doc, pages in iter_citations(answer):
        for page in pages:
            key = (doc, page)
            if key not in seen:
                seen.add(key)
                passages.append((*key, retrieved.get(key)))
    return passages


def _graph_readings(steps: list[Step]) -> list[dict]:
    return [s.result for s in steps if s.tool == "read_graph" and "error" not in s.result]


def verification_summary(answer: str, steps: list[Step]) -> dict:
    """Counts behind the badge under each answer: cited pages backed by a retrieved passage or
    graph reading, cited pages with neither, successful calculator runs and graph readings."""
    passages = _cited_passages(answer, steps)
    return {
        "verified": sum(1 for *_, quote in passages if quote is not None),
        "unverified": sum(1 for *_, quote in passages if quote is None),
        "calculations": sum(1 for s in steps if s.tool == "calculate" and "error" not in s.result),
        "graphs": len(_graph_readings(steps)),
    }


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def verification_html(answer: str, steps: list[Step]) -> str:
    counts = verification_summary(answer, steps)
    calcs = f" · {_plural(counts['calculations'], 'calculation')} by the calculator" if counts["calculations"] else ""
    if counts["graphs"]:
        calcs += f" · {_plural(counts['graphs'], 'value')} read from graphs (approximate)"
    trace = f'<span class="dc-muted"> · {_plural(len(steps), "tool step")}, shown in the evidence panel</span>'
    if counts["unverified"]:
        mark, kind = "!", "warn"
        text = (
            f'<b>{counts["unverified"]} of {counts["verified"] + counts["unverified"]} cited pages not found</b> '
            f"in the retrieved passages: check them in the evidence panel{calcs}"
        )
    elif counts["verified"]:
        mark, kind = "&#10003;", "ok"
        text = f'<b>Grounding check passed</b> · {_plural(counts["verified"], "cited page")} matched to retrieved passages{calcs}{trace}'
    else:
        mark, kind = "i", ""
        text = f"No datasheet pages cited{calcs}{trace}"
    return f'<div class="dc-verify {kind}"><span class="mark">{mark}</span><span>{text}</span></div>'


def _step_html(step: Step) -> str:
    args, result = step.arguments, step.result
    if "error" in result:
        return (
            f'<div class="dc-step error"><span class="dc-tag">{html.escape(step.tool.split("_")[0].upper())}</span>'
            f'<span class="dc-muted">{html.escape(str(result["error"]))}</span></div>'
        )
    if step.tool == "search_datasheets":
        pages = ", ".join(f'p.{h["page"]}' for h in result.get("results", [])) or "no matches"
        return f'<div class="dc-step"><span class="dc-tag">SEARCH</span>&ldquo;{html.escape(str(args.get("query", "")))}&rdquo; <span class="dc-muted">&rarr; {pages}</span></div>'
    if step.tool == "calculate":
        return f'<div class="dc-step"><span class="dc-tag">CALC</span>{html.escape(result["expression"])} <span class="dc-muted">&rarr; {html.escape(result["display"])}</span></div>'
    if step.tool == "web_search":
        return f'<div class="dc-step"><span class="dc-tag">WEB</span>&ldquo;{html.escape(str(args.get("query", "")))}&rdquo; <span class="dc-muted">&rarr; {len(result.get("results", []))} results</span></div>'
    if step.tool == "read_graph":
        figure = f' {html.escape(str(result["figure"]))}' if result.get("figure") else ""
        return (
            f'<div class="dc-step"><span class="dc-tag">GRAPH</span>{html.escape(result["doc"])} p.{result["page"]}{figure} '
            f'<span class="dc-muted">&rarr; {html.escape(str(result["value"]))}</span></div>'
        )
    if step.tool == "fetch_datasheet_pdf":
        return f'<div class="dc-step"><span class="dc-tag">LOAD</span>{html.escape(result["added"])} <span class="dc-muted">({result["pages"]} pages)</span></div>'
    return f'<div class="dc-step"><span class="dc-tag">{html.escape(step.tool.upper())}</span></div>'


def live_step_html(step: Step) -> str:
    return '<div class="dc-timeline">' + _step_html(step) + "</div>"


def evidence_html(turn: dict, question: str) -> str:
    steps: list[Step] = turn.get("steps", [])
    passages = _cited_passages(turn["content"], steps)
    calcs = [s.result for s in steps if s.tool == "calculate" and "error" not in s.result]
    parts = [
        '<div class="dc-ev"><div class="dc-panel-title">Evidence</div>',
        f'<div class="for">For: {html.escape(question[:160])}{"…" if len(question) > 160 else ""}</div>',
        f'<div class="sec src"><span class="dot"></span>Cited passages <span class="count">{len(passages)}</span></div>',
    ]
    if not passages:
        parts.append('<div class="dc-muted" style="font-size:0.8rem">No datasheet pages were cited.</div>')
    for doc, page, quote in passages:
        if quote is None:
            parts.append(
                f'<div class="dc-quote missing"><div class="ref">{html.escape(doc)} · p.{page}</div>'
                '<div class="txt">Cited, but not among the passages the agent retrieved. Check this page yourself.</div></div>'
            )
        else:
            parts.append(
                f'<div class="dc-quote"><div class="ref">{html.escape(doc)} · p.{page}</div>'
                f'<div class="txt">{html.escape(quote)}</div></div>'
            )
    graphs = _graph_readings(steps)
    if graphs:
        parts.append(f'<div class="sec src"><span class="dot"></span>Graph readings <span class="count">{len(graphs)}</span></div>')
    for reading in graphs:
        figure = f' · {html.escape(str(reading["figure"]))}' if reading.get("figure") else ""
        conditions = f'<div class="txt dc-muted">Curve: {html.escape(str(reading["conditions"]))}</div>' if reading.get("conditions") else ""
        image = ""
        if reading.get("_image"):
            data = base64.b64encode(reading["_image"]).decode()
            image = f'<details><summary>Show the page that was read</summary><img alt="Page {reading["page"]}" src="data:image/jpeg;base64,{data}"></details>'
        parts.append(
            f'<div class="dc-graph"><div class="ref">{html.escape(reading["doc"])} · p.{reading["page"]}{figure}</div>'
            f'<div class="val">{html.escape(str(reading["value"]))}</div>'
            f'<div class="txt">{html.escape(str(reading.get("reading") or ""))}</div>{conditions}'
            f'<div class="txt dc-muted">Read by a vision model from a typical curve. Confidence: {html.escape(str(reading.get("confidence") or "not given"))}.</div>'
            f"{image}</div>"
        )
    parts.append(f'<div class="sec calc"><span class="dot"></span>Calculations <span class="count">{len(calcs)}</span></div>')
    if not calcs:
        parts.append('<div class="dc-muted" style="font-size:0.8rem">No calculations were needed.</div>')
    for calc in calcs:
        inputs = " · ".join(f"{html.escape(k)} = {html.escape(v)}" for k, v in calc.get("inputs", {}).items())
        parts.append(
            f'<div class="dc-calc"><div class="expr">{html.escape(calc["expression"])}</div>'
            + (f'<div class="in">{inputs}</div>' if inputs else "")
            + f'<div class="out">= {html.escape(calc["display"])}</div></div>'
        )
    parts.append(f'<div class="sec trace"><span class="dot"></span>Agent trace <span class="count">{len(steps)}</span></div>')
    parts.append('<div class="dc-timeline">' + "".join(_step_html(s) for s in steps) + "</div></div>")
    return "".join(parts)


def empty_evidence_html() -> str:
    return (
        '<div class="dc-ev"><div class="dc-panel-title">Evidence</div><div class="dc-empty-ev">'
        "Ask a question and this panel shows what the answer rests on."
        '<div class="row"><span class="sw" style="background:var(--src)"></span><span><b style="color:var(--ink)">Cited passages</b>: '
        "the datasheet text or graph behind every spec, by page.</span></div>"
        '<div class="row"><span class="sw" style="background:var(--calc)"></span><span><b style="color:var(--ink)">Calculations</b>: '
        "each expression, its inputs with units, and the result.</span></div>"
        '<div class="row"><span class="sw" style="background:var(--ok)"></span><span><b style="color:var(--ink)">Agent trace</b>: '
        "every search, lookup and calculation, in order.</span></div>"
        "</div></div>"
    )


def spec_card_html(card: dict | None, doc: str, pages: int, model: str) -> str:
    if not card:
        return f'<div class="dc-card"><div class="pn">{html.escape(doc)}</div><div class="mf">{pages} pages · no spec card</div></div>'
    rows = "".join(
        f'<tr><td>{html.escape(str(s.get("parameter") or ""))}'
        + (f'<br><span class="cond">{html.escape(str(s["conditions"]))}</span>' if s.get("conditions") else "")
        + f'</td><td class="v">{html.escape(str(s.get("value") or ""))}</td>'
        f'<td class="p"><span class="pg">p.{html.escape(str(s.get("page") or "?"))}</span></td></tr>'
        for s in card.get("key_specs", [])
        if isinstance(s, dict)
    )
    return (
        f'<div class="dc-card"><div class="pn">{html.escape(card.get("part_number") or doc)}</div>'
        f'<div class="mf">{html.escape(card.get("manufacturer") or "")} · {html.escape(doc)} · {pages} pages</div>'
        + (f'<div class="ds">{html.escape(card["description"])}</div>' if card.get("description") else "")
        + (f"<table>{rows}</table>" if rows else "")
        + f'<div class="note">Extracted by {html.escape(model.split("/")[-1])}. Check values against the cited page.</div></div>'
    )


def _meter_row(name: str, used: int, cap: int) -> str:
    share = min(used / cap, 1.0) if cap else 1.0
    warn = " warn" if share >= 0.8 else ""
    return (
        f'<div class="dc-meter"><div class="row"><span>{name}</span><span>{used:,} / {cap:,}</span></div>'
        f'<div class="bar"><div class="fill{warn}" style="width:{share * 100:.1f}%"></div></div></div>'
    )


def usage_html(session_tokens: int, totals: dict, limits) -> str:
    return (
        _meter_row("Session", session_tokens, limits.per_session)
        + _meter_row("Today", totals["day_tokens"], limits.per_day)
        + _meter_row("All-time", totals["total_tokens"], limits.total)
    )


def models_html(reasoning: str, fast: str, vision: str, web_search: bool) -> str:
    return (
        '<div class="dc-models">'
        f"Reasoning <b>{html.escape(reasoning.split('/')[-1])}</b><br>"
        f"Spec cards <b>{html.escape(fast.split('/')[-1])}</b><br>"
        f"Graphs <b>{html.escape(vision.split('/')[-1])}</b><br>"
        "Inference <b>Nebius Token Factory</b><br>"
        f"Web search <b>{'Tavily' if web_search else 'off'}</b></div>"
    )
