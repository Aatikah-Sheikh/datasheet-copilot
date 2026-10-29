"""Nebius Token Factory client (OpenAI-compatible API)."""

import json
import re

from openai import OpenAI

from .config import Settings

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
# Turns off Nemotron's hidden reasoning for a call (Token Factory passes this to the chat template).
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
_BRACKETED = re.compile(r"(\[[^\[\]\n]*\])")
# "lm358.pdf p.4" or "lm2596-d.pdf p.1 & p.4" written without brackets (models do this in tables)
_BARE_CITATION = re.compile(
    r"(?<![\w./-])[\w.-]+\.pdf\s+pp?\.\s*\d{1,4}(?:\s*(?:,|&|and|-|–)\s*(?:pp?\.\s*)?\d{1,4})*",
    re.IGNORECASE,
)


def make_client(settings: Settings) -> OpenAI:
    if not settings.api_key:
        raise RuntimeError("NEBIUS_API_KEY is not set. Add it to .env or Streamlit secrets.")
    # Retries cover rate limits, 5xx and dropped connections (the SDK backs off between tries);
    # the timeout stops a stalled request leaving a visitor watching a spinner indefinitely.
    return OpenAI(base_url=settings.base_url, api_key=settings.api_key, timeout=120, max_retries=3)


def _bracket_bare_citations(text: str) -> str:
    """Wrap unbracketed 'doc.pdf p.N' references in [ ] so they are chipped and checked like the rest."""
    parts = _BRACKETED.split(text)  # odd indices are existing [...] segments, left untouched
    for i in range(0, len(parts), 2):
        parts[i] = _BARE_CITATION.sub(lambda m: f"[{m.group(0)}]", parts[i])
    return "".join(parts)


def clean(text: str | None) -> str:
    """Strip any inline reasoning block, normalise citation brackets to [ ], convert LaTeX
    \\( \\) and \\[ \\] delimiters to the $ / $$ form Streamlit renders, and bracket bare citations."""
    text = _THINK.sub("", text or "").strip()
    text = text.replace("【", "[").replace("】", "]")
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: f"$${m.group(1).strip()}$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.+?)\\\)", lambda m: f"${m.group(1).strip()}$", text, flags=re.DOTALL)
    return _bracket_bare_citations(text)


def parse_json_object(text: str) -> dict | None:
    """Parse the first JSON object in a model reply, tolerating surrounding prose or code fences."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
