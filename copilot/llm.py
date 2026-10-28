"""Nebius Token Factory client (OpenAI-compatible API)."""

import json
import re

from openai import OpenAI

from .config import Settings

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def make_client(settings: Settings) -> OpenAI:
    if not settings.api_key:
        raise RuntimeError("NEBIUS_API_KEY is not set. Add it to .env or Streamlit secrets.")
    return OpenAI(base_url=settings.base_url, api_key=settings.api_key)


def clean(text: str | None) -> str:
    """Strip any inline reasoning block, normalise citation brackets to [ ], and convert
    LaTeX \\( \\) and \\[ \\] delimiters to the $ / $$ form Streamlit renders."""
    text = _THINK.sub("", text or "").strip()
    text = text.replace("【", "[").replace("】", "]")
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: f"$${m.group(1).strip()}$$", text, flags=re.DOTALL)
    return re.sub(r"\\\((.+?)\\\)", lambda m: f"${m.group(1).strip()}$", text, flags=re.DOTALL)


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
