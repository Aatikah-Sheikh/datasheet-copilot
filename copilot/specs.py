"""Spec cards: a fast Nemotron Nano call that summarises a datasheet's headline specs on upload."""

from .library import Library
from .llm import NO_THINKING, clean, parse_json_object

PROMPT = """You extract a spec card from the first pages of an electronic component datasheet.
Return only a JSON object with these keys:
  "part_number": string or null
  "manufacturer": string or null
  "description": one short line, or null
  "key_specs": list of up to 8 objects {"parameter": str, "value": str, "conditions": str or null, "page": int}
Copy values exactly as printed, including units and min/typ/max labels. Use the [page N] markers for "page".
If something is not on these pages, use null or leave it out. Never infer or estimate a value."""

MAX_INPUT_CHARS = 12000
MAX_SPECS = 8
# Extraction needs no chain of thought (NO_THINKING): with reasoning on, Nemotron Nano can spend
# its whole output budget thinking and return empty content (seen on the AD8421 datasheet).


def extract_spec_card(client, model: str, library: Library, doc: str) -> dict | None:
    pages = library.first_pages(doc, count=3)
    if not pages:
        return None
    text = "\n\n".join(f"[page {number}]\n{body}" for number, body in pages)[:MAX_INPUT_CHARS]
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": text}],
        temperature=0,
        max_tokens=1500,
        extra_body=NO_THINKING,
    )
    card = parse_json_object(clean(response.choices[0].message.content))
    if card is not None:
        specs = card.get("key_specs")
        card["key_specs"] = specs[:MAX_SPECS] if isinstance(specs, list) else []
    return card
