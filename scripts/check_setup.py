"""Check the Token Factory key, list available Nemotron models, and confirm tool calling works.

Run from the repository root:  python scripts/check_setup.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import APIError  # noqa: E402

from copilot.config import Settings  # noqa: E402
from copilot.llm import make_client  # noqa: E402
from copilot.tools import _CALCULATE  # noqa: E402


def main() -> int:
    settings = Settings.load()
    try:
        client = make_client(settings)
        model_ids = sorted(m.id for m in client.models.list())
    except (RuntimeError, APIError) as exc:
        print(f"FAIL: {exc}")
        return 1

    nvidia = [m for m in model_ids if "nemotron" in m.lower() or m.lower().startswith("nvidia/")]
    print(f"OK: key works, {len(model_ids)} models visible.")
    print("NVIDIA / Nemotron models:")
    for model_id in nvidia:
        print(f"  {model_id}")

    ok = True
    for label, model in (("reasoning", settings.reasoning_model), ("fast", settings.fast_model)):
        found = model in model_ids
        ok &= found
        print(f"{'OK' if found else 'MISSING'}: {label} model {model}")
    if not ok:
        print("Set NEBIUS_REASONING_MODEL / NEBIUS_FAST_MODEL in .env to IDs from the list above.")
        return 1

    response = client.chat.completions.create(
        model=settings.reasoning_model,
        messages=[{"role": "user", "content": "Use the calculate tool to compute 1/(2*pi*R*C) for R = 1 kohm, C = 1 nF, in kHz."}],
        tools=[_CALCULATE],
        tool_choice="auto",
        max_tokens=2048,
    )
    calls = response.choices[0].message.tool_calls or []
    if not calls:
        print("FAIL: the reasoning model did not make a tool call. Try another Nemotron model.")
        return 1
    print(f"OK: tool calling works: {calls[0].function.name}({json.dumps(json.loads(calls[0].function.arguments))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
