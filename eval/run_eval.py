"""Compare plain Nemotron Super with Datasheet Copilot on questions with known answers.

Both use the same model. "Plain" gets only the question; Datasheet Copilot gets the two
datasheets and its tools. Each answer is scored against eval/questions.json, and the results
are written to eval/results.json and eval/RESULTS.md.

    python eval/run_eval.py              # all questions (about 150k tokens, roughly $0.06)
    python eval/run_eval.py Q4 Q13       # rerun selected questions, keeping the other results
    python eval/run_eval.py --rescore    # re-score saved answers (no API calls)

Automatic scoring matches the expected value with regexes. Every answer was also read by hand:
eval/review.json records where that reading differs from the automatic score, and why.

Web search is off so both sides work from the same information: the copilot has only the
two loaded datasheets.
"""

import json
import re
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from copilot.agent import iter_citations, run_agent, unsupported_citations  # noqa: E402
from copilot.config import Settings  # noqa: E402
from copilot.fetch import download_pdf  # noqa: E402
from copilot.figures import read_graph  # noqa: E402
from copilot.library import Library  # noqa: E402
from copilot.llm import clean, make_client  # noqa: E402
from copilot.tools import ToolBox  # noqa: E402
from copilot.usage import Limits, MeteredClient, UsageStore  # noqa: E402

HERE = Path(__file__).resolve().parent
PLAIN_PROMPT = (
    "You are an expert electronics engineer. Answer the question concisely with specific values and units. "
    "Do any arithmetic needed and give the final number."
)


def _plain_text(answer: str) -> str:
    r"""Thin spaces to spaces, and LaTeX such as \boxed{8\ \text{V}} to 8 V, so the regexes see plain units."""
    text = answer.replace("\u2009", " ").replace("\u202f", " ").replace("\u2212", "-").replace("\\ ", " ")
    for _ in range(3):  # nested commands
        text = re.sub(r"\\(?:text|mathrm|boxed|mathbf)\{([^{}]*)\}", r"\1", text)
    return text


def correct(answer: str, question: dict) -> bool:
    text = _plain_text(answer)
    if "accept" in question:
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in question["accept"])
    spec = question["range"]
    numbers = re.findall(rf"(\d+(?:\.\d+)?)\s*{spec['unit']}(?!\s*/)", text)
    return any(spec["low"] <= float(n) <= spec["high"] for n in numbers)


def cites_expected_page(answer: str, question: dict) -> bool:
    return any(doc == question["doc"] and page in question["pages"] for _, doc, pages in iter_citations(answer) for page in pages)


def plain(client, model: str, question: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": PLAIN_PROMPT}, {"role": "user", "content": question}],
        temperature=0.2,
        max_tokens=8192,
    )
    return clean(response.choices[0].message.content)


def ask(client, library: Library, settings: Settings, session: dict, q: dict) -> dict:
    start, session["tokens"] = time.time(), 0
    plain_answer = plain(client, settings.reasoning_model, q["question"])
    plain_tokens, session["tokens"] = session["tokens"], 0
    toolbox = ToolBox(library, None, partial(read_graph, client, settings.vision_model))
    result = run_agent(client, settings.reasoning_model, toolbox, q["question"])
    row = {
        "id": q["id"], "kind": q["kind"], "question": q["question"], "expected": q["expected"],
        "plain": {"answer": plain_answer, "tokens": plain_tokens},
        "copilot": {
            "answer": result.answer,
            "unsupported_citations": [f"{d} p.{p}" for d, p in unsupported_citations(result.answer, result.steps)],
            "tools": [s.tool for s in result.steps],
            "tokens": session["tokens"],
        },
    }
    print(f"{q['id']}: {row['copilot']['tools']}  {time.time() - start:.0f}s", flush=True)
    return row


def score(row: dict, q: dict, review: dict) -> dict:
    """Automatic scores for one result row, plus the hand-checked verdicts from review.json."""
    plain_row, copilot = row["plain"], row["copilot"]
    plain_row["correct"] = correct(plain_row["answer"], q)
    copilot["correct"] = correct(copilot["answer"], q)
    copilot["cites_expected_page"] = cites_expected_page(copilot["answer"], q)
    if q["kind"] == "calculation":
        copilot["used_calculator"] = "calculate" in copilot["tools"]
    notes = review.get(q["id"], {})
    row["reviewed"] = {
        "plain": notes.get("plain", plain_row["correct"]),
        "copilot": notes.get("copilot", copilot["correct"]),
        "note": notes.get("note", ""),
    }
    return row


def main(args: list[str]) -> None:
    spec = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    review_path, saved_path = HERE / "review.json", HERE / "results.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    saved = {r["id"]: r for r in json.loads(saved_path.read_text(encoding="utf-8"))} if saved_path.exists() else {}
    settings = Settings.load()

    if args != ["--rescore"]:
        session = {"tokens": 0}
        big = 5_000_000  # the eval has its own budget; the app's caps and usage file are untouched
        store = UsageStore(Path(tempfile.mkdtemp()) / "usage.json")
        client = MeteredClient(make_client(settings), store, Limits(big, big, big), session)
        library = Library()
        for name, url in spec["datasheets"].items():
            _, data = download_pdf(url)
            library.add_pdf(name, data)
        for q in spec["questions"]:
            if not args or q["id"] in args:
                saved[q["id"]] = ask(client, library, settings, session, q)

    results = [score(saved[q["id"]], q, review) for q in spec["questions"] if q["id"] in saved]
    saved_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    text = report(results, settings)
    (HERE / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def report(results: list[dict], settings: Settings) -> str:
    n = len(results)
    calcs = [r for r in results if r["kind"] == "calculation"]
    lines = [
        "# Evaluation: plain Nemotron vs Datasheet Copilot",
        "",
        f"Both sides use `{settings.reasoning_model}`. Plain Nemotron gets only the question. Datasheet Copilot gets "
        f"the two datasheets and its tools (web search off), with graph readings by `{settings.vision_model}`. "
        "Questions and correct answers: [questions.json](questions.json). Full answers: [results.json](results.json). "
        "Every answer was also read by hand; [review.json](review.json) records where that reading differs from the "
        "automatic score, and why. One run per question; answers vary between runs.",
        "",
        "| | Plain Nemotron Super | Datasheet Copilot |",
        "|---|---|---|",
        f"| Correct answers (checked by hand) | {sum(r['reviewed']['plain'] for r in results)}/{n} | {sum(r['reviewed']['copilot'] for r in results)}/{n} |",
        f"| Correct answers (automatic) | {sum(r['plain']['correct'] for r in results)}/{n} | {sum(r['copilot']['correct'] for r in results)}/{n} |",
        f"| Cites a page that states the expected value | n/a | {sum(r['copilot']['cites_expected_page'] for r in results)}/{n} |",
        f"| Every citation on a page retrieved in that turn | n/a | {sum(not r['copilot']['unsupported_citations'] for r in results)}/{n} |",
        f"| Calculation questions computed with the calculator | n/a | {sum(r['copilot']['used_calculator'] for r in calcs)}/{len(calcs)} |",
        f"| Tokens used | {sum(r['plain']['tokens'] for r in results):,} | {sum(r['copilot']['tokens'] for r in results):,} |",
        "",
        "| Question | Expected | Plain | Copilot | Note |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} ({r['kind']}): {r['question']} | {r['expected']} | {_mark(r['reviewed']['plain'])} | "
            f"{_mark(r['reviewed']['copilot'])} | {r['reviewed']['note']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main(sys.argv[1:])
