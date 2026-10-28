# Datasheet Copilot

A design assistant for electronics engineers and students that answers questions from component datasheets. It cites the datasheet page behind every spec and computes every number with units.

Built for the Nebius x NVIDIA Global AI Hackathon (Best Apps and Agents track). It runs on **NVIDIA Nemotron** models served by **Nebius Token Factory**.

## The problem

Answering a design question from datasheets means finding the right value in a 60-page PDF, checking whether it is typical or worst-case, and then doing the arithmetic. A general chatbot can do this, but it often recalls specs from memory and does the arithmetic in its head. That produces a confident number with no way to check it.

Datasheet Copilot is built so that each answer can be checked:

- **Specs come only from the loaded datasheets**, cited inline as `[document p.N]`. If the datasheet does not state a value, the app says so rather than guessing.
- **The model does no arithmetic itself.** Every calculation goes through a unit-aware calculator (`pint`), so it catches mistakes like adding volts to amps.
- **Every step is shown.** Each answer has a "Working" panel listing each search (with the pages it hit) and each calculation (inputs with units → result).

## What it does

- Upload datasheet PDFs, or load one by URL.
- **Spec cards**: on upload, Nemotron Nano extracts the part number and headline specs, each with its page number.
- **Agentic Q&A**: Nemotron Super plans the answer and calls these tools:
  - `search_datasheets`: BM25 search over page-tagged chunks, returns passages with document and page
  - `calculate`: safe expression evaluator with units, dB helpers, `parallel()`, and physical constants
  - `web_search` (Tavily, optional): finds manufacturer datasheet PDFs or alternative parts
  - `fetch_datasheet_pdf`: downloads a public PDF and adds it to the library (private network addresses are refused)
  - `list_datasheets`

Example questions:

- What is the -3 dB bandwidth, and what 10–90% rise time does that imply?
- At the rated responsivity, what output swing do I get into 50 Ω for 1 mW input?
- Compare the key specs of the loaded op-amps in a table, with citations.

## How Nebius and NVIDIA are used

| Component | Model / service | Why |
|---|---|---|
| Agent reasoning and tool calling | `nvidia/nemotron-3-super-120b-a12b` on Nebius Token Factory | Multi-step planning: which spec to look up, which formula applies, worst-case vs typical |
| Spec-card extraction on upload | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` on Nebius Token Factory | Fast, low-cost extraction call that runs once per document |
| Inference API | Nebius Token Factory (OpenAI-compatible) | Every agent turn and spec card is a runtime call to `api.tokenfactory.nebius.com` |
| Web search (optional) | Tavily | Finds datasheets that are not loaded yet |

Model IDs are set in `.env`. Run `python scripts/check_setup.py` to list the Nemotron models your key can access and to confirm tool calling works.

## Architecture

```
Streamlit UI (app.py)
  │  upload / URL ──► Library: pypdf text → page-tagged chunks → BM25 index
  │  spec card    ──► Nemotron Nano (Token Factory)
  │  question     ──► Agent loop (copilot/agent.py)
  │                     Nemotron Super (Token Factory) ⇄ ToolBox
  │                       search_datasheets · calculate (pint) · web_search (Tavily)
  │                       fetch_datasheet_pdf · list_datasheets
  ▼
  Answer with [doc p.N] citations + "Working" panel of every tool call
```

| Path | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `copilot/agent.py` | Tool-calling loop with a step limit and a final no-tools answer |
| `copilot/tools.py` | Tool schemas and handlers |
| `copilot/calculator.py` | AST whitelist evaluator with `pint` units |
| `copilot/library.py` | PDF extraction, chunking, BM25 search |
| `copilot/specs.py` | Spec-card extraction prompt |
| `copilot/fetch.py` | PDF download with size limit and private-address check |
| `scripts/check_setup.py` | Verifies key, model IDs and tool calling |
| `tests/` | Calculator, retrieval and agent-loop tests (no API key needed) |

## Setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/<your-username>/datasheet-copilot.git
cd datasheet-copilot
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then add NEBIUS_API_KEY (and optionally TAVILY_API_KEY)
python scripts/check_setup.py
streamlit run app.py
```

Get a Token Factory API key at [tokenfactory.nebius.com](https://tokenfactory.nebius.com). Tavily keys are at [tavily.com](https://tavily.com).

### Tests

```bash
pip install pytest
pytest
```

The tests use a stub client, so they run without an API key.

### Deploying on Streamlit Community Cloud

1. Push this repository to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), create an app from the repository with `app.py` as the entry point.
3. Under **Advanced settings → Secrets**, add:

   ```toml
   NEBIUS_API_KEY = "..."
   TAVILY_API_KEY = "..."
   ```

## Limitations

- Text extraction uses `pypdf`. Scanned datasheets with no text layer are rejected, and some tables extract with jumbled columns, so the agent may miss a value that is only in a graph or image.
- Retrieval is keyword-based (BM25). It handles part numbers and spec names well, but it can miss a paraphrase.
- Spec cards are produced by a language model. They carry page numbers so each value can be checked, but check them before relying on them.
- The library is per session and held in memory. Uploaded datasheets are not stored.

## Licence

[MIT](LICENSE)
