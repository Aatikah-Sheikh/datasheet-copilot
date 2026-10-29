# Datasheet Copilot

A design assistant for electronics engineers and students that answers questions from component datasheets. It cites the datasheet page behind every spec and computes every number with units.

Built for the Nebius x NVIDIA Global AI Hackathon (Best Apps and Agents track). It runs on **NVIDIA Nemotron** open models served by **Nebius Token Factory**.

- **Live demo:** https://datasheet-copilot.streamlit.app/ (if it has been idle, Streamlit takes about 30 seconds to wake it)
- **Licence:** [MIT](LICENSE) (OSI-approved)

![Datasheet Copilot answering a comparison: the cited answer on the left, and on the right the evidence panel with the datasheet passages, the calculation and the agent trace (web search, datasheet download, searches, calculation)](docs/answer-and-evidence.jpg)

*Asked to compare the TI OPA350 with the loaded LM358, the agent found the OPA350 datasheet with Tavily, downloaded it, searched both, and computed the ratio with the calculator. Every value carries a page citation, and the evidence panel shows the passage behind each one.*

## The problem

Answering a design question from datasheets means finding the right value in a 60-page PDF, checking whether it is typical or worst-case, and then doing the arithmetic. A general chatbot can do this, but it often recalls specs from memory and does the arithmetic in its head. That produces a confident number with no way to check it.

Datasheet Copilot is built so that each answer can be checked:

- **Specs come only from the loaded datasheets**, cited inline as `[document p.N]`. If the datasheet does not state a value, the app says so rather than guessing.
- **Values that are only in graphs are read too.** Many specs appear only as typical-characteristics curves. A third Nemotron model, the vision-language Nemotron 3.5 Super VL, reads the value off the graph on the cited page, and the evidence panel shows the page it read so you can check it.
- **The model does no arithmetic itself.** Every calculation goes through a unit-aware calculator (`pint`), so it catches mistakes like adding volts to amps.
- **Every step is shown.** The evidence panel beside each answer shows the exact passage behind each citation, and a trace of every search (with the pages it hit), web lookup, download and calculation (inputs with units → result).
- **The app checks the model.** After each answer, every `[doc p.N]` citation is compared with the pages actually retrieved in that turn, and any claim of using the calculator is compared with the calculate calls actually made. An answer that fails is sent back once to verify or remove the claim, and a badge under every answer shows the result: how many cited pages were matched to retrieved passages, or which were not.

## Measured: plain Nemotron vs Datasheet Copilot

Fourteen questions about TI's LM358 and OPA350 datasheets, with correct answers read by hand from the PDFs: 8 lookups, 4 calculations and 2 values that appear only in graphs. Both sides use the same model, Nemotron 3 Super. Plain Nemotron gets only the question; Datasheet Copilot gets the two datasheets and its tools. Every answer was read by hand as well as scored automatically.

| | Plain Nemotron 3 Super | Datasheet Copilot |
|---|---|---|
| Correct answers | **4/14** | **14/14** |
| Cites a page that states the value | n/a | 13/14 |
| Every citation on a page retrieved in that turn | n/a | 14/14 |
| Calculation questions computed with the calculator | n/a | 4/4 |
| Tokens used (all 14 questions) | 9,551 | 133,193 (about $0.05) |

The model's mistakes without the datasheet are the ones that matter in a design: the older LM358's 32 V rating given for the LM358B (40 V), the typical offset voltage given as the maximum, a 25 °C limit used for a 0 °C to 70 °C question, and 10 MHz for an op-amp with a 38 MHz gain-bandwidth product. The same model, given the datasheets and tools, got all fourteen right.

The evaluation also found two bugs, both now fixed: the model once said it used the calculator without calling it, and the check missed it because of Markdown bold; and it cited the right value from the wrong part's table, which led to section labels on passages. Details, every answer and how to rerun it: [eval/RESULTS.md](eval/RESULTS.md) (`python eval/run_eval.py`). It is one run per question, and answers vary between runs.

## What it does

- Upload datasheet PDFs, or load one by URL.
- **Spec cards**: on upload, Nemotron Nano extracts the part number and headline specs, each with its page number.
- **Agentic Q&A**: Nemotron Super plans the answer and calls these tools:
  - `search_datasheets`: BM25 search over page-tagged chunks, returns passages with document and page
  - `read_graph`: renders a datasheet page and asks Nemotron 3.5 Super VL to read a value off its graphs (figure, curve conditions, approximate value, confidence)
  - `calculate`: safe expression evaluator with units, dB helpers, `parallel()`, and physical constants
  - `web_search` (Tavily, optional): finds manufacturer datasheet PDFs or alternative parts
  - `fetch_datasheet_pdf`: downloads a public PDF and adds it to the library (private network addresses are refused)
  - `list_datasheets`

Example questions:

- What is the -3 dB bandwidth, and what 10–90% rise time does that imply?
- On a 15 V supply, what is the largest output swing the LM358 can deliver at 20 kHz? (answered from a graph)
- At the rated responsivity, what output swing do I get into 50 Ω for 1 mW input?
- Compare the key specs of the loaded op-amps in a table, with citations.

## How NVIDIA Nemotron and Nebius are used

| Component | Model / service | Why |
|---|---|---|
| Agent reasoning and tool calling | NVIDIA **Nemotron 3 Super** (`nvidia/nemotron-3-super-120b-a12b`) on Nebius Token Factory | Multi-step planning: which spec to look up, which formula applies, worst-case vs typical |
| Reading values off graphs | NVIDIA **Nemotron 3.5 Super VL** (`nvidia/Nemotron-3.5-Super-VL`) on Nebius Token Factory | Vision-language model: reads typical-characteristics curves, including log axes, from an image of the page |
| Spec-card extraction on upload | NVIDIA **Nemotron 3 Nano** (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`) on Nebius Token Factory | Fast, low-cost structured extraction that runs once per document |
| Inference API | **Nebius Token Factory** (OpenAI-compatible) | Every agent turn and spec card is a runtime call to `api.tokenfactory.nebius.com` |
| Web search (optional) | Tavily | Finds datasheets that are not loaded yet |

### How Nemotron is used

- **Nemotron 3 Super is the agent.** `copilot/agent.py` runs a tool-calling loop in which Super chooses which tools to call: it can request several datasheet searches in one step, fetch a missing datasheet, then call the calculator with the values it found. Its answer is checked against the tool results before it is shown (see above).
- **Tool use is enforced on the first step.** The first call of each turn uses `tool_choice="required"`, so Super has to search before it answers. With tool use left optional, it answered a follow-up comparison from memory with citations to pages it had never retrieved.
- **Nemotron 3 Nano reads every new datasheet.** `copilot/specs.py` sends the first pages to Nano for a JSON spec card at temperature 0. Reasoning is turned off for this call (`chat_template_kwargs: {"enable_thinking": false}`): with it on, Nano could spend the whole output budget thinking and return nothing.
- **Nemotron 3.5 Super VL reads graphs, with reasoning on.** `copilot/figures.py` renders the page with PDFium, trims the blank margins and sends it with the question. With reasoning off, it misread a log frequency axis on the LM358 (10 V at 20 kHz where the graph shows about 4.5 V); with reasoning on, it read about 5 V. One reading costs about 5,000 tokens.
- **Reasoning is also turned off for the agent's forced final answer.** That call is made when the loop ends without an answer (step limit reached or an empty reply), so it should answer from the results it already has, not think again.

### Where Token Factory accelerated the build

- **No infrastructure.** All three Nemotron models are served from one endpoint with one API key, so there were no GPUs to provision or model servers to run. The app is a thin Streamlit front end that deploys free on Streamlit Community Cloud.
- **Standard SDK.** Token Factory is OpenAI-compatible, so the app uses the official `openai` Python package with only `base_url` changed (`copilot/llm.py`). Tool calling, `tool_choice`, JSON output and image input worked with the standard request format.
- **Three models behind one key.** Adding graph reading meant listing the models the key could use (`/v1/models`), finding Nemotron 3.5 Super VL there, and calling it with the same client. No new account, endpoint or SDK.
- **Right-sized models through configuration.** Moving between Nemotron sizes is a change to one environment variable, which made it quick to try Super and Nano on each job and keep the cheaper model where it was good enough.
- **Fast setup checks.** `scripts/check_setup.py` lists the models available to your key via `/v1/models` and makes a test tool call, so a wrong model ID or key shows up before the app runs.
- **Controlled spend.** Every request is routed through a metered client (`copilot/usage.py`) that records the token usage Token Factory returns and refuses calls beyond per-session, per-day and all-time caps.

Other Nebius services: none. Inference is the only Nebius service this project uses.

## Architecture

```
Streamlit UI (app.py)
  │  upload / URL ──► Library: pypdf text → page-tagged chunks → BM25 index
  │  spec card    ──► Nemotron Nano (Token Factory)
  │  graph page   ──► PDFium render ──► Nemotron 3.5 Super VL (Token Factory)
  │  question     ──► Agent loop (copilot/agent.py)
  │                     Nemotron Super (Token Factory) ⇄ ToolBox
  │                       search_datasheets · read_graph · calculate (pint)
  │                       web_search (Tavily) · fetch_datasheet_pdf · list_datasheets
  │                     grounding check: cited pages retrieved? calculator claims real?
  ▼
  Answer with [doc p.N] citations + evidence panel (passages and tool trace)
```

| Path | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `copilot/agent.py` | Tool-calling loop with a step limit, grounding checks and a final no-tools answer |
| `copilot/llm.py` | Token Factory client and output clean-up |
| `copilot/usage.py` | Metered client and token caps |
| `copilot/tools.py` | Tool schemas and handlers |
| `copilot/calculator.py` | AST whitelist evaluator with `pint` units |
| `copilot/library.py` | PDF extraction, chunking, BM25 search |
| `copilot/specs.py` | Spec-card extraction prompt |
| `copilot/figures.py` | Page rendering and graph reading with Nemotron 3.5 Super VL |
| `copilot/fetch.py` | PDF download with size limit and private-address check |
| `scripts/check_setup.py` | Verifies key, model IDs and tool calling |
| `tests/` | Calculator, retrieval, citation, graph-reading and agent-loop tests (no API key needed) |
| `eval/` | Plain Nemotron vs Datasheet Copilot on questions with known answers |

## Setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Aatikah-Sheikh/datasheet-copilot.git
cd datasheet-copilot
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then add NEBIUS_API_KEY (and optionally TAVILY_API_KEY)
python scripts/check_setup.py
streamlit run app.py
```

Get a Token Factory API key at [tokenfactory.nebius.com](https://tokenfactory.nebius.com). Tavily keys are at [tavily.com](https://tavily.com).

Open http://localhost:8501 and click **Load TI LM358 op-amp** to try it without uploading anything.

Optional settings in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `MAX_TOKENS_PER_SESSION` | 100000 | Token cap for one browser session |
| `MAX_TOKENS_PER_DAY` | 300000 | Token cap per UTC day, across all sessions |
| `MAX_TOKENS_TOTAL` | 1000000 | All-time token cap |
| `NEBIUS_REASONING_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | Agent model |
| `NEBIUS_FAST_MODEL` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | Spec-card model |
| `NEBIUS_VISION_MODEL` | `nvidia/Nemotron-3.5-Super-VL` | Graph-reading model |

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

- Text extraction uses `pypdf`. Scanned datasheets with no text layer are rejected, and some tables extract with jumbled columns.
- Graph readings are approximate, and they come from typical curves, not guaranteed limits. The app labels them that way and shows the page that was read.
- Retrieval is keyword-based (BM25). It handles part numbers and spec names well, but it can miss a paraphrase. Each passage is labelled with its section heading (for example `[Section: 5.7 Electrical Characteristics: LM358, LM358A]`), because many datasheets cover several part variants in separate tables.
- Spec cards are produced by a language model. They carry page numbers so each value can be checked, but check them before relying on them.
- The library is per session and held in memory. Uploaded datasheets are not stored.

- The grounding check catches citations to pages that were never retrieved and false claims of using the calculator. It does not yet catch arithmetic the model does silently in its prose.

## Licence

[MIT](LICENSE), an OSI-approved open-source licence.
