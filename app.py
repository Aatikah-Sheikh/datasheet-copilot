"""Datasheet Copilot: Streamlit front end."""

import hashlib
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Datasheet Copilot", page_icon=":material/memory:", layout="wide")


def _load_streamlit_secrets() -> list[str]:
    """On Streamlit Community Cloud, keys live in st.secrets; expose top-level values as env vars.
    Returns the secret names found (names only, never values) for the setup message."""
    try:
        names = []
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                os.environ.setdefault(key.strip(), str(value).strip())
                names.append(key)
        return names
    except Exception:  # no secrets.toml when running locally with .env
        return []


SECRET_NAMES = _load_streamlit_secrets()

from openai import APIError  # noqa: E402

from copilot import ui  # noqa: E402
from copilot.agent import Step, run_agent  # noqa: E402
from copilot.config import Settings  # noqa: E402
from copilot.fetch import FetchError, download_pdf  # noqa: E402
from copilot.library import Library, extract_pages  # noqa: E402
from copilot.llm import make_client  # noqa: E402
from copilot.specs import extract_spec_card  # noqa: E402
from copilot.tools import ToolBox  # noqa: E402
from copilot.usage import BudgetExceeded, Limits, MeteredClient, UsageStore  # noqa: E402

SAMPLE = ("TI LM358 op-amp", "https://www.ti.com/lit/ds/symlink/lm358.pdf")
EXAMPLES = [
    "What is the gain-bandwidth product, and up to what frequency does a non-inverting gain of 20 hold?",
    "What is the absolute maximum supply voltage, and how much margin does a 24 V rail leave?",
    "List the absolute maximum ratings and the recommended operating conditions, with pages.",
    "What is the worst-case input offset voltage over temperature, and what output error does it cause at a gain of 100?",
]

settings = Settings.load()
LIMITS = Limits(settings.max_tokens_per_session, settings.max_tokens_per_day, settings.max_tokens_total)
MAX_AGENT_STEPS = 8  # room for a comparison across two datasheets plus one citation recheck


@st.cache_resource
def _client(api_key: str, base_url: str):
    return make_client(settings)


@st.cache_resource
def _usage_store() -> UsageStore:
    """Shared by every session in this process, so the day and all-time caps are app-wide."""
    return UsageStore(Path(__file__).resolve().parent / "usage.json")


@st.cache_data(ttl="1d", max_entries=32, show_spinner=False)
def _download(url: str) -> tuple[str, bytes]:
    """Shared across sessions, so the sample loads instantly for every visitor after the first.
    Failed downloads raise and are not cached."""
    return download_pdf(url)


@st.cache_data(ttl="1d", max_entries=32, show_spinner=False)
def _pages(data: bytes) -> list[str]:
    """PDF text extraction takes ~10 s for a 68-page datasheet; cache it by file content."""
    return extract_pages(data)


class _NoCard(Exception):
    """Raised inside the cached function so an empty spec card is retried rather than cached."""


@st.cache_data(ttl="7d", max_entries=64, show_spinner=False)
def _cached_card(name: str, digest: str, _client, _library) -> dict:
    """Spec cards keyed by document name and text, so the same datasheet costs Nano tokens once.
    Arguments starting with _ are not part of the cache key."""
    card = extract_spec_card(_client, settings.fast_model, _library, name)
    if not card:
        raise _NoCard(name)
    return card


def _state():
    s = st.session_state
    if "library" not in s:
        s.library = Library()
        s.cards = {}
        s.chat = []  # {"role", "content", "steps"}
        s.seen_uploads = set()
        s.usage = {"tokens": 0}
    return s


def _add_document(s, client, name: str, data: bytes) -> None:
    pages_with_text = s.library.add_pages(name, _pages(data))
    if pages_with_text == 0:
        s.library.remove(name)
        st.sidebar.error(f"{name}: no extractable text. It is probably a scanned PDF.")
        return
    _make_card(s, client, name)


def _load_url(s, client, url: str) -> None:
    try:
        with st.spinner("Downloading datasheet..."):
            name, data = _download(url.strip())
        _add_document(s, client, name, data)
    except FetchError as exc:
        st.sidebar.error(str(exc))


def _make_card(s, client, name: str) -> None:
    if client is None or name in s.cards:
        return
    try:
        text = "".join(body for _, body in s.library.first_pages(name, count=3))
        with st.spinner(f"Reading {name} with Nemotron Nano..."):
            s.cards[name] = _cached_card(name, hashlib.sha256(text.encode()).hexdigest(), client, s.library)
    except _NoCard:
        s.cards[name] = None
    except APIError as exc:
        s.cards[name] = None
        st.sidebar.warning(f"Spec card for {name} failed: {exc.message}")
    except BudgetExceeded as exc:
        s.cards[name] = None
        st.sidebar.warning(f"Spec card skipped: {exc}")


def _sidebar(s, client) -> None:
    with st.sidebar:
        ui.render_brand()
        st.html('<div class="dc-side-title">Datasheets</div>')
        uploads = st.file_uploader("Upload PDF datasheets", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
        for upload in uploads or []:
            key = (upload.name, upload.size)
            if key not in s.seen_uploads:
                s.seen_uploads.add(key)
                _add_document(s, client, upload.name, upload.getvalue())

        with st.form("by_url", clear_on_submit=True, border=False):
            url = st.text_input("Load a datasheet PDF by URL", placeholder="https://…/datasheet.pdf")
            if st.form_submit_button("Load from URL", icon=":material/link:", use_container_width=True) and url:
                _load_url(s, client, url)

        for doc, pages in s.library.page_counts.items():
            st.html(ui.spec_card_html(s.cards.get(doc), doc, pages, settings.fast_model))

        st.divider()
        st.html('<div class="dc-side-title">Token budget</div>')
        st.html(ui.usage_html(s.usage["tokens"], _usage_store().snapshot(), LIMITS))
        st.html(ui.models_html(settings.reasoning_model, settings.fast_model, bool(settings.tavily_api_key)))
        if s.chat and st.button("Clear conversation", icon=":material/restart_alt:", use_container_width=True):
            s.chat = []
            st.rerun()


def _empty_state(s, client) -> None:
    ui.render_how_it_works()
    if not s.library.docs():
        with st.container(border=True):
            left, right = st.columns([3, 2], vertical_alignment="center")
            left.markdown("**No datasheet loaded yet.** Try it on a real one.")
            if right.button(f"Load {SAMPLE[0]}", icon=":material/download:", type="primary", use_container_width=True, disabled=client is None):
                _load_url(s, client, SAMPLE[1])
                st.rerun()
    ui.panel_title("Try asking")
    columns = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if columns[i % 2].button(example, key=f"example_{i}", use_container_width=True, disabled=client is None):
            s.pending = example
            st.rerun()


def _evidence_panel(s) -> None:
    answers = [t for t in s.chat if t["role"] == "assistant"]
    if not answers:
        st.html(ui.empty_evidence_html())
        return
    index = len(answers) - 1
    if len(answers) > 1:
        index = st.selectbox(
            "Evidence for", range(len(answers)), index=index, label_visibility="collapsed",
            format_func=lambda i: f"Q{i + 1}: {answers[i].get('question', '')[:70]}",
        )
    st.html(ui.evidence_html(answers[index], answers[index].get("question", "")))


def main() -> None:
    s = _state()
    client = None
    if settings.api_key:
        client = MeteredClient(_client(settings.api_key, settings.base_url), _usage_store(), LIMITS, s.usage)

    ui.inject_css()
    ui.render_hero()
    if client is None:
        found = ", ".join(f"`{n}`" for n in SECRET_NAMES) or "none"
        st.error(
            "NEBIUS_API_KEY is not set. Add it to `.env` (local) or the app's secrets (hosted: "
            f"Manage app → Settings → Secrets, then Reboot). Secrets found: {found}."
        )

    _sidebar(s, client)

    chat_col, evidence_col = st.columns([1.55, 1], gap="large")
    with evidence_col:
        _evidence_panel(s)
    with chat_col:
        ui.panel_title("Conversation")
        for turn in s.chat:
            ui.render_turn(turn)
        live = st.empty()  # holds the empty state, then the in-progress turn, above the input box
        if not s.chat:
            with live.container():
                _empty_state(s, client)
        question = st.chat_input("Ask a design question about your datasheets", disabled=client is None) or s.pop("pending", None)
    if not question or client is None:
        return

    s.chat.append({"role": "user", "content": question})
    history = [{"role": t["role"], "content": t["content"]} for t in s.chat[:-1]][-8:]
    toolbox = ToolBox(s.library, settings.tavily_api_key)
    turn_area = live.container()  # replaces the empty state, if it was showing
    with turn_area:
        ui.render_turn(s.chat[-1])
    with turn_area, st.chat_message("assistant", avatar=":material/memory:"):
        status = st.status("Planning with Nemotron Super...", expanded=True)

        def on_step(step: Step) -> None:
            status.update(label=f"Working: {step.tool.replace('_', ' ')}...")
            with status:
                st.html(ui.live_step_html(step))

        try:
            result = run_agent(
                client, settings.reasoning_model, toolbox, question, history,
                max_steps=MAX_AGENT_STEPS, on_step=on_step,
            )
        except APIError as exc:
            status.update(label="Request failed", state="error")
            st.error(f"Token Factory request failed: {exc.message}")
            s.chat.pop()
            return
        except BudgetExceeded as exc:
            status.update(label="Token cap reached", state="error")
            st.warning(str(exc))
            s.chat.pop()
            return
        status.update(label="Done", state="complete", expanded=False)

    s.chat.append({
        "role": "assistant",
        "content": result.answer or "_The model returned an empty answer._",
        "steps": result.steps,
        "question": question,
    })
    for doc in toolbox.added_docs:
        _make_card(s, client, doc)
    st.rerun()  # redraw so the answer, sidebar documents and token counts are all current


main()
