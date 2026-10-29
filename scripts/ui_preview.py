"""Offline layout preview: renders a recorded answer with no API calls.

Run from the repository root:  streamlit run scripts/ui_preview.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Datasheet Copilot · preview", page_icon=":material/memory:", layout="wide")

from copilot import ui  # noqa: E402
from copilot.agent import Step  # noqa: E402
from copilot.calculator import calculate  # noqa: E402
from copilot.llm import clean  # noqa: E402
from copilot.usage import Limits  # noqa: E402

LIMITS = Limits(100_000, 300_000, 1_000_000)
CARD = {
    "part_number": "LM358B",
    "manufacturer": "Texas Instruments",
    "description": "Dual operational amplifier, 36 V, 300 µA quiescent current",
    "key_specs": [
        {"parameter": "Supply range", "value": "3 V to 36 V", "conditions": "B, BA versions", "page": 1},
        {"parameter": "Quiescent current", "value": "300 µA/ch", "conditions": "B, BA versions", "page": 1},
        {"parameter": "Unity-gain bandwidth", "value": "1.2 MHz", "conditions": "B, BA versions", "page": 1},
        {"parameter": "Input offset voltage max", "value": "3 mV", "conditions": "25 °C, A and B", "page": 1},
    ],
}
QUESTION = "What is the absolute maximum supply voltage for the LM358, and how much margin does a 24 V rail leave as a percentage?"
ANSWER = """The absolute maximum supply voltage for the LM358 is **32 V** (or ±16 V split) [lm358.pdf p.4]. A 24 V rail leaves a **25 % margin** below that limit [calculate].

**Working**

| Item | Value | Source |
|---|---|---|
| Absolute maximum supply, $V_S$ | 32 V | [lm358.pdf p.4] |
| Rail voltage | 24 V | Given |
| Margin $(32 - 24)/32$ | 25 % | [calculate] |

| Parameter | Absolute-maximum rating | Source (page) |
|---|---|---|
| Supply voltage | LM358B/LM358BA/LM2904B/LM2904BA: ±20 V or 40 V; LM158/LM258/LM358/LM158A/LM258A/LM358A/LM2904V: ±16 V or 32 V | lm358.pdf p.4 |
| Differential input | LM358B/LM358BA/LM2904B/LM2904BA/LM158/LM258/LM358/LM158A/LM258A/LM358A/LM2904V: –32 V to +32 V | lm358.pdf p.4 |

**Caveats**
- The 32 V limit applies to the total supply $(V_+ - V_-)$; split rails must not exceed ±16 V.
- Absolute maximum is a damage threshold, not an operating point. Stay within the recommended operating conditions [lm358.pdf p.5]."""
ANSWER = clean(ANSWER)  # as the agent does, so bare "doc.pdf p.N" references become citations
P4 = ("5 Specifications 5.1 Absolute Maximum Ratings over operating ambient temperature range (unless otherwise noted) "
      "MIN MAX UNIT Supply voltage, VS = ([V+] – [V–]) LM358, LM358A, LM258, LM258A, LM158, LM158A ±16 or 32 V "
      "LM2904, LM2904V ±13 or 26 V Differential input voltage, VID –32 32 V Input voltage, VI Either input –0.3 32 V")
STEPS = [
    Step("search_datasheets", {"query": "absolute maximum ratings supply voltage"}, {
        "results": [{"doc": "lm358.pdf", "page": 4, "text": P4}, {"doc": "lm358.pdf", "page": 31, "text": "Revision history: changed voltage values"}],
    }),
    Step("calculate", {}, calculate("(Vmax - Vrail)/Vmax*100", {"Vmax": "32 V", "Vrail": "24 V"})),
]

ui.inject_css()
with st.sidebar:
    ui.render_brand()
    st.html('<div class="dc-side-title">Datasheets</div>')
    st.file_uploader("Upload PDF datasheets", type="pdf", label_visibility="collapsed")
    st.text_input("Load a datasheet PDF by URL", placeholder="https://…/datasheet.pdf")
    st.button("Load from URL", icon=":material/link:", use_container_width=True)
    st.html(ui.spec_card_html(CARD, "lm358.pdf", 68, "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"))
    st.divider()
    st.html('<div class="dc-side-title">Token budget</div>')
    st.html(ui.usage_html(14_200, {"day_tokens": 47_400, "total_tokens": 47_400}, LIMITS))
    st.html(ui.models_html("nvidia/nemotron-3-super-120b-a12b", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", True))
    view = st.segmented_control("Preview state", ["Conversation", "Empty state"], default="Conversation")

ui.render_hero()
chat_col, evidence_col = st.columns([1.55, 1], gap="large")
with chat_col:
    ui.panel_title("Conversation")
    if view == "Empty state":
        ui.render_how_it_works()
        with st.container(border=True):
            left, right = st.columns([3, 2], vertical_alignment="center")
            left.markdown("**No datasheet loaded yet.** Try it on a real one.")
            right.button("Load TI LM358 op-amp", icon=":material/download:", type="primary", use_container_width=True)
        ui.panel_title("Try asking")
        cols = st.columns(2)
        for i, q in enumerate([
            "What is the gain-bandwidth product, and up to what frequency does a non-inverting gain of 20 hold?",
            "What is the absolute maximum supply voltage, and how much margin does a 24 V rail leave?",
            "List the absolute maximum ratings and the recommended operating conditions, with pages.",
            "What is the worst-case input offset voltage over temperature, and what output error does it cause at a gain of 100?",
        ]):
            cols[i % 2].button(q, key=f"q{i}", use_container_width=True)
    else:
        ui.render_turn({"role": "user", "content": QUESTION})
        ui.render_turn({"role": "assistant", "content": ANSWER, "steps": STEPS})
    st.chat_input("Ask a design question about your datasheets")
with evidence_col:
    if view == "Empty state":
        st.html(ui.empty_evidence_html())
    else:
        st.html(ui.evidence_html({"content": ANSWER, "steps": STEPS}, QUESTION))
