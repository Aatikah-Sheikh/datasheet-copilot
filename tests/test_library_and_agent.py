import json
from types import SimpleNamespace

import pytest

from copilot.agent import Step, parse_citation, run_agent, unsupported_citations
from copilot.library import CHUNK_CHARS, Library
from copilot.tools import ToolBox


def _library() -> Library:
    library = Library()
    library.add_pages(
        "PDA05CF2.pdf",
        [
            "PDA05CF2 InGaAs amplified detector. Overview and features.",
            "Specifications. Bandwidth (-3 dB) 150 MHz. Peak responsivity 1.0 A/W at 1550 nm.",
        ],
    )
    library.add_pages("OPA847.pdf", ["Gain bandwidth product 3.9 GHz. Input offset voltage 0.1 mV."])
    return library


def test_search_returns_cited_page():
    hits = _library().search("bandwidth -3 dB")
    assert hits[0].doc == "PDA05CF2.pdf"
    assert hits[0].page == 2


def test_search_can_filter_by_document():
    hits = _library().search("gain bandwidth", doc="OPA847.pdf")
    assert hits and all(h.doc == "OPA847.pdf" for h in hits)


def test_hyphenated_words_match_their_parts():
    library = Library()
    library.add_pages("opa350.pdf", ["Features. Unity-Gain Bandwidth: 38 MHz. Rail-to-rail output."])
    assert library.search("unity gain bandwidth")[0].doc == "opa350.pdf"


def test_long_page_is_fully_chunked():
    library = Library()
    text = "x " * 1000 + "uniqueterm"
    library.add_pages("long.pdf", [text])
    assert len(text) > CHUNK_CHARS
    assert library.search("uniqueterm")[0].page == 1


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


class StubClient:
    """Replays scripted replies in the shape the OpenAI SDK returns."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        content, tool_calls = self.replies.pop(0)
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_agent_runs_tools_then_answers():
    client = StubClient(
        [
            (None, [_tool_call("1", "search_datasheets", {"query": "bandwidth"})]),
            (None, [_tool_call("2", "calculate", {"expression": "0.35/f", "variables": {"f": "150 MHz"}, "result_unit": "ns"})]),
            ("<think>scratch</think>Rise time is 2.333 ns [PDA05CF2.pdf p.2].", None),
        ]
    )
    result = run_agent(client, "test-model", ToolBox(_library()), "What is the rise time?")

    assert result.answer == "Rise time is 2.333 ns [PDA05CF2.pdf p.2]."
    assert [s.tool for s in result.steps] == ["search_datasheets", "calculate"]
    assert result.steps[1].result["display"] == "2.333 ns"
    tool_messages = [m for m in client.requests[-1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["1", "2"]


def test_agent_stops_after_max_steps():
    looping = [(None, [_tool_call(str(i), "list_datasheets", {})]) for i in range(3)]
    client = StubClient([*looping, ("Best effort answer.", None)])
    result = run_agent(client, "test-model", ToolBox(_library()), "Loop", max_steps=3)

    assert result.answer == "Best effort answer."
    assert len(result.steps) == 3
    assert "tools" not in client.requests[-1]


def test_first_call_requires_a_tool_when_datasheets_are_loaded():
    client = StubClient([
        (None, [_tool_call("1", "search_datasheets", {"query": "bandwidth"})]),
        ("Bandwidth is 150 MHz [PDA05CF2.pdf p.2].", None),
    ])
    run_agent(client, "test-model", ToolBox(_library()), "Bandwidth?")
    assert client.requests[0]["tool_choice"] == "required"
    assert client.requests[1]["tool_choice"] == "auto"


def test_answer_with_unretrieved_citations_is_sent_back_once():
    client = StubClient([
        (None, [_tool_call("1", "search_datasheets", {"query": "bandwidth", "doc": "PDA05CF2.pdf"})]),
        # Cites OPA847 p.1, which no search in this turn returned (answering from memory).
        ("PDA05CF2 is 150 MHz [PDA05CF2.pdf p.2]; OPA847 GBW is 3.9 GHz [OPA847.pdf p.1].", None),
        (None, [_tool_call("2", "search_datasheets", {"query": "gain bandwidth", "doc": "OPA847.pdf"})]),
        ("PDA05CF2 is 150 MHz [PDA05CF2.pdf p.2]; OPA847 GBW is 3.9 GHz [OPA847.pdf p.1].", None),
    ])
    result = run_agent(client, "test-model", ToolBox(_library()), "Compare bandwidths")

    nudges = [m for m in client.requests[2]["messages"] if m["role"] == "user" and "did not retrieve in this turn" in m["content"]]
    assert len(nudges) == 1 and "OPA847.pdf p.1" in nudges[0]["content"]
    assert [s.tool for s in result.steps] == ["search_datasheets", "search_datasheets"]
    assert unsupported_citations(result.answer, result.steps) == []


def test_false_calculator_claim_is_sent_back():
    client = StubClient([
        (None, [_tool_call("1", "search_datasheets", {"query": "bandwidth", "doc": "PDA05CF2.pdf"})]),
        ("Ratio is 2 (using the calculate tool) [PDA05CF2.pdf p.2].", None),
        (None, [_tool_call("2", "calculate", {"expression": "300 MHz / (150 MHz)"})]),
        ("Ratio is 2 [calculate] [PDA05CF2.pdf p.2].", None),
    ])
    result = run_agent(client, "test-model", ToolBox(_library()), "Ratio?")

    rechecks = [m for m in client.requests[2]["messages"] if m["role"] == "user" and "no calculate call" in m["content"]]
    assert len(rechecks) == 1
    assert [s.tool for s in result.steps] == ["search_datasheets", "calculate"]


@pytest.mark.parametrize(
    "inner, expected",
    [
        ("lm358.pdf p.4", ("lm358.pdf", [4])),
        ("lm2596-d.pdf p.1 & p.4", ("lm2596-d.pdf", [1, 4])),
        ("LM2596.PDF p.2, p.10", ("LM2596.PDF", [2, 10])),
        ("lm358.pdf pp. 4-5", ("lm358.pdf", [4, 5])),
        ("read_graph: lm358.pdf p.23", ("lm358.pdf", [23])),
        ("Source: lm358.pdf p.4", ("lm358.pdf", [4])),
        ("calculate", None),
        ("see note", None),
    ],
)
def test_parse_citation(inner, expected):
    assert parse_citation(inner) == expected


def test_multi_page_citation_checks_each_page_and_ignores_links():
    steps = [Step("search_datasheets", {"query": "x"}, {"results": [{"doc": "a.pdf", "page": 1, "text": ""}]})]
    answer = "Value [a.pdf p.1 & p.4]. See [the docs](https://example.com)."
    assert unsupported_citations(answer, steps) == [("a.pdf", 4)]


def test_verification_badge_counts_backed_and_unbacked_citations():
    from copilot.ui import verification_html, verification_summary

    steps = [
        Step("search_datasheets", {"query": "x"}, {"results": [{"doc": "a.pdf", "page": 1, "text": "x"}]}),
        Step("calculate", {"expression": "1+1"}, {"expression": "1+1", "display": "2"}),
        Step("calculate", {"expression": "V+I"}, {"error": "units"}),
    ]
    assert verification_summary("A [a.pdf p.1] B [a.pdf p.1 & p.4].", steps) == {"verified": 1, "unverified": 1, "calculations": 1, "graphs": 0}
    assert "not found" in verification_html("A [a.pdf p.1 & p.4].", steps)
    assert "Grounding check passed" in verification_html("A [a.pdf p.1].", steps)


def test_clean_converts_latex_and_citation_brackets():
    from copilot.llm import clean

    text = r"Margin \(\frac{8}{32}\) is 25% 【lm358.pdf p.4】" + "\n" + r"\[ f = \frac{GBW}{G} \]"
    assert clean(text) == "Margin $\\frac{8}{32}$ is 25% [lm358.pdf p.4]\n$$f = \\frac{GBW}{G}$$"


def test_clean_brackets_bare_citations_but_not_existing_ones():
    from copilot.llm import clean

    text = "| Supply | 32 V | lm358.pdf p.4 |\nGain [lm358.pdf p.10]; LDO lm2596-d.pdf p.1 & p.4."
    assert clean(text) == (
        "| Supply | 32 V | [lm358.pdf p.4] |\nGain [lm358.pdf p.10]; LDO [lm2596-d.pdf p.1 & p.4]."
    )


def test_toolbox_reports_bad_input_as_error():
    toolbox = ToolBox(_library())
    assert "error" in toolbox.call("calculate", {"expression": "V + I", "variables": {"V": "1 V", "I": "1 A"}})
    assert "error" in toolbox.call("web_search", {"query": "x"})  # no Tavily key configured
    assert "error" in toolbox.call("search_datasheets", "{not json")


def _one_page_pdf() -> bytes:
    import io

    import pypdfium2 as pdfium

    pdf, buffer = pdfium.PdfDocument.new(), io.BytesIO()
    pdf.new_page(200, 200)
    pdf.save(buffer)
    return buffer.getvalue()


def test_read_graph_sends_the_page_image_and_returns_a_cited_reading():
    from copilot.figures import read_graph

    client = StubClient([('{"figure": "Figure 5-48", "reading": "At 25 °C", "value": "≈ 36 mA", '
                          '"conditions": "VCC = 15 V", "confidence": "high"}', None)])
    result = read_graph(client, "vl-model", _one_page_pdf(), "lm358.pdf", 1, "Source current at 25 °C?")

    assert (result["doc"], result["page"], result["value"]) == ("lm358.pdf", 1, "≈ 36 mA")
    assert result["_image"][:2] == b"\xff\xd8"  # JPEG preview for the evidence panel
    content = client.requests[0]["messages"][1]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_read_graph_without_a_value_is_an_error():
    from copilot.figures import read_graph

    client = StubClient([('{"figure": null, "reading": "No graph on this page", "value": null}', None)])
    assert "error" in read_graph(client, "vl-model", _one_page_pdf(), "a.pdf", 1, "Gain at 1 MHz?")


def test_graph_reading_counts_as_evidence_and_its_image_is_not_sent_to_the_model():
    from copilot.agent import retrieved_pages
    from copilot.ui import verification_summary

    library = _library()
    library.files["PDA05CF2.pdf"] = b"%PDF"
    reading = {"doc": "PDA05CF2.pdf", "page": 2, "figure": "Figure 3", "reading": "r", "value": "≈ 1 V", "_image": b"jpeg"}
    toolbox = ToolBox(library, graph_reader=lambda data, doc, page, question: reading)
    assert "read_graph" in [t["function"]["name"] for t in toolbox.schemas()]
    assert "read_graph" not in [t["function"]["name"] for t in ToolBox(library).schemas()]
    assert "error" in toolbox.call("read_graph", {"doc": "PDA05CF2.pdf", "page": 9, "question": "q"})
    assert "error" in toolbox.call("read_graph", {"doc": "OPA847.pdf", "page": 1, "question": "q"})  # no PDF bytes
    assert "error" in toolbox.call("read_graph", {"doc": "PDA05CF2.pdf", "page": "two", "question": "q"})

    client = StubClient([
        (None, [_tool_call("1", "read_graph", {"doc": "PDA05CF2.pdf", "page": 2, "question": "swing at 10 mA"})]),
        ("Swing is ≈ 1 V [PDA05CF2.pdf p.2].", None),
    ])
    result = run_agent(client, "test-model", toolbox, "Swing?")

    assert ("PDA05CF2.pdf", 2) in retrieved_pages(result.steps)
    assert unsupported_citations(result.answer, result.steps) == []
    sent = [m["content"] for m in client.requests[1]["messages"] if m["role"] == "tool"][0]
    assert "_image" not in sent and "≈ 1 V" in sent
    assert verification_summary(result.answer, result.steps) == {"verified": 1, "unverified": 0, "calculations": 0, "graphs": 1}


def test_trim_margins_crops_blank_space_and_a_lone_footer():
    from PIL import Image, ImageDraw

    from copilot.figures import trim_margins

    page = Image.new("RGB", (1000, 1300), "white")
    draw = ImageDraw.Draw(page)
    draw.rectangle((100, 100, 900, 600), outline="black", width=3)  # the figures
    draw.line((100, 1250, 900, 1250), fill="black", width=2)  # footer, far below
    assert trim_margins(page, pad=0).size == (801, 501)


def test_chunks_are_labelled_with_their_section_so_part_variants_stay_apart():
    library = Library()
    filler = "Row of table values. " * 90  # pushes the GBW row into a later chunk than the heading
    library.add_pages("lm358.pdf", [
        "Table of Contents\n5.7 Electrical Characteristics: LM358 ..... 10",
        "5.7 Electrical Characteristics: LM358, LM358A\n" + filler + "GBW Gain bandwidth product 0.7 MHz",
        "5.9 Electrical Characteristics: LM158, LM158A\n3.53 VRMS, VS = 36 V\nGBW Gain bandwidth product 0.7 MHz",
    ])
    by_page = {c.page: c.text for c in library.chunks if "GBW" in c.text}
    assert by_page[2].startswith("[Section: 5.7 Electrical Characteristics: LM358, LM358A]")
    assert by_page[3].startswith("5.9 Electrical")  # its own heading opens the chunk: no stale label
    assert not any(c.text.startswith("[Section") for c in library.chunks if c.page == 1)


def test_calculator_claims_are_caught_through_markdown_but_denials_are_not():
    from copilot.agent import false_calculator_claim

    assert false_calculator_claim("Using the **calculate** tool: 1.2 MHz / 20 = 60 kHz", [])
    assert not false_calculator_claim("No calculator was needed.", [])
