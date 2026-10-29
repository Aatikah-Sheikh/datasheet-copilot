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

    nudges = [m for m in client.requests[2]["messages"] if m["role"] == "user" and "did not retrieve" in m["content"]]
    assert len(nudges) == 1 and "OPA847.pdf p.1" in nudges[0]["content"]
    assert [s.tool for s in result.steps] == ["search_datasheets", "search_datasheets"]
    assert unsupported_citations(result.answer, result.steps) == []


@pytest.mark.parametrize(
    "inner, expected",
    [
        ("lm358.pdf p.4", ("lm358.pdf", [4])),
        ("lm2596-d.pdf p.1 & p.4", ("lm2596-d.pdf", [1, 4])),
        ("LM2596.PDF p.2, p.10", ("LM2596.PDF", [2, 10])),
        ("lm358.pdf pp. 4-5", ("lm358.pdf", [4, 5])),
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


def test_clean_converts_latex_and_citation_brackets():
    from copilot.llm import clean

    text = r"Margin \(\frac{8}{32}\) is 25% 【lm358.pdf p.4】" + "\n" + r"\[ f = \frac{GBW}{G} \]"
    assert clean(text) == "Margin $\\frac{8}{32}$ is 25% [lm358.pdf p.4]\n$$f = \\frac{GBW}{G}$$"


def test_toolbox_reports_bad_input_as_error():
    toolbox = ToolBox(_library())
    assert "error" in toolbox.call("calculate", {"expression": "V + I", "variables": {"V": "1 V", "I": "1 A"}})
    assert "error" in toolbox.call("web_search", {"query": "x"})  # no Tavily key configured
    assert "error" in toolbox.call("search_datasheets", "{not json")
