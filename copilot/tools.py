"""Tools the agent can call. Each returns a dict; failures return {"error": ...}.
Keys starting with "_" (such as a rendered page image) are for the UI and are not sent to the model."""

import json
from collections.abc import Callable

from .calculator import CalcError, calculate
from .fetch import FetchError, download_pdf
from .library import Library

SNIPPET_CHARS = 800  # every step re-sends earlier results, so passages are the main token cost
# Tavily searches these first: an open web search for "<part> datasheet" mostly returns
# finance and jobs pages. Falls back to the open web if nothing comes back.
DATASHEET_DOMAINS = [
    "ti.com", "analog.com", "st.com", "onsemi.com", "microchip.com", "nxp.com", "infineon.com",
    "renesas.com", "rohm.com", "diodes.com", "vishay.com", "thorlabs.com", "mouser.com", "digikey.com",
]

_SEARCH = {
    "type": "function",
    "function": {
        "name": "search_datasheets",
        "description": (
            "Search the loaded datasheets. Returns passages with document name and page number "
            "to cite. Use specific terms as they appear in datasheets, e.g. 'bandwidth -3 dB', "
            "'input offset voltage max', 'absolute maximum ratings supply voltage'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords to search for."},
                "doc": {"type": "string", "description": "Optional: restrict to one document name."},
                "k": {"type": "integer", "description": "Number of passages (default 4, max 6)."},
            },
            "required": ["query"],
        },
    },
}

_CALCULATE = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate an engineering expression with units. Put every input in `variables` as a "
            "string with units, e.g. {\"R\": \"50 ohm\", \"C\": \"12 pF\"}. Operators: + - * / **. "
            "Functions: sqrt, abs, min, max, log10, ln, exp, sin, cos, tan, atan, db (power ratio), "
            "db20 (amplitude ratio), from_db, from_db20, parallel(R1, R2, ...). Constants: pi, k_B, "
            "q_e, c0, h_planck. Express gains in dB as plain numbers and convert with from_db/from_db20."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. 1/(2*pi*R*C)"},
                "variables": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Inputs with units, e.g. {\"f_3dB\": \"150 MHz\"}.",
                },
                "result_unit": {"type": "string", "description": "Unit for the answer, e.g. 'MHz', 'ns', 'mW'."},
            },
            "required": ["expression"],
        },
    },
}

_READ_GRAPH = {
    "type": "function",
    "function": {
        "name": "read_graph",
        "description": (
            "Read a value off a graph on one datasheet page, using a vision model on an image of the page. "
            "Use it for values shown only in figures, such as typical-characteristics curves. Figure captions "
            "are in the page text, so find the page with search_datasheets first (e.g. 'Figure output swing "
            "vs output current'). Readings are approximate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doc": {"type": "string", "description": "Document name."},
                "page": {"type": "integer", "description": "Page number holding the graph."},
                "question": {
                    "type": "string",
                    "description": "What to read, naming the figure, the x value and the condition, e.g. "
                    "'Figure 5-48: source current limit at 25 °C'.",
                },
            },
            "required": ["doc", "page", "question"],
        },
    },
}

_LIST = {
    "type": "function",
    "function": {
        "name": "list_datasheets",
        "description": "List the loaded datasheets and their page counts.",
        "parameters": {"type": "object", "properties": {}},
    },
}

_WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web (Tavily) to find a manufacturer datasheet PDF URL or candidate "
            "alternative parts. Searches manufacturer and distributor sites first. Results are leads, "
            "not facts: load the datasheet PDF with fetch_datasheet_pdf and cite it."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

_FETCH = {
    "type": "function",
    "function": {
        "name": "fetch_datasheet_pdf",
        "description": "Download a datasheet PDF from a public URL and add it to the library.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
}


# (pdf bytes, doc, page, question) -> reading dict; copilot.figures.read_graph with a client bound
GraphReader = Callable[[bytes, str, int, str], dict]


class ToolBox:
    def __init__(
        self, library: Library, tavily_api_key: str | None = None, graph_reader: GraphReader | None = None
    ) -> None:
        self.library = library
        self.tavily_api_key = tavily_api_key
        self.graph_reader = graph_reader
        self.added_docs: list[str] = []

    def schemas(self) -> list[dict]:
        tools = [_SEARCH, _CALCULATE, _LIST, _FETCH]
        if self.graph_reader:
            tools.append(_READ_GRAPH)
        if self.tavily_api_key:
            tools.append(_WEB_SEARCH)
        return tools

    def call(self, name: str, arguments: str | dict) -> dict:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return {"error": "Arguments were not valid JSON"}
        handler = getattr(self, f"_tool_{name}", None)
        disabled = (name == "web_search" and not self.tavily_api_key) or (name == "read_graph" and not self.graph_reader)
        if handler is None or disabled:
            return {"error": f"Unknown tool '{name}'"}
        try:
            return handler(**arguments)
        except TypeError as exc:
            return {"error": f"Bad arguments for {name}: {exc}"}
        except (CalcError, FetchError, ValueError) as exc:  # ValueError: e.g. a page number that is not a number
            return {"error": str(exc)}

    def _tool_search_datasheets(self, query: str, doc: str | None = None, k: int = 4) -> dict:
        if not self.library.docs():
            return {"error": "No datasheets are loaded yet."}
        if doc and doc not in self.library.docs():
            return {"error": f"No document named '{doc}'. Loaded: {self.library.docs()}"}
        hits = self.library.search(query, k=max(1, min(int(k), 6)), doc=doc)
        if not hits:
            return {"results": [], "note": "No matching passages. Try different keywords."}
        return {"results": [{"doc": h.doc, "page": h.page, "text": h.text[:SNIPPET_CHARS]} for h in hits]}

    def _tool_calculate(self, expression: str, variables: dict | None = None, result_unit: str | None = None) -> dict:
        return calculate(expression, variables, result_unit)

    def _tool_read_graph(self, doc: str, page: int, question: str) -> dict:
        if doc not in self.library.files:
            return {"error": f"No page images for '{doc}'. Loaded: {self.library.docs()}"}
        page = int(page)
        if not 1 <= page <= self.library.page_counts[doc]:
            return {"error": f"{doc} has pages 1 to {self.library.page_counts[doc]}."}
        return self.graph_reader(self.library.files[doc], doc, page, question)

    def _tool_list_datasheets(self) -> dict:
        return {"datasheets": [{"doc": d, "pages": n} for d, n in self.library.page_counts.items()]}

    def _tool_web_search(self, query: str) -> dict:
        from tavily import TavilyClient

        client = TavilyClient(api_key=self.tavily_api_key)
        try:
            results = client.search(query, max_results=6, include_domains=DATASHEET_DOMAINS).get("results", [])
            if not results:
                results = client.search(query, max_results=6).get("results", [])
        except Exception as exc:  # network, quota and auth errors all surface to the agent
            return {"error": f"Web search failed: {exc}"}
        return {
            "results": [
                {"title": r.get("title"), "url": r.get("url"), "snippet": (r.get("content") or "")[:300]}
                for r in results[:5]
            ]
        }

    def _tool_fetch_datasheet_pdf(self, url: str) -> dict:
        name, data = download_pdf(url)
        try:
            pages_with_text = self.library.add_pdf(name, data)
        except Exception:  # pypdf raises many types for damaged files
            return {"error": f"{name} could not be read. It may be damaged or encrypted."}
        if pages_with_text == 0:
            self.library.remove(name)
            return {"error": "The PDF has no extractable text (probably scanned)."}
        self.added_docs.append(name)
        return {"added": name, "pages": self.library.page_counts[name], "pages_with_text": pages_with_text}
