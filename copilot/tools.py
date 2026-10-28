"""Tools the agent can call. Each returns a JSON-serialisable dict; failures return {"error": ...}."""

import json

from .calculator import CalcError, calculate
from .fetch import FetchError, download_pdf
from .library import Library

SNIPPET_CHARS = 1200

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
                "k": {"type": "integer", "description": "Number of passages (default 5, max 8)."},
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
            "alternative parts. Results are leads, not facts: load the datasheet and cite it."
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


class ToolBox:
    def __init__(self, library: Library, tavily_api_key: str | None = None) -> None:
        self.library = library
        self.tavily_api_key = tavily_api_key
        self.added_docs: list[str] = []

    def schemas(self) -> list[dict]:
        tools = [_SEARCH, _CALCULATE, _LIST, _FETCH]
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
        if handler is None or (name == "web_search" and not self.tavily_api_key):
            return {"error": f"Unknown tool '{name}'"}
        try:
            return handler(**arguments)
        except TypeError as exc:
            return {"error": f"Bad arguments for {name}: {exc}"}
        except (CalcError, FetchError) as exc:
            return {"error": str(exc)}

    def _tool_search_datasheets(self, query: str, doc: str | None = None, k: int = 5) -> dict:
        if not self.library.docs():
            return {"error": "No datasheets are loaded yet."}
        if doc and doc not in self.library.docs():
            return {"error": f"No document named '{doc}'. Loaded: {self.library.docs()}"}
        hits = self.library.search(query, k=max(1, min(int(k), 8)), doc=doc)
        if not hits:
            return {"results": [], "note": "No matching passages. Try different keywords."}
        return {"results": [{"doc": h.doc, "page": h.page, "text": h.text[:SNIPPET_CHARS]} for h in hits]}

    def _tool_calculate(self, expression: str, variables: dict | None = None, result_unit: str | None = None) -> dict:
        return calculate(expression, variables, result_unit)

    def _tool_list_datasheets(self) -> dict:
        return {"datasheets": [{"doc": d, "pages": n} for d, n in self.library.page_counts.items()]}

    def _tool_web_search(self, query: str) -> dict:
        from tavily import TavilyClient

        try:
            response = TavilyClient(api_key=self.tavily_api_key).search(query, max_results=5)
        except Exception as exc:  # network, quota and auth errors all surface to the agent
            return {"error": f"Web search failed: {exc}"}
        return {
            "results": [
                {"title": r.get("title"), "url": r.get("url"), "snippet": (r.get("content") or "")[:400]}
                for r in response.get("results", [])
            ]
        }

    def _tool_fetch_datasheet_pdf(self, url: str) -> dict:
        name, data = download_pdf(url)
        pages_with_text = self.library.add_pdf(name, data)
        if pages_with_text == 0:
            self.library.remove(name)
            return {"error": "The PDF has no extractable text (probably scanned)."}
        self.added_docs.append(name)
        return {"added": name, "pages": self.library.page_counts[name], "pages_with_text": pages_with_text}
