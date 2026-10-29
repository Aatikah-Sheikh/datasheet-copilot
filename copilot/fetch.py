"""Download a datasheet PDF from a public URL, refusing private network addresses."""

import ipaddress
import re
import socket
from urllib.parse import urljoin, urlparse

import requests

MAX_PDF_BYTES = 20 * 1024 * 1024
TIMEOUT_S = 20


class FetchError(ValueError):
    pass


def _check_public(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise FetchError("Only http(s) URLs are supported")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, None)}
    except socket.gaierror as exc:
        raise FetchError(f"Could not resolve {parsed.hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise FetchError("URLs on private or local networks are not allowed")
    return parsed.hostname


_HREF = re.compile(r"""href\s*=\s*["']([^"'#]+)["']""", re.IGNORECASE)


def _datasheet_links(page_url: str, html: str) -> list[str]:
    """Likely datasheet PDF links on a product page, best first."""
    links = []
    for href in _HREF.findall(html):
        absolute = urljoin(page_url, href.strip())
        lower = absolute.lower()
        if lower.startswith("http") and (".pdf" in lower or "/lit/ds/" in lower or "/lit/gpn/" in lower):
            links.append(absolute)
    ranked = sorted(dict.fromkeys(links), key=lambda u: not ("datasheet" in u.lower() or "/ds/" in u.lower() or "/gpn/" in u.lower()))
    return ranked[:3]


def download_pdf(url: str, redirects_left: int = 5, follow_page: bool = True) -> tuple[str, bytes]:
    """Return (suggested file name, PDF bytes). Each redirect hop is re-checked. If the URL is a
    product web page, follow up to three datasheet links found on it (one hop, no deeper)."""
    host = _check_public(url)
    try:
        with requests.get(
            url,
            stream=True,
            timeout=TIMEOUT_S,
            allow_redirects=False,
            headers={"User-Agent": "DatasheetCopilot/1.0"},
        ) as response:
            if response.is_redirect:
                if redirects_left <= 0:
                    raise FetchError("Too many redirects")
                location = urljoin(url, response.headers.get("Location", ""))
                return download_pdf(location, redirects_left - 1, follow_page)
            response.raise_for_status()
            data = bytearray()
            for block in response.iter_content(64 * 1024):
                data.extend(block)
                if len(data) > MAX_PDF_BYTES:
                    raise FetchError("PDF is larger than 20 MB")
    except requests.RequestException as exc:
        raise FetchError(f"Download failed: {exc}") from exc
    if not bytes(data[:5]).startswith(b"%PDF"):
        if follow_page:
            html = bytes(data).decode("utf-8", errors="replace")
            for link in _datasheet_links(url, html):
                try:
                    return download_pdf(link, follow_page=False)
                except FetchError:
                    continue
        raise FetchError("That URL did not return a PDF, and no datasheet PDF link was found on the page")
    name = urlparse(url).path.rsplit("/", 1)[-1] or f"{host}.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name, bytes(data)
