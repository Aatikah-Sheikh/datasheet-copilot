"""Download a datasheet PDF from a public URL, refusing private network addresses."""

import ipaddress
import socket
from urllib.parse import urlparse

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


def download_pdf(url: str, redirects_left: int = 5) -> tuple[str, bytes]:
    """Return (suggested file name, PDF bytes). Each redirect hop is re-checked."""
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
                location = requests.compat.urljoin(url, response.headers.get("Location", ""))
                return download_pdf(location, redirects_left - 1)
            response.raise_for_status()
            data = bytearray()
            for block in response.iter_content(64 * 1024):
                data.extend(block)
                if len(data) > MAX_PDF_BYTES:
                    raise FetchError("PDF is larger than 20 MB")
    except requests.RequestException as exc:
        raise FetchError(f"Download failed: {exc}") from exc
    if not bytes(data[:5]).startswith(b"%PDF"):
        raise FetchError("That URL did not return a PDF (it may be a web page linking to one)")
    name = urlparse(url).path.rsplit("/", 1)[-1] or f"{host}.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name, bytes(data)
