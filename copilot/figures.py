"""Graph reading: renders a datasheet page and asks Nemotron VL to read a value off its figures.

Many datasheet specs appear only as typical-characteristics curves (output swing against load,
gain against frequency), which text extraction cannot see.
"""

import base64
import io
import threading

import numpy as np
import pypdfium2 as pdfium
from openai import APIError
from PIL import Image, ImageChops

from .llm import clean, parse_json_object

RENDER_DPI = 150  # a letter page at 1275 x 1650 px: axis labels and grid lines stay legible
PREVIEW_WIDTH = 900  # the evidence-panel copy is downscaled JPEG, to keep the session small
FIELDS = ("figure", "reading", "value", "conditions", "confidence")
# PDFium is not thread-safe, and Streamlit runs each browser session in its own thread.
_PDFIUM = threading.Lock()

PROMPT = """You read values from graphs in electronic component datasheets.
The image is one datasheet page. Answer the question from the figure that shows the quantity asked for.
Find the x value on the axis first (check whether the axis is logarithmic), follow the curve for the
stated condition (supply voltage, temperature, load), and read the y value at that point.
Return only a JSON object with these keys:
  "figure": figure number and title as printed, or null
  "reading": one sentence saying what was read, at which x value and on which curve
  "value": the value read, with units and "≈", or null if the page has no graph that answers it
  "conditions": the curve's conditions as printed (e.g. "VCC = 15 V, TA = 25 °C"), or null
  "confidence": "high", "medium" or "low"
Never use a value that is not on this page's graphs."""


def render_page(data: bytes, page: int, dpi: int = RENDER_DPI):
    """PIL image of a 1-based page of a PDF."""
    with _PDFIUM:
        pdf = pdfium.PdfDocument(data)
        try:
            return pdf[page - 1].render(scale=dpi / 72).to_pil().convert("RGB")
        finally:
            pdf.close()


def trim_margins(image, pad: int = 24):
    """Crop the blank border (many graph pages are half empty), so the figures fill more of the image.
    A short footer below a wide blank gap (copyright line, page number) is cropped too."""
    ink = np.asarray(ImageChops.difference(image, Image.new("RGB", image.size, "white")).convert("L")) > 24
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    if not rows.size:
        return image
    gaps = np.flatnonzero(np.diff(rows) > 0.1 * image.height)  # blank bands taller than 10% of the page
    bottom = rows[-1]
    if gaps.size and rows[-1] - rows[gaps[-1] + 1] < 0.06 * image.height:
        bottom = rows[gaps[-1]]
    box = (cols[0] - pad, rows[0] - pad, cols[-1] + 1 + pad, bottom + 1 + pad)  # crop edges are exclusive
    return image.crop((max(box[0], 0), max(box[1], 0), min(box[2], image.width), min(box[3], image.height)))


def _encode(image, fmt: str, **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    return buffer.getvalue()


def read_graph(client, model: str, data: bytes, doc: str, page: int, question: str) -> dict:
    image = trim_margins(render_page(data, page))
    png = base64.b64encode(_encode(image, "PNG")).decode()
    try:
        # Reasoning stays on here: with it off, the model misread a log frequency axis (10 V at
        # 20 kHz where the LM358 graph shows about 4.5 V); with it on, it read about 5 V.
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}},
                    {"type": "text", "text": f"Page {page} of {doc}. Question: {question}"},
                ]},
            ],
            temperature=0,
            max_tokens=6000,
        )
    except APIError as exc:
        return {"error": f"Graph reading failed: {exc.message}"}
    reading = parse_json_object(clean(response.choices[0].message.content))
    if not reading or not reading.get("value"):
        note = (reading or {}).get("reading")
        return {"error": f"No graph on {doc} p.{page} answers this." + (f" {note}" if note else "")}
    preview = image.resize((PREVIEW_WIDTH, round(image.height * PREVIEW_WIDTH / image.width)))
    return {
        "doc": doc,
        "page": page,
        **{key: reading.get(key) for key in FIELDS},
        "_image": _encode(preview, "JPEG", quality=80),  # for the evidence panel; not sent to the model
    }
