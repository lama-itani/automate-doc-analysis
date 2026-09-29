"""Pure OCR/VLM extraction logic for the per-document job (M-2).

Ported near-verbatim from the Agent Studio tool
``AEAD-v5.1-import/studio-data/tool_templates/ocr_tool_ac9opUz/tool.py`` — the
VLM call, orientation correction, PDF text-layer fast path, and AcroForm/XFA
extraction. The tool-sandbox framing (argparse entry point, ``UserParameters``/
``ToolParameters``, session-relative file resolution, flat-file debug writers,
page-level thread parallelism) is stripped; the Cloudera AI Inference call is
rewired to go through :class:`ps06.ocr.http_client.ReauthHTTPClient` instead of
a raw ``OpenAI`` client, since fresh-JWT + 401-retry is this build's whole reason
for existing.

:func:`extract` is a pure function of ``(file_path, config, client)`` — it knows
nothing about case/document identity, timing, or persistence. Those are the
job's concern (:mod:`ps06.ocr.job`, step 7), which composes this module's
:class:`ExtractionResult` into the full :class:`ps06.ocr.envelope.OcrResult`.
Every failure mode raises a typed :class:`ExtractionError` subclass rather than
returning an error dict, so the job can catch one exception type and always
reach a ``FAILED`` transition with a populated ``error_detail`` — no silent
failures.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}
SUPPORTED_EXTENSIONS = {".pdf", *IMAGE_EXTENSIONS}
DEFAULT_PDF_DPI = 300
MAX_FILE_SIZE_MB = 50

ORIENTATION_LABELS = frozenset({"NONE", "ROTATE_90", "ROTATE_180", "ROTATE_270"})

ORIENTATION_PROMPT = (
    "You are an image orientation classifier. "
    "Examine the text in this image and decide how far it is rotated away from its "
    "correct reading orientation.\n\n"
    "Reply with EXACTLY ONE of the following labels and nothing else — "
    "no punctuation, no explanation, no whitespace before or after:\n\n"
    "  NONE          — text is already correctly oriented, upright and readable\n"
    "  ROTATE_90     — text is rotated 90 degrees clockwise from upright\n"
    "  ROTATE_180    — text is rotated 180 degrees (upside-down) from upright\n"
    "  ROTATE_270    — text is rotated 270 degrees clockwise (i.e. 90 degrees "
    "counter-clockwise) from upright\n\n"
    "Output only the label."
)

DEFAULT_OCR_PROMPT = (
    "You are an advanced OCR and document transcription engine. "
    "Transcribe all text content from this image exactly as it appears, "
    "preserving reading order, headings, lists, and tables using precise markdown formatting. "
    "CRITICAL INSTRUCTION FOR FORMS: You must explicitly detect and represent the state of all visual form elements. "
    "Use '[X]' for marked, checked, or filled checkboxes and radio buttons. "
    "Use '[ ]' for empty or unchecked checkboxes and radio buttons. "
    "If present, indicate non-textual elements using structural tags like '[Signature]' or '[Stamp]'. "
    "Do not summarize, translate, or add any conversational commentary — "
    "output strictly the transcribed markdown content. Try not hallucinate outputting garbage sequence of characters. "
    "Copy identification and registry numbers exactly as printed, including every digit and punctuation mark."
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class OcrJobConfig(BaseModel):
    """Configuration for one OCR job invocation.

    ``endpoint_url`` is consumed by :mod:`ps06.ocr.job` (step 7) to construct the
    :class:`~ps06.ocr.http_client.ReauthHTTPClient` passed into :func:`extract` —
    it is not read anywhere in this module, since extraction always receives an
    already-built client. It lives on this model anyway because it is naturally
    part of "the OCR job's configuration" (mirrors the port source's
    ``UserParameters``, which also mixes client-construction fields with
    per-call fields).
    """

    model_config = ConfigDict(frozen=True)

    endpoint_url: str
    model_name: str
    max_tokens: int = 2500
    pdf_dpi: int = DEFAULT_PDF_DPI
    prompt: str = Field(default=DEFAULT_OCR_PROMPT)
    temperature: float = 0.0
    use_text_layer_fast_path: bool = True
    min_text_layer_chars: int = 60


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """Base class for extraction errors. job.py catches this to reach FAILED."""


class UnsupportedFileTypeError(ExtractionError):
    """Raised when the file extension is not in ``SUPPORTED_EXTENSIONS``."""


class FileTooLargeError(ExtractionError):
    """Raised when the file exceeds ``MAX_FILE_SIZE_MB``."""


class EmptyDocumentError(ExtractionError):
    """Raised when a PDF has zero pages."""


class EmptyExtractionError(ExtractionError):
    """Raised when the model returned no usable content for the whole file."""


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


class ExtractionResult(BaseModel):
    """Pure extraction facts for one document. No identity, timing, or model name.

    :mod:`ps06.ocr.envelope` (step 6) composes this into the full ``OcrResult``
    alongside case/document identity, ``processing_seconds``, and ``extracted_at``.
    """

    model_config = ConfigDict(frozen=True)

    file_type: str  # "pdf" | "image"
    extracted_text: str
    page_count: int
    pages_via_text_layer: int
    pages_via_vlm: int
    orientation_corrections: dict[int, str]  # page_num -> label, non-NONE only
    acroform_fields: dict[str, str]
    xfa_fields: dict[str, str]


# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------


def _encode_pil_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pil_from_b64(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ---------------------------------------------------------------------------
# Orientation correction
# ---------------------------------------------------------------------------


def _correct_orientation(img: Image.Image, label: str) -> Image.Image:
    # PIL's rotate(angle) turns the image counter-clockwise by ``angle`` degrees.
    # A label of ROTATE_90 means the content is 90 degrees clockwise from upright,
    # so undoing it requires the same counter-clockwise rotate(90).
    if label == "ROTATE_90":
        return img.rotate(90, expand=True)
    if label == "ROTATE_180":
        return img.rotate(180)
    if label == "ROTATE_270":
        return img.rotate(270, expand=True)
    return img


# ---------------------------------------------------------------------------
# VLM call — the auth-wrapped seam
# ---------------------------------------------------------------------------


def _call_vlm(
    client: Any,
    config: OcrJobConfig,
    encoded_image: str,
    mime_type: str,
    prompt: str,
    *,
    max_tokens: int | None = None,
    disable_thinking: bool = False,
) -> str:
    """Call the VLM through ``client.chat_completion`` (a ReauthHTTPClient).

    ``client`` is duck-typed to :class:`ps06.ocr.http_client.ReauthHTTPClient` —
    only ``.chat_completion(...)`` is used, which already strips the returned
    content, so no additional ``.strip()`` is needed here.
    """
    extra_body: dict[str, Any] = {}
    if disable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"},
                },
            ],
        }
    ]
    return client.chat_completion(
        messages=messages,
        max_tokens=max_tokens or config.max_tokens,
        temperature=config.temperature,
        extra_body=extra_body or None,
    )


# ---------------------------------------------------------------------------
# Orientation detection
# ---------------------------------------------------------------------------


def _detect_orientation(client: Any, config: OcrJobConfig, encoded_image: str) -> str:
    raw = _call_vlm(
        client,
        config,
        encoded_image,
        mime_type="image/png",
        prompt=ORIENTATION_PROMPT,
        max_tokens=16,
        disable_thinking=True,
    )

    candidate = re.sub(r"[^A-Z0-9_]", "", raw.upper().strip())

    if candidate in ORIENTATION_LABELS:
        logger.debug("orientation detected raw=%r resolved=%s", raw, candidate)
        return candidate

    for label in ORIENTATION_LABELS:
        if label in candidate:
            logger.debug("orientation detected raw=%r resolved=%s (fuzzy match)", raw, label)
            return label

    logger.debug("orientation detected raw=%r resolved=NONE (unrecognised)", raw)
    return "NONE"


# ---------------------------------------------------------------------------
# Programmatic form field extraction (AcroForm + XFA)
# ---------------------------------------------------------------------------


def _extract_acroform_fields(file_path: str) -> dict[str, str]:
    # Soft-fails into a sentinel key rather than raising: a form-metadata
    # extraction problem must not fail an otherwise-successful OCR pass.
    fields: dict[str, str] = {}
    try:
        doc = pymupdf.open(file_path)
        for page in doc:
            for widget in page.widgets():
                name = widget.field_name or f"unnamed_field_page{page.number}_{widget.rect}"
                value = widget.field_value
                if value is not None and str(value).strip():
                    key = name if name not in fields else f"{name}_p{page.number + 1}"
                    fields[key] = str(value).strip()
        doc.close()
    except Exception as exc:
        fields["__acroform_extraction_error__"] = str(exc)
    return fields


def _extract_xfa_fields(file_path: str) -> dict[str, str]:
    # Soft-fails into a sentinel key, same rationale as _extract_acroform_fields.
    fields: dict[str, str] = {}

    try:
        doc = pymupdf.open(file_path)

        if hasattr(doc, "xfa_fields"):
            xfa = doc.xfa_fields()
            if xfa:
                doc.close()
                return {k: str(v) for k, v in xfa.items() if v is not None}

        catalog_xref = doc.pdf_catalog()
        catalog_obj = doc.xref_object(catalog_xref, compressed=False)

        acroform_match = re.search(r"/AcroForm\s+(\d+)\s+\d+\s+R", catalog_obj)
        if acroform_match:
            acroform_xref = int(acroform_match.group(1))
            acroform_obj = doc.xref_object(acroform_xref, compressed=False)

            xfa_match = re.search(r"/XFA\s+(\d+)\s+\d+\s+R", acroform_obj)
            if xfa_match:
                xfa_xref = int(xfa_match.group(1))
                xfa_stream = doc.xref_stream(xfa_xref)
                if xfa_stream:
                    fields = _parse_xfa_xml(xfa_stream)

        doc.close()

    except Exception as exc:
        fields["__xfa_extraction_error__"] = str(exc)

    return fields


def _parse_xfa_xml(raw_xml: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}

    chunk_starts = [m.start() for m in re.finditer(rb"<(?![\?/!])\w", raw_xml)]
    if len(chunk_starts) <= 1:
        xml_fragments = [raw_xml]
    else:
        xml_fragments = (
            [raw_xml]
            + [raw_xml[chunk_starts[i] : chunk_starts[i + 1]] for i in range(len(chunk_starts) - 1)]
            + [raw_xml[chunk_starts[-1] :]]
        )

    for fragment in xml_fragments:
        try:
            root = ET.fromstring(fragment)
        except ET.ParseError:
            continue

        tag = root.tag.lower()
        if "datasets" not in tag and "data" not in tag:
            continue

        for elem in root.iter():
            text = (elem.text or "").strip()
            if not text:
                continue
            local_tag = re.sub(r"\{[^}]*\}", "", elem.tag)  # strip namespace
            fields[local_tag] = text

        if fields:
            break

    return fields


def _format_form_fields_block(acroform: dict[str, str], xfa: dict[str, str]) -> str:
    lines: list[str] = []

    if acroform:
        lines.append("## Programmatically Extracted Form Fields (AcroForm)\n")
        for name, value in acroform.items():
            if name.startswith("__") and name.endswith("__"):
                lines.append(f"> ⚠️ **Extraction warning:** {value}\n")
            else:
                lines.append(f"- **{name}**: {value}")
        lines.append("")

    if xfa:
        lines.append("## Programmatically Extracted Form Fields (XFA)\n")
        for name, value in xfa.items():
            if name.startswith("__") and name.endswith("__"):
                lines.append(f"> ⚠️ **Extraction warning:** {value}\n")
            else:
                lines.append(f"- **{name}**: {value}")
        lines.append("")

    if not lines:
        return ""

    header = (
        "# Fillable Form Fields (Programmatic Extraction)\n\n"
        "> The following values were extracted directly from the PDF's form field "
        "data structures (AcroForm /V entries and/or XFA datasets). "
        "They are guaranteed accurate regardless of whether the appearance stream "
        "was rendered correctly in the visual OCR pass below.\n\n"
    )
    return header + "\n".join(lines) + "\n\n---\n\n"


# ---------------------------------------------------------------------------
# Main pipeline per image
# ---------------------------------------------------------------------------


def _process_image_bytes(
    client: Any,
    config: OcrJobConfig,
    b64_input: str,
    mime_type: str = "image/png",
) -> tuple[str, str]:
    if mime_type != "image/png":
        b64_target = _encode_pil_image(_pil_from_b64(b64_input))
    else:
        b64_target = b64_input

    orientation = _detect_orientation(client, config, b64_target)

    if orientation != "NONE":
        corrected = _correct_orientation(_pil_from_b64(b64_target), orientation)
        b64_target = _encode_pil_image(corrected)
        logger.debug("orientation corrected label=%s", orientation)

    ocr_text = _call_vlm(
        client,
        config,
        b64_target,
        mime_type="image/png",
        prompt=config.prompt,
        disable_thinking=True,
    )

    return ocr_text, orientation


# ---------------------------------------------------------------------------
# PDF and image extraction entry points
# ---------------------------------------------------------------------------


def _pdf_page_text_layer(page: pymupdf.Page) -> str:
    try:
        return (page.get_text() or "").strip()
    except Exception:
        return ""


def _render_page_png_b64(page: pymupdf.Page, dpi: int) -> str:
    mat = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


def _process_pdf_page_vlm(
    client: Any,
    config: OcrJobConfig,
    png_b64: str,
) -> tuple[str, str]:
    return _process_image_bytes(client, config, png_b64, mime_type="image/png")


def _extract_pdf(
    file_path: str, config: OcrJobConfig, client: Any
) -> tuple[str, int, int, int, dict[int, str], dict[str, str], dict[str, str]]:
    """Returns (extracted_text, page_count, pages_via_text_layer, pages_via_vlm,
    orientation_corrections, acroform_fields, xfa_fields)."""
    acroform_fields = _extract_acroform_fields(file_path)
    xfa_fields = _extract_xfa_fields(file_path)
    form_block = _format_form_fields_block(acroform_fields, xfa_fields)

    doc = pymupdf.open(file_path)
    if doc.page_count == 0:
        doc.close()
        raise EmptyDocumentError("PDF appears to have no pages.")

    page_count = doc.page_count
    vlm_jobs: list[tuple[int, str]] = []
    page_texts: dict[int, str] = {}
    pages_via_text_layer = 0

    for page_num in range(page_count):
        page = doc[page_num]
        display_num = page_num + 1
        text_layer = _pdf_page_text_layer(page)
        if config.use_text_layer_fast_path and len(text_layer) >= config.min_text_layer_chars:
            page_texts[display_num] = f"--- Page {display_num} [text layer] ---\n{text_layer}"
            pages_via_text_layer += 1
            continue
        vlm_jobs.append((display_num, _render_page_png_b64(page, config.pdf_dpi)))

    doc.close()

    orientation_corrections: dict[int, str] = {}
    pages_via_vlm = 0
    for display_num, png_b64 in vlm_jobs:
        ocr_text, orientation = _process_pdf_page_vlm(client, config, png_b64)
        pages_via_vlm += 1
        header = f"--- Page {display_num}"
        if orientation != "NONE":
            orientation_corrections[display_num] = orientation
            header += f" [orientation corrected: {orientation}]"
        header += " ---"
        page_texts[display_num] = f"{header}\n{ocr_text}"

    ordered = [page_texts[n] for n in sorted(page_texts)]
    ocr_output = "\n\n".join(ordered)
    extracted_text = form_block + ocr_output

    return (
        extracted_text,
        page_count,
        pages_via_text_layer,
        pages_via_vlm,
        orientation_corrections,
        acroform_fields,
        xfa_fields,
    )


def _extract_image(file_path: str, config: OcrJobConfig, client: Any) -> tuple[str, dict[int, str]]:
    """Returns (extracted_text, orientation_corrections)."""
    ext = Path(file_path).suffix.lower()
    mime_type = IMAGE_EXTENSIONS[ext]
    with open(file_path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("utf-8")
    ocr_text, orientation = _process_image_bytes(client, config, encoded, mime_type)
    orientation_corrections = {1: orientation} if orientation != "NONE" else {}
    return ocr_text, orientation_corrections


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def extract(file_path: str, config: OcrJobConfig, client: Any) -> ExtractionResult:
    """Extract text/fields from one document. Raises ExtractionError subclasses
    on any failure — never returns a partial or error-shaped result."""
    ext = Path(file_path).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(f"Unsupported file type '{ext}'.")

    if os.path.getsize(file_path) / (1024 * 1024) > MAX_FILE_SIZE_MB:
        raise FileTooLargeError(f"File too large. Maximum supported size is {MAX_FILE_SIZE_MB} MB.")

    if ext == ".pdf":
        (
            extracted_text,
            page_count,
            pages_via_text_layer,
            pages_via_vlm,
            orientation_corrections,
            acroform_fields,
            xfa_fields,
        ) = _extract_pdf(file_path, config, client)
        result = ExtractionResult(
            file_type="pdf",
            extracted_text=extracted_text,
            page_count=page_count,
            pages_via_text_layer=pages_via_text_layer,
            pages_via_vlm=pages_via_vlm,
            orientation_corrections=orientation_corrections,
            acroform_fields=acroform_fields,
            xfa_fields=xfa_fields,
        )
    else:
        extracted_text, orientation_corrections = _extract_image(file_path, config, client)
        result = ExtractionResult(
            file_type="image",
            extracted_text=extracted_text,
            page_count=1,
            pages_via_text_layer=0,
            pages_via_vlm=1,
            orientation_corrections=orientation_corrections,
            acroform_fields={},
            xfa_fields={},
        )

    # Whole-assembled-text check (form block + OCR body), matching the port
    # source's granularity: an AcroForm-only PDF the VLM can't read at all will
    # still "succeed" on form data alone, since the form block is prepended
    # before this check runs. That's intentional parity, not a bug.
    if not result.extracted_text.strip():
        raise EmptyExtractionError("Model returned an empty response.")

    return result
