"""Pure document-type classification logic (M-2.5).

Runs as a step inside the per-document OCR job (:mod:`ps06.ocr.job`), right
after :func:`ps06.ocr.extraction.extract`, reusing the *same*
:class:`~ps06.ocr.http_client.ReauthHTTPClient` instance rather than minting a
second one — there is no separate classification job/invocation, matching the
AEAD source prototype's single "OCR Agent" (OCR + classify in one pass) and
this build's per-invocation token isolation, which only makes sense measured
per *job*, not per LLM call within a job.

:func:`classify` is a pure function of ``(extracted_text, config, client)`` —
like :func:`ps06.ocr.extraction.extract`, it knows nothing about case/document
identity, timing, or persistence. Unlike extraction, it never raises for an
ambiguous or unparseable model response: a classification the model can't
resolve confidently degrades to :attr:`DocumentType.OTHER`, not a job failure.
Only genuine infrastructure failures (auth exhaustion, network errors) can
raise out of this module, and those are already covered by ``job.py``'s
existing ``except Exception`` -> ``FAILED`` handling — no new exception
handling is needed there.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class DocumentType(str, Enum):
    """Document-type categories for an immigration case file.

    Matches PS-06 checklist Req #10's category list, except birth-certificate
    generation is a separate :attr:`ClassificationResult.generation` field
    rather than baked into the type (``BIRTH_CERT_G1``/``G2``/``G3``) — a
    single document frequently cannot self-determine which generation it is
    without cross-referencing the rest of the case file (that resolution is
    M-4's job, not this one).
    """

    APPLICATION = "APPLICATION"
    ID_DOCUMENT = "ID_DOCUMENT"
    BIRTH_CERT = "BIRTH_CERT"
    OTHER = "OTHER"
    EMPTY = "EMPTY"


_GENERATION_LABELS = frozenset({"G1", "G2", "G3"})

# Below this many non-boilerplate characters, a document is classified EMPTY
# deterministically, without calling the LLM at all (cheaper, and there is
# nothing meaningful to ask a model to classify).
MIN_CHARS_FOR_CLASSIFICATION = 10

_PAGE_HEADER_RE = re.compile(r"^--- Page \d+.*?---$", re.MULTILINE)


def _strip_boilerplate(extracted_text: str) -> str:
    """Strip the ``--- Page N ... ---`` headers ``extraction.py`` always adds.

    Used only to decide whether there is any *real* content to classify —
    the headers themselves would otherwise make an effectively-blank page
    look non-empty.
    """
    return _PAGE_HEADER_RE.sub("", extracted_text).strip()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CLASSIFICATION_PROMPT = """\
You are a document classifier for a Spanish-language immigration case-file
system. You will be given the OCR-transcribed text of ONE document. Classify
it into exactly one category based on Spanish-language cues that are typical
of that category's schema fields.

Categories and their typical Spanish cues:

  APPLICATION   - An application/petition form. Look for headers like
                  "SOLICITUD", "SOLICITUD DE...", form fields resembling
                  "Nombre del solicitante", "Apellido del padre",
                  "Apellido de la madre", "Fecha de solicitud".

  ID_DOCUMENT   - A government-issued identity document (passport, national
                  ID, cedula). Look for "PASAPORTE", "DNI", "CEDULA",
                  "Cedula de identidad", "Tipo de ID", "Numero de ID",
                  MRZ-like lines (P<<<...), a photo/signature block.

  BIRTH_CERT    - A birth certificate. Look for "CERTIFICADO DE NACIMIENTO",
                  "Acta de Nacimiento", "grado_certificado", fields naming a
                  parent's identity/nationality such as "Nombre progenitor 1",
                  "Nombre progenitor 2", "Nacionalidad progenitor 1".

  EMPTY         - The text is blank, contains only OCR noise/garbage
                  characters, or has no discernible document content at all
                  (e.g. a blank/near-blank scanned page).

  OTHER         - The text clearly represents SOME document, but it does not
                  match any category above (e.g. a cover letter, an
                  unrelated receipt, correspondence).

If the document is a birth certificate AND you can determine which
generation it explicitly declares (look for an explicit label such as
"grado_certificado": "G_1"/"G_2"/"G_3", or unambiguous self-declared
relationship language identifying the certificate-holder's generation
relative to the case), state that generation. Do NOT guess a generation from
context you cannot verify within this single document - leave it unstated
if it is not explicit.

Reply on the FIRST line with EXACTLY ONE of: APPLICATION, ID_DOCUMENT,
BIRTH_CERT, OTHER, EMPTY - nothing else on that line.
If and only if document_type is BIRTH_CERT and a generation is explicitly
stated in the document, reply on a SECOND line with exactly one of:
G1, G2, G3. Otherwise, do not output a second line.

Document text:
---
{extracted_text}
---
"""


class ClassificationConfig(BaseModel):
    """Configuration for one classification call within an OCR job."""

    model_config = ConfigDict(frozen=True)

    max_tokens: int = 32
    temperature: float = 0.0
    prompt_template: str = Field(default=DEFAULT_CLASSIFICATION_PROMPT)
    max_text_chars: int = 4000
    min_chars_for_classification: int = MIN_CHARS_FOR_CLASSIFICATION


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    """Result of classifying one document's extracted text."""

    model_config = ConfigDict(frozen=True)

    document_type: DocumentType
    generation: Literal["G1", "G2", "G3"] | None = None
    # Placeholder for PS-06 checklist Req #11 ("confidence score logging",
    # status "To be developed" / Phase 2) — always 1.0 for now. Kept as a
    # real, always-populated float (not None) so a future real value is a
    # pure value swap, not a schema change.
    confidence: float = 1.0
    # The raw model output (or "" for the EMPTY short-circuit / no LLM call).
    # Kept for audit/debuggability, especially for the OTHER fallback case,
    # where it's the only record of what the model actually said.
    raw_response: str = ""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _call_classifier_llm(
    client: Any, config: ClassificationConfig, extracted_text: str
) -> str:
    """Call the LLM through ``client.chat_completion`` with a text-only prompt.

    Unlike ``ps06.ocr.extraction._call_vlm``, there is no image payload —
    classification reasons over already-extracted text, not pixels.
    """
    prompt = config.prompt_template.format(
        extracted_text=extracted_text[: config.max_text_chars]
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return client.chat_completion(
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _resolve_label(candidate: str, valid_labels: frozenset[str]) -> str | None:
    """Exact match first, then a fuzzy substring match. Returns None if
    nothing in ``valid_labels`` can be resolved from ``candidate``."""
    if candidate in valid_labels:
        return candidate
    for label in valid_labels:
        if label in candidate:
            return label
    return None


def _parse_classification_response(raw: str) -> tuple[DocumentType, str | None]:
    """Parse the model's raw reply into ``(document_type, generation)``.

    Falls back to ``(DocumentType.OTHER, None)`` on anything unparseable —
    this function never raises. A second line is only consulted when the
    first line resolves to ``BIRTH_CERT``.
    """
    lines = [line.strip() for line in raw.upper().splitlines() if line.strip()]
    if not lines:
        logger.debug("classification response empty raw=%r -> OTHER", raw)
        return DocumentType.OTHER, None

    first = re.sub(r"[^A-Z0-9_]", "", lines[0])
    valid_types = frozenset(t.value for t in DocumentType)
    resolved = _resolve_label(first, valid_types)
    if resolved is None:
        logger.debug("classification response unparseable raw=%r -> OTHER", raw)
        return DocumentType.OTHER, None

    document_type = DocumentType(resolved)
    if document_type is not DocumentType.BIRTH_CERT or len(lines) < 2:
        return document_type, None

    second = re.sub(r"[^A-Z0-9_]", "", lines[1])
    generation = _resolve_label(second, _GENERATION_LABELS)
    return document_type, generation


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def classify(
    extracted_text: str, config: ClassificationConfig, client: Any
) -> ClassificationResult:
    """Classify one document's extracted text. Never raises.

    A near-blank document (after stripping the ``--- Page N ---`` boilerplate
    ``extraction.py`` always adds) is classified ``EMPTY`` deterministically,
    without calling the LLM. Otherwise, the LLM is called once and its
    response is parsed with a fuzzy fallback to ``OTHER`` — an ambiguous or
    unparseable classification is not treated as a processing failure.
    """
    stripped = _strip_boilerplate(extracted_text)
    if len(stripped) < config.min_chars_for_classification:
        return ClassificationResult(document_type=DocumentType.EMPTY, raw_response="")

    raw = _call_classifier_llm(client, config, extracted_text)
    document_type, generation = _parse_classification_response(raw)
    return ClassificationResult(
        document_type=document_type, generation=generation, raw_response=raw
    )
