"""Structured field extraction for ID documents and birth certificates (Fix Plan item 4).

Runs as a step inside the per-document OCR job (:mod:`ps06.ocr.job`), right
after :func:`ps06.classification.classifier.classify`, reusing the *same*
:class:`~ps06.ocr.http_client.ReauthHTTPClient` instance rather than minting a
second token — same per-job (not per-LLM-call) token isolation rationale as
``classifier.py``.

:func:`extract_fields` is a pure function of
``(extracted_text, document_type, config, client)``. Like :func:`classify`,
it never raises for an ambiguous or unparseable model response: an ID/cert
whose fields the model can't produce as valid JSON degrades to an empty
result (every field stays missing downstream, in :mod:`ps06.rules.adapter`),
never a job failure and never a fabricated value. Only genuine infrastructure
failures (auth exhaustion, network errors) can raise out of this module, and
those are already covered by ``job.py``'s existing ``except Exception`` ->
``FAILED`` handling.

Unlike classification, this step only runs for ``ID_DOCUMENT``/``BIRTH_CERT``
documents — ``APPLICATION`` fields come from AcroForm data (``adapter.py``),
and ``OTHER``/``EMPTY`` have no structured schema to extract.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ps06.classification.classifier import DocumentType
from ps06.rules.normalize import strip_vlm_artifacts
from ps06.rules.schemas import CERTIFICADO_SOURCE_LABELS, ID_SOURCE_LABELS

logger = logging.getLogger(__name__)

_GENERATION_LABELS = frozenset({"G1", "G2", "G3"})


def _normalize_label(label: str) -> str:
    """Case/whitespace-insensitive key, matching ``adapter.py``'s convention."""
    return " ".join(label.strip().lower().split())


#: Normalized Spanish source label -> canonical attribute, one index per schema.
_ID_LABEL_INDEX: dict[str, str] = {
    _normalize_label(label): attr for attr, label in ID_SOURCE_LABELS.items()
}
_CERTIFICADO_LABEL_INDEX: dict[str, str] = {
    _normalize_label(label): attr for attr, label in CERTIFICADO_SOURCE_LABELS.items()
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_ID_EXTRACTION_PROMPT = """\
You are a structured-data extractor for a Spanish-language immigration
case-file system. You will be given the OCR-transcribed text of ONE identity
document (passport, national ID, cedula). Extract the following fields if
and only if their value is explicitly present in the text. Do not guess,
infer, or fabricate a value for a field that is not clearly present.

Reply with EXACTLY ONE JSON object (no markdown fence, no extra text) whose
keys are EXACTLY these Spanish labels and whose values are the field's value
as a plain string, trimmed of surrounding whitespace:

  "Tipo de ID", "Número de ID", "Apellidos", "Nombres", "Nacionalidad",
  "Sexo", "Fecha de nacimiento", "Fecha de emisión", "Fecha de vencimiento"

Omit any key whose value is not present in the text. Dates should be copied
verbatim as written in the document (do not reformat them).

Document text:
---
{extracted_text}
---
"""

DEFAULT_CERTIFICADO_EXTRACTION_PROMPT = """\
You are a structured-data extractor for a Spanish-language immigration
case-file system. You will be given the OCR-transcribed text of ONE birth
certificate. Extract the following fields if and only if their value is
explicitly present in the text. Do not guess, infer, or fabricate a value for
a field that is not clearly present.

Reply with EXACTLY ONE JSON object (no markdown fence, no extra text) whose
keys are EXACTLY these labels and whose values are the field's value as a
plain string, trimmed of surrounding whitespace:

  "grado_certificado", "Nombre G_X", "Fecha de nacimiento G_X", "sexo G_X",
  "Lugar de inscripción G_X", "Fecha de Inscripción G_X",
  "Nombre progenitor 1 G_X", "Nº Documento progenitor 1 G_X",
  "Nacionalidad progenitor 1 G_X", "Nombre progenitor 2 G_X",
  "Nº Documento progenitor 2 G_X", "Nacionalidad progenitor 2 G_X",
  "Apostillado G_X"

"grado_certificado" is this document's own self-declared generation
(G1/G2/G3) if and only if it is explicitly stated in the text (e.g. a label
reading "grado_certificado: G_1") — leave it out if not explicit, never
infer it from context. "Apostillado G_X" should be the apostille stamp/seal
text if present, or omitted if there is no apostille.

Omit any key whose value is not present in the text. Dates should be copied
verbatim as written in the document (do not reformat them).

Document text:
---
{extracted_text}
---
"""


class FieldExtractionConfig(BaseModel):
    """Configuration for one field-extraction call within an OCR job."""

    model_config = ConfigDict(frozen=True)

    max_tokens: int = 500
    temperature: float = 0.0
    id_prompt_template: str = Field(default=DEFAULT_ID_EXTRACTION_PROMPT)
    certificado_prompt_template: str = Field(default=DEFAULT_CERTIFICADO_EXTRACTION_PROMPT)
    max_text_chars: int = 4000


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


class FieldExtractionResult(BaseModel):
    """Result of extracting one document's structured fields.

    ``fields`` is keyed by canonical attribute name (matching
    :class:`ps06.rules.schemas.IdData`/:class:`ps06.rules.schemas.CertificadoData`
    field names), not the Spanish source label — the label -> attribute
    resolution happens once, here, so :mod:`ps06.rules.adapter` can consume
    this directly without re-deriving it. Only non-empty values are present;
    absence of a key means the field is missing, never a fabricated empty
    string.
    """

    model_config = ConfigDict(frozen=True)

    fields: dict[str, str] = Field(default_factory=dict)
    # The raw model output (or "" if no LLM call was made). Kept for
    # audit/debuggability, especially when parsing fails.
    raw_response: str = ""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _call_extraction_llm(
    client: Any, config: FieldExtractionConfig, prompt_template: str, extracted_text: str
) -> str:
    """Call the LLM through ``client.chat_completion`` with a text-only prompt."""
    prompt = prompt_template.format(extracted_text=extracted_text[: config.max_text_chars])
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return client.chat_completion(
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_extraction_response(raw: str, label_index: dict[str, str]) -> dict[str, str]:
    """Parse the model's raw JSON reply into canonical-attr -> value.

    Never raises: any non-JSON, non-dict, or otherwise malformed response
    yields an empty dict (every field stays missing downstream), the same
    degrade-never-fabricate posture as ``classifier._parse_classification_response``.
    """
    try:
        parsed = json.loads(strip_vlm_artifacts(raw))
    except (json.JSONDecodeError, TypeError):
        logger.debug("field extraction response not valid JSON raw=%r", raw)
        return {}

    if not isinstance(parsed, dict):
        logger.debug("field extraction response not a JSON object raw=%r", raw)
        return {}

    resolved: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        attr = label_index.get(_normalize_label(key))
        if attr is None:
            continue
        stripped = value.strip()
        if not stripped:
            continue
        if attr == "grado_certificado":
            upper = stripped.replace("_", "").replace(" ", "").upper()
            if upper not in _GENERATION_LABELS:
                continue
            resolved[attr] = upper
            continue
        resolved[attr] = stripped

    return resolved


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def extract_fields(
    extracted_text: str,
    document_type: DocumentType,
    config: FieldExtractionConfig,
    client: Any,
) -> FieldExtractionResult:
    """Extract one document's structured fields. Never raises.

    Only ``ID_DOCUMENT``/``BIRTH_CERT`` documents are extracted — any other
    ``document_type`` returns an empty result without calling the LLM, since
    neither schema applies.
    """
    if document_type is DocumentType.ID_DOCUMENT:
        prompt_template = config.id_prompt_template
        label_index = _ID_LABEL_INDEX
    elif document_type is DocumentType.BIRTH_CERT:
        prompt_template = config.certificado_prompt_template
        label_index = _CERTIFICADO_LABEL_INDEX
    else:
        return FieldExtractionResult()

    raw = _call_extraction_llm(client, config, prompt_template, extracted_text)
    fields = _parse_extraction_response(raw, label_index)
    return FieldExtractionResult(fields=fields, raw_response=raw)
