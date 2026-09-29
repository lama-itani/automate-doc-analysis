"""Full per-document OCR job result (M-2, step 6; classification added M-2.5).

Composes :class:`ps06.ocr.extraction.ExtractionResult` (pure extraction facts)
with the case/document identity, timing, and model fields that the job
(:mod:`ps06.ocr.job`, step 7) knows and extraction does not. This is the model
serialized into the ``document_extraction.payload`` JSON column (migration
0002 in :mod:`ps06.status.db`), so it must be a complete, self-contained
record — readable without cross-referencing the DB row it lives in. That is
why ``processing_seconds`` appears both here and as its own DB column: the
column exists for queryability (e.g. sorting without JSON parsing), not to
avoid storing the value twice.

``classification`` (:class:`ps06.classification.classifier.ClassificationResult`)
is optional only for backward compatibility with ``document_extraction`` rows
persisted before M-2.5 existed — every job invocation from M-2.5 onward always
populates it. No verdict/rules fields beyond classification — M-4's rules
engine is still out of scope here (see the "M-2 session 2" Progress Log entry
for the original resolved scope decision).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ps06.classification.classifier import ClassificationResult
from ps06.ocr.extraction import ExtractionResult


def _utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string (second precision, ``Z`` suffix).

    Mirrors ``ps06.status.store._utcnow_iso`` rather than importing it —
    ``ps06.ocr`` and ``ps06.status`` stay decoupled, and this is a 3-line
    pure function, not worth a cross-package dependency for.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


class OcrResult(BaseModel):
    """Full result of one per-document OCR job invocation."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    document_id: str
    source_file: str  # basename only, not the full local path
    extraction: ExtractionResult
    classification: ClassificationResult | None = None
    processing_seconds: float
    model_name: str
    extracted_at: str  # ISO-8601 UTC

    @classmethod
    def build(
        cls,
        *,
        case_id: str,
        document_id: str,
        file_path: str,
        extraction: ExtractionResult,
        processing_seconds: float,
        model_name: str,
        classification: ClassificationResult | None = None,
    ) -> OcrResult:
        """Construct an ``OcrResult``, deriving ``source_file`` and ``extracted_at``."""
        return cls(
            case_id=case_id,
            document_id=document_id,
            source_file=Path(file_path).name,
            extraction=extraction,
            classification=classification,
            processing_seconds=processing_seconds,
            model_name=model_name,
            extracted_at=_utcnow_iso(),
        )
