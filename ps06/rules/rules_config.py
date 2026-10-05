"""Deterministic configuration for the Lineage + Semaphore engines (M-4, step 3).

The AEAD source's Data Aggregation agent baked every threshold, checklist
entry, and flag-severity rule into prompt text. This module ports those as
real, checked-in-under-Git-and-legal-sign-off data (:mod:`ps06.rules.rules_config`
+ ``rules.yaml``) rather than prose an LLM has to re-derive every run.

``RulesConfig()`` with no arguments constructs a fully valid config purely
from Python defaults; the checked-in ``ps06/rules/rules.yaml`` mirrors those
defaults exactly, so ``RulesConfig.from_yaml("ps06/rules/rules.yaml")`` and
``RulesConfig()`` are equivalent until someone edits the YAML.

A missing-expected-document check is deliberately not a flag *name* in
:class:`SeverityMapping` — the AEAD source describes it procedurally (the
"MISSING DOCUMENT RULE") rather than giving it a discrete flag constant like
``DOCUMENTO_VENCIDO``. It is checked structurally against
:attr:`RulesConfig.expected_documents` by a later step (Semaphore engine),
not looked up here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ps06.classification.classifier import DocumentType
from ps06.rules.schemas import Generation


class GapBand(BaseModel):
    """A generational-gap-in-years acceptance band (inclusive bounds)."""

    model_config = ConfigDict(frozen=True)

    min_years: int
    max_years: int


class GapBandsConfig(BaseModel):
    """Generational-gap-in-years bands (Lineage Engine).

    ``typical`` is the normal range for a parent/child generational gap;
    outside ``hard``, the assignment is rejected and ``BRECHA_GENERACIONAL``
    is raised.
    """

    model_config = ConfigDict(frozen=True)

    typical: GapBand = Field(default_factory=lambda: GapBand(min_years=20, max_years=35))
    hard: GapBand = Field(default_factory=lambda: GapBand(min_years=14, max_years=60))


class EntityResolutionConfig(BaseModel):
    """Lineage Engine entity-resolution parameters.

    Two records merge into one resolved :class:`~ps06.rules.schemas.Person`
    if they match on at least ``min_criteria`` of ``total_criteria`` signals
    (full name fuzzy match, birth date, document number).
    """

    model_config = ConfigDict(frozen=True)

    min_criteria: int = 2
    total_criteria: int = 3
    #: Fed to ``normalize.dates_within_years(..., tolerance_years=...)`` for
    #: the "Birth date (±1 year)" entity-resolution criterion.
    dob_tolerance_years: int = 1


class ExpectedDocument(BaseModel):
    """One of the Semaphore Engine's 5 expected-document checklist entries.

    ``generation`` disambiguates the 3 entries that all map to
    ``DocumentType.BIRTH_CERT`` (Solicitante/Progenitor/Abuelo's certs) —
    ``document_type`` alone cannot, since one enum member covers all three.
    ``None`` for the two non-birth-certificate entries.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    document_type: DocumentType
    generation: Optional[Generation] = None


class SeverityMapping(BaseModel):
    """Flag-name -> ROJO/AMARILLO severity classification.

    Names are verbatim from the AEAD source's "Conflict Flags" list (Spanish
    where the source uses Spanish; ``DOCUMENT_INCOMPLETE`` is kept in
    English exactly as the source has it — that is the source's own choice,
    not a translation decision made here).
    """

    model_config = ConfigDict(frozen=True)

    rojo: tuple[str, ...] = (
        "DOCUMENTO_VENCIDO",
        "NOMBRE_AMBIGUO",
        "NOMBRE_INCONSISTENTE",
        "ID_INCONSISTENTE",
        "DOCUMENTO_CRUZADO",
        "FECHA_CONFLICTO",
        "BRECHA_GENERACIONAL",
    )
    amarillo: tuple[str, ...] = (
        "DOCUMENT_INCOMPLETE",
        "NACIONALIDAD_INCONSISTENTE",
        "FECHA_PRESENTACION_FALTANTE",
        "GENERACION_INFERIDA_BAJA_CONFIANZA",
    )


class RulesConfig(BaseModel):
    """All deterministic parameters for the Lineage + Semaphore engines."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gap_bands: GapBandsConfig = Field(default_factory=GapBandsConfig)
    entity_resolution: EntityResolutionConfig = Field(default_factory=EntityResolutionConfig)
    #: ``DOCUMENTO_VENCIDO`` is checked against the application's own
    #: ``fecha_presentacion`` plus this many months, not the run's
    #: ``evaluation_date`` — a passport must still be valid this far past
    #: *when the applicant actually applied*, not past whatever day the
    #: pipeline happens to run. When ``fecha_presentacion`` is
    #: absent/unparseable, the check is skipped entirely (not evaluated
    #: against ``evaluation_date``) and ``FECHA_PRESENTACION_FALTANTE`` is
    #: raised instead (semaphore.py).
    document_validity_months_past_application: int = 6
    expected_documents: tuple[ExpectedDocument, ...] = Field(
        default_factory=lambda: (
            ExpectedDocument(
                label="Solicitud Principal",
                document_type=DocumentType.APPLICATION,
            ),
            ExpectedDocument(
                label="Identificación del Solicitante",
                document_type=DocumentType.ID_DOCUMENT,
            ),
            ExpectedDocument(
                label="Certificado de Nacimiento del Solicitante",
                document_type=DocumentType.BIRTH_CERT,
                generation="G1",
            ),
            ExpectedDocument(
                label="Certificado de Nacimiento del Progenitor",
                document_type=DocumentType.BIRTH_CERT,
                generation="G2",
            ),
            ExpectedDocument(
                label="Certificado de Nacimiento Español de origen",
                document_type=DocumentType.BIRTH_CERT,
                generation="G3",
            ),
        )
    )
    severity: SeverityMapping = Field(default_factory=SeverityMapping)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RulesConfig":
        """Load and validate a :class:`RulesConfig` from a YAML file.

        Fails fast: a missing file raises ``FileNotFoundError``, malformed
        YAML raises ``yaml.YAMLError``, and an unknown or mistyped key raises
        ``pydantic.ValidationError`` (this model's ``extra="forbid"``). A
        YAML that legitimately omits a nested section/field is a supported
        partial override — the omitted field falls back to its Python
        default, the same as any other pydantic model with defaults (e.g.
        ``OcrJobConfig``); that is not treated as a silent failure.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if raw is None:
            raise ValueError(f"{path}: empty or invalid YAML (no top-level mapping)")
        return cls.model_validate(raw)
