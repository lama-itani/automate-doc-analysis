"""Canonical structured document models for the rules engine (M-4, step 1).

The AEAD prototype's OCR Agent mapped every document into one of three Spanish
JSON schemas (``SOLICITUD`` / ``ID`` / ``CERTIFICADO``) embedded in its prompt
``backstory``. The files-to-reuse table says to extract those "as code, not as
a prompt" — this module is that extraction: the three schemas as real Pydantic
models, plus the resolved :class:`Person`/:class:`CaseBundle` structures the
Lineage and Semaphore engines actually reason over.

**Spanish coherency:** the exact source field labels are preserved verbatim in
the ``*_SOURCE_LABELS`` maps below (canonical snake_case attribute -> the label
as it appears in the AEAD schema), so the adapter can map real extracted data in
and downstream output can render Spanish column values. The Python attribute
names are ASCII snake_case only because Spanish labels contain spaces, accents,
and the templated ``G_X`` suffix and cannot be attribute names.

This module is deliberately free of parsing/normalization logic (that is
:mod:`ps06.rules.normalize`, step 2) and of any I/O — it only defines shapes.
Every value that originates from OCR is optional (``None`` when the field was
absent or not yet extracted); nothing here fabricates data.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

# ``DocumentType`` is the M-2.5 taxonomy; the rules engine keys off the same
# enum rather than defining a parallel one.
from ps06.classification.classifier import DocumentType

Generation = Literal["G1", "G2", "G3"]


# ---------------------------------------------------------------------------
# Roles (Spanish, verbatim from the AEAD "Data Aggregation Agent" goal)
# ---------------------------------------------------------------------------


class Role(str, Enum):
    """Genealogical role in the citizenship-by-descent chain.

    Names and generation mapping are taken verbatim from the source's
    Lineage Engine: G1=Solicitante, G2=Progenitor, G3=Abuelo/a.
    """

    SOLICITANTE = "Solicitante"  # G1 — the applicant / birth-date anchor
    PROGENITOR = "Progenitor"  # G2 — parent named in G1's certificate
    ABUELO = "Abuelo/a"  # G3 — parent named in G2's certificate


#: Role <-> generation is a fixed bijection in the source logic.
ROLE_BY_GENERATION: dict[str, Role] = {
    "G1": Role.SOLICITANTE,
    "G2": Role.PROGENITOR,
    "G3": Role.ABUELO,
}
GENERATION_BY_ROLE: dict[Role, str] = {v: k for k, v in ROLE_BY_GENERATION.items()}


# ---------------------------------------------------------------------------
# The three document schemas (SOLICITUD / ID / CERTIFICADO)
# ---------------------------------------------------------------------------


class SolicitudData(BaseModel):
    """Application form (``tipo_documento: "SOLICITUD"``)."""

    model_config = ConfigDict(frozen=True)

    tipo_anexo: Optional[str] = None
    nombre_solicitante: Optional[str] = None
    apellido_padre: Optional[str] = None
    apellido_madre: Optional[str] = None
    nacionalidad: Optional[str] = None
    estado_civil: Optional[str] = None
    numero_id: Optional[str] = None
    domicilio: Optional[str] = None
    provincia: Optional[str] = None
    pais: Optional[str] = None
    telf_contacto: Optional[str] = None
    email: Optional[str] = None
    nacionalidad_origen_progenitor: Optional[str] = None
    lugar_presentacion: Optional[str] = None
    fecha_presentacion: Optional[str] = None
    op_vecindad: Optional[str] = None
    firma: bool = False


#: canonical attribute -> exact Spanish label in the AEAD SOLICITUD schema.
SOLICITUD_SOURCE_LABELS: dict[str, str] = {
    "tipo_anexo": "Tipo de Anexo",
    "nombre_solicitante": "Nombre del solicitante",
    "apellido_padre": "Apellido del padre",
    "apellido_madre": "Apellido de la madre",
    "nacionalidad": "Nacionalidad",
    "estado_civil": "Estado Civil",
    "numero_id": "Numero de ID",
    "domicilio": "Domicilio",
    "provincia": "Provincia",
    "pais": "País",
    "telf_contacto": "Telf. Contacto",
    "email": "E-mail",
    "nacionalidad_origen_progenitor": "Nacionalidad de origen del progenitor",
    "lugar_presentacion": "Lugar de presentación",
    "fecha_presentacion": "Fecha de presentación",
    "op_vecindad": "Op. Vecindad",
    "firma": "Firma",
}


class IdData(BaseModel):
    """Identity document (``tipo_documento: "ID"``)."""

    model_config = ConfigDict(frozen=True)

    tipo_id: Optional[str] = None
    numero_id: Optional[str] = None
    apellidos: Optional[str] = None
    nombres: Optional[str] = None
    nacionalidad: Optional[str] = None
    sexo: Optional[str] = None
    fecha_nacimiento: Optional[str] = None
    fecha_emision: Optional[str] = None
    fecha_vencimiento: Optional[str] = None


#: canonical attribute -> exact Spanish label in the AEAD ID schema.
ID_SOURCE_LABELS: dict[str, str] = {
    "tipo_id": "Tipo de ID",
    "numero_id": "Número de ID",
    "apellidos": "Apellidos",
    "nombres": "Nombres",
    "nacionalidad": "Nacionalidad",
    "sexo": "Sexo",
    "fecha_nacimiento": "Fecha de nacimiento",
    "fecha_emision": "Fecha de emisión",
    "fecha_vencimiento": "Fecha de vencimiento",
}


class CertificadoData(BaseModel):
    """Birth certificate (``tipo_documento: "CERTIFICADO"``).

    The source schema suffixes every field with a templated ``G_X`` (replaced by
    G1/G2/G3 per document), e.g. ``"Nombre G_X"``, ``"Nombre progenitor 1 G_X"``.
    Because that suffix is dynamic it cannot be a static field label, so these
    canonical fields drop it; :data:`CERTIFICADO_SOURCE_LABELS` records the
    source label template. ``grado_certificado`` carries the document's own
    self-declared generation when present (often absent -> resolved in lineage).
    """

    model_config = ConfigDict(frozen=True)

    grado_certificado: Optional[Generation] = None
    nombre: Optional[str] = None
    fecha_nacimiento: Optional[str] = None
    sexo: Optional[str] = None
    lugar_inscripcion: Optional[str] = None
    fecha_inscripcion: Optional[str] = None
    progenitor1_nombre: Optional[str] = None
    progenitor1_num_doc: Optional[str] = None
    progenitor1_nacionalidad: Optional[str] = None
    progenitor2_nombre: Optional[str] = None
    progenitor2_num_doc: Optional[str] = None
    progenitor2_nacionalidad: Optional[str] = None
    apostillado: bool = False


#: canonical attribute -> exact Spanish label template in the AEAD CERTIFICADO
#: schema (``G_X`` is the source's own generation placeholder).
CERTIFICADO_SOURCE_LABELS: dict[str, str] = {
    "grado_certificado": "grado_certificado",
    "nombre": "Nombre G_X",
    "fecha_nacimiento": "Fecha de nacimiento G_X",
    "sexo": "sexo G_X",
    "lugar_inscripcion": "Lugar de inscripción G_X",
    "fecha_inscripcion": "Fecha de Inscripción G_X",
    "progenitor1_nombre": "Nombre progenitor 1 G_X",
    "progenitor1_num_doc": "Nº Documento progenitor 1 G_X",
    "progenitor1_nacionalidad": "Nacionalidad progenitor 1 G_X",
    "progenitor2_nombre": "Nombre progenitor 2 G_X",
    "progenitor2_num_doc": "Nº Documento progenitor 2 G_X",
    "progenitor2_nacionalidad": "Nacionalidad progenitor 2 G_X",
    "apostillado": "Apostillado G_X",
}


# ---------------------------------------------------------------------------
# Per-document canonical wrapper
# ---------------------------------------------------------------------------


class CanonicalDocument(BaseModel):
    """One document, adapted from its persisted ``OcrResult`` into a typed shape.

    Exactly one of ``solicitud``/``identidad``/``certificado`` is populated for a
    recognized document; all three are ``None`` for ``OTHER``/``EMPTY`` (or a
    ``NOT RECOGNIZED`` document, in the source's terms). ``missing_fields`` names
    the structured fields the rules engine wanted but that were not available in
    the extraction — this is how the engine stays honest about the deferred
    LLM field-extraction: a birth certificate whose structured fields could not
    be populated is *incomplete*, never silently treated as complete.
    """

    model_config = ConfigDict(frozen=True)

    case_id: str
    document_id: str
    source_file: str
    document_type: DocumentType
    generation: Optional[Generation] = None
    solicitud: Optional[SolicitudData] = None
    identidad: Optional[IdData] = None
    certificado: Optional[CertificadoData] = None
    #: Non-empty when structured fields the engine needs were unavailable — the
    #: seam for the deferred birth-cert/ID field-extraction follow-up.
    missing_fields: tuple[str, ...] = ()
    #: Normalized (NFKC + fence-stripped) full text, for text-level fallbacks.
    normalized_text: str = ""


# ---------------------------------------------------------------------------
# Resolved lineage entities
# ---------------------------------------------------------------------------


class Person(BaseModel):
    """A resolved individual in the family graph (output of entity resolution).

    Assembled by :mod:`ps06.rules.lineage` by merging matching records across
    documents. ``fecha_nacimiento`` is the parsed date (``None`` if absent or
    unparseable) so the Lineage Engine can compute generational gaps directly.
    """

    model_config = ConfigDict(frozen=True)

    nombre: Optional[str] = None
    fecha_nacimiento: Optional[date] = None
    lugar_nacimiento: Optional[str] = None
    num_documento: Optional[str] = None
    nacionalidad: Optional[str] = None
    sexo: Optional[str] = None
    role: Optional[Role] = None
    generation: Optional[Generation] = None
    #: document_ids that contributed to this resolved person (audit trail).
    source_document_ids: tuple[str, ...] = ()


class CaseBundle(BaseModel):
    """All of a case's canonical documents, the unit the engine evaluates."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    documents: tuple[CanonicalDocument, ...] = ()

    def by_type(self, document_type: DocumentType) -> tuple[CanonicalDocument, ...]:
        """All documents classified as ``document_type`` (may be empty)."""
        return tuple(d for d in self.documents if d.document_type is document_type)

    @property
    def application(self) -> Optional[CanonicalDocument]:
        """The single application form, or ``None`` if absent.

        If more than one is present the first is returned; multiplicity is a
        data problem the Semaphore Engine surfaces, not something to resolve here.
        """
        apps = self.by_type(DocumentType.APPLICATION)
        return apps[0] if apps else None

    @property
    def birth_certificates(self) -> tuple[CanonicalDocument, ...]:
        return self.by_type(DocumentType.BIRTH_CERT)

    @property
    def identity_documents(self) -> tuple[CanonicalDocument, ...]:
        return self.by_type(DocumentType.ID_DOCUMENT)
