"""Entity resolution + generational role assignment (M-4, step 6).

The AEAD source's "Lineage Engine" resolves who's who across a case's
documents (a birth certificate's ``progenitor1_*``/``progenitor2_*`` fields
name a person who may also appear as their own G1/G2 applicant elsewhere) and
walks G1 (Solicitante) -> G2 (Progenitor) -> G3 (Abuelo/a), checking each
generational hop against the "hard" gap band before accepting it.

This module owns entity resolution (2-of-3 matching, union-find grouping) and
every conflict flag that requires grouping raw values by resolved person
first: ``NOMBRE_AMBIGUO``, ``NOMBRE_INCONSISTENTE``, ``ID_INCONSISTENTE``,
``DOCUMENTO_CRUZADO``, ``FECHA_CONFLICTO``, ``BRECHA_GENERACIONAL``,
``NACIONALIDAD_INCONSISTENTE``. :mod:`ps06.rules.flags` only supplies the
pure per-value predicates; :mod:`ps06.rules.semaphore` (step 7) only
classifies severity from the flags this module raises — it never re-derives
person-level grouping.

Two design decisions confirmed with the user, not dictated by the extracted
source:

1. **Ambiguous merge**: if candidate records fall into more than one
   match-graph component (i.e. the pool doesn't collapse to a single
   resolved person), the role is never force-merged into a "best" match —
   ``person=None``, ``ambiguous=True``, and ``NOMBRE_AMBIGUO`` is raised.
2. **Corroboration**: a resolved person needs >=2 distinct source documents
   to be usable as VERDE-grade evidence; that gate is enforced in
   :mod:`ps06.rules.semaphore`, but this module always records
   ``corroborating_document_ids`` so the gate has something to check.

No I/O, no case/document identity beyond what's in ``CaseBundle`` already.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

from pydantic import BaseModel, ConfigDict

from ps06.rules import flags
from ps06.rules.normalize import (
    dates_within_years,
    doc_numbers_match,
    fold_doc_number,
    names_fuzzy_match,
    parse_date,
)
from ps06.rules.rules_config import EntityResolutionConfig, GapBand, RulesConfig
from ps06.rules.schemas import (
    CanonicalDocument,
    CaseBundle,
    Generation,
    Person,
    ROLE_BY_GENERATION,
)


class _CandidateRecord:
    """One person-shaped record extracted from a single document field group.

    Bridges :class:`SolicitudData`/:class:`IdData`/:class:`CertificadoData`'s
    incompatible field names/raw-string dates into the one shape entity
    resolution operates on. Not a pydantic model (purely internal, built and
    discarded within a single :func:`resolve_lineage` call) — a plain
    ``__slots__``-free object is enough and avoids paying validation cost for
    every candidate record in a case.
    """

    __slots__ = (
        "name",
        "fecha_nacimiento",
        "num_documento",
        "nacionalidad",
        "sexo",
        "lugar_nacimiento",
        "source_document_id",
    )

    def __init__(
        self,
        *,
        name: Optional[str],
        fecha_nacimiento: Optional[date],
        num_documento: Optional[str],
        nacionalidad: Optional[str],
        sexo: Optional[str],
        lugar_nacimiento: Optional[str],
        source_document_id: str,
    ) -> None:
        self.name = name
        self.fecha_nacimiento = fecha_nacimiento
        self.num_documento = num_documento
        self.nacionalidad = nacionalidad
        self.sexo = sexo
        self.lugar_nacimiento = lugar_nacimiento
        self.source_document_id = source_document_id

    @property
    def is_empty(self) -> bool:
        """True if every substantive field is absent (nothing to resolve on)."""
        return not (
            self.name
            or self.fecha_nacimiento
            or self.num_documento
            or self.nacionalidad
            or self.sexo
            or self.lugar_nacimiento
        )


class RoleResolution(BaseModel):
    """The outcome of resolving one generation's candidate pool."""

    model_config = ConfigDict(frozen=True)

    generation: Generation
    person: Optional[Person] = None
    rejected: bool = False
    ambiguous: bool = False
    flags: tuple[str, ...] = ()
    corroborating_document_ids: tuple[str, ...] = ()


class LineageResult(BaseModel):
    """The full G1/G2/G3 resolution for one case."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    g1: RoleResolution
    g2: RoleResolution
    g3: RoleResolution
    #: Flags inherently spanning roles rather than belonging to one — today
    #: only ``DOCUMENTO_CRUZADO`` (one document number claimed by >1 name).
    cross_case_flags: tuple[str, ...] = ()

    @property
    def roles(self) -> tuple[RoleResolution, RoleResolution, RoleResolution]:
        return (self.g1, self.g2, self.g3)

    @property
    def all_flags(self) -> frozenset[str]:
        result: set[str] = set(self.cross_case_flags)
        for role in self.roles:
            result.update(role.flags)
        return frozenset(result)


def _join_name(*parts: Optional[str]) -> Optional[str]:
    present = [p for p in parts if p]
    return " ".join(present) if present else None


def _records_from_solicitud(doc: CanonicalDocument) -> tuple[_CandidateRecord, ...]:
    """The application form's own applicant (G1) record, if present."""
    solicitud = doc.solicitud
    if solicitud is None:
        return ()
    return (
        _CandidateRecord(
            name=_join_name(
                solicitud.nombre_solicitante,
                solicitud.apellido_padre,
                solicitud.apellido_madre,
            ),
            fecha_nacimiento=None,
            num_documento=solicitud.numero_id,
            nacionalidad=solicitud.nacionalidad,
            sexo=None,
            lugar_nacimiento=None,
            source_document_id=doc.document_id,
        ),
    )


def _records_from_id(doc: CanonicalDocument) -> tuple[_CandidateRecord, ...]:
    """The identity document's own holder record."""
    identidad = doc.identidad
    if identidad is None:
        return ()
    return (
        _CandidateRecord(
            name=_join_name(identidad.nombres, identidad.apellidos),
            fecha_nacimiento=parse_date(identidad.fecha_nacimiento),
            num_documento=identidad.numero_id,
            nacionalidad=identidad.nacionalidad,
            sexo=identidad.sexo,
            lugar_nacimiento=None,
            source_document_id=doc.document_id,
        ),
    )


def _records_from_certificado(
    doc: CanonicalDocument, *, use_progenitor: Optional[int] = None
) -> tuple[_CandidateRecord, ...]:
    """A birth certificate's subject record, or one named progenitor's record.

    ``use_progenitor=None`` returns the certificate's own subject (the person
    whose birth it certifies). ``use_progenitor=1``/``2`` returns the named
    ``progenitor{1,2}_*`` record instead — that person has no birth date/sex/
    place on this document, only a name, document number, and nationality.
    """
    certificado = doc.certificado
    if certificado is None:
        return ()

    if use_progenitor is None:
        return (
            _CandidateRecord(
                name=certificado.nombre,
                fecha_nacimiento=parse_date(certificado.fecha_nacimiento),
                num_documento=None,
                nacionalidad=None,
                sexo=certificado.sexo,
                lugar_nacimiento=certificado.lugar_inscripcion,
                source_document_id=doc.document_id,
            ),
        )

    if use_progenitor == 1:
        name, num_doc, nacionalidad = (
            certificado.progenitor1_nombre,
            certificado.progenitor1_num_doc,
            certificado.progenitor1_nacionalidad,
        )
    else:
        name, num_doc, nacionalidad = (
            certificado.progenitor2_nombre,
            certificado.progenitor2_num_doc,
            certificado.progenitor2_nacionalidad,
        )

    if not (name or num_doc or nacionalidad):
        return ()

    return (
        _CandidateRecord(
            name=name,
            fecha_nacimiento=None,
            num_documento=num_doc,
            nacionalidad=nacionalidad,
            sexo=None,
            lugar_nacimiento=None,
            source_document_id=doc.document_id,
        ),
    )


class _UnionFind:
    """Minimal union-find over integer indices, for match-graph components."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _match_score(
    a: _CandidateRecord, b: _CandidateRecord, *, dob_tolerance_years: int
) -> int:
    """Count of the 3 entity-resolution criteria ``a``/``b`` agree on."""
    score = 0
    if names_fuzzy_match(a.name, b.name):
        score += 1
    if dates_within_years(a.fecha_nacimiento, b.fecha_nacimiento, tolerance_years=dob_tolerance_years):
        score += 1
    if doc_numbers_match(a.num_documento, b.num_documento):
        score += 1
    return score


def _merge_person(
    generation: Generation, component: list[_CandidateRecord]
) -> tuple[Person, tuple[str, ...]]:
    """Merge one match-graph component into a single :class:`Person`.

    Representative field values are the first non-``None`` value across the
    component's records — conflicts (if any) are recorded as flags, never
    silently dropped or averaged away.
    """
    def first(values: Iterable[Optional[str]]) -> Optional[str]:
        for v in values:
            if v:
                return v
        return None

    def first_date(values: Iterable[Optional[date]]) -> Optional[date]:
        for v in values:
            if v is not None:
                return v
        return None

    person = Person(
        nombre=first(r.name for r in component),
        fecha_nacimiento=first_date(r.fecha_nacimiento for r in component),
        lugar_nacimiento=first(r.lugar_nacimiento for r in component),
        num_documento=first(r.num_documento for r in component),
        nacionalidad=first(r.nacionalidad for r in component),
        sexo=first(r.sexo for r in component),
        role=ROLE_BY_GENERATION[generation],
        generation=generation,
        source_document_ids=tuple(
            dict.fromkeys(r.source_document_id for r in component)
        ),
    )

    hit_flags: list[str] = []
    if flags.nombre_inconsistente(r.name for r in component):
        hit_flags.append("NOMBRE_INCONSISTENTE")
    if flags.fecha_conflicto(r.fecha_nacimiento for r in component):
        hit_flags.append("FECHA_CONFLICTO")
    if flags.id_inconsistente(r.num_documento for r in component):
        hit_flags.append("ID_INCONSISTENTE")
    if flags.nacionalidad_inconsistente(r.nacionalidad for r in component):
        hit_flags.append("NACIONALIDAD_INCONSISTENTE")

    return person, tuple(hit_flags)


def _resolve_role(
    generation: Generation,
    records: Iterable[_CandidateRecord],
    er_config: EntityResolutionConfig,
) -> RoleResolution:
    """Resolve one generation's candidate pool into a :class:`RoleResolution`."""
    pool = [r for r in records if not r.is_empty]

    if not pool:
        return RoleResolution(generation=generation, person=None)

    uf = _UnionFind(len(pool))
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if _match_score(pool[i], pool[j], dob_tolerance_years=er_config.dob_tolerance_years) >= er_config.min_criteria:
                uf.union(i, j)

    components: dict[int, list[_CandidateRecord]] = {}
    for idx, record in enumerate(pool):
        components.setdefault(uf.find(idx), []).append(record)

    if len(components) > 1:
        return RoleResolution(
            generation=generation,
            person=None,
            ambiguous=True,
            flags=("NOMBRE_AMBIGUO",),
        )

    (component,) = components.values()
    person, hit_flags = _merge_person(generation, component)
    corroborating_ids = tuple(dict.fromkeys(r.source_document_id for r in component))
    return RoleResolution(
        generation=generation,
        person=person,
        flags=hit_flags,
        corroborating_document_ids=corroborating_ids,
    )


def _certificados_for_generation(
    bundle: CaseBundle, generation: Generation
) -> tuple[CanonicalDocument, ...]:
    return tuple(
        doc for doc in bundle.birth_certificates if doc.generation == generation
    )


def _child_pool(
    parent_resolution: RoleResolution,
    parent_certs: tuple[CanonicalDocument, ...],
    own_certs: tuple[CanonicalDocument, ...],
) -> list[_CandidateRecord]:
    """Build the next generation's candidate pool.

    If the parent generation resolved cleanly, its own birth certificate(s)
    name this generation as progenitor 1/2 — safe to pull in. If the parent
    is unresolved or ambiguous, that progenitor information can't be trusted
    to belong to any single person, so only this generation's own
    self-tagged certificate(s) are used.
    """
    pool: list[_CandidateRecord] = []
    if parent_resolution.person is not None and not parent_resolution.ambiguous:
        for cert in parent_certs:
            pool.extend(_records_from_certificado(cert, use_progenitor=1))
            pool.extend(_records_from_certificado(cert, use_progenitor=2))
    for cert in own_certs:
        pool.extend(_records_from_certificado(cert))
    return pool


def _apply_generational_gap_check(
    parent: RoleResolution, child: RoleResolution, hard_band: GapBand
) -> RoleResolution:
    """Reject ``child`` if its gap from ``parent`` falls outside the hard band.

    Skipped entirely (no flag, no rejection) if either resolved person's
    birth date is unknown — an absent DOB is not evidence of a bad gap.
    """
    if (
        parent.person is None
        or child.person is None
        or parent.person.fecha_nacimiento is None
        or child.person.fecha_nacimiento is None
    ):
        return child

    if flags.brecha_generacional(
        parent.person.fecha_nacimiento, child.person.fecha_nacimiento, hard_band
    ):
        return child.model_copy(
            update={
                "rejected": True,
                "flags": child.flags + ("BRECHA_GENERACIONAL",),
            }
        )
    return child


def _cross_role_flags(all_records: Iterable[_CandidateRecord]) -> tuple[str, ...]:
    """``DOCUMENTO_CRUZADO``: one document number shared by >=2 distinct names."""
    by_doc_number: dict[str, list[_CandidateRecord]] = {}
    for record in all_records:
        folded = fold_doc_number(record.num_documento)
        if folded:
            by_doc_number.setdefault(folded, []).append(record)

    for group in by_doc_number.values():
        if len(group) > 1 and flags.documento_cruzado(r.name for r in group):
            return ("DOCUMENTO_CRUZADO",)
    return ()


def resolve_lineage(bundle: CaseBundle, config: RulesConfig) -> LineageResult:
    """Resolve a case's G1/G2/G3 roles and their person-level conflict flags."""
    er_config = config.entity_resolution

    g1_certs = _certificados_for_generation(bundle, "G1")
    g1_pool: list[_CandidateRecord] = []
    if bundle.application is not None:
        g1_pool.extend(_records_from_solicitud(bundle.application))
    for doc in bundle.identity_documents:
        g1_pool.extend(_records_from_id(doc))
    for cert in g1_certs:
        g1_pool.extend(_records_from_certificado(cert))
    g1 = _resolve_role("G1", g1_pool, er_config)

    g2_certs = _certificados_for_generation(bundle, "G2")
    g2_pool = _child_pool(g1, g1_certs, g2_certs)
    g2 = _resolve_role("G2", g2_pool, er_config)
    g2 = _apply_generational_gap_check(g1, g2, config.gap_bands.hard)

    g3_certs = _certificados_for_generation(bundle, "G3")
    g3_pool = _child_pool(g2, g2_certs, g3_certs)
    g3 = _resolve_role("G3", g3_pool, er_config)
    g3 = _apply_generational_gap_check(g2, g3, config.gap_bands.hard)

    cross_case_flags = _cross_role_flags(g1_pool + g2_pool + g3_pool)

    return LineageResult(
        case_id=bundle.case_id,
        g1=g1,
        g2=g2,
        g3=g3,
        cross_case_flags=cross_case_flags,
    )
