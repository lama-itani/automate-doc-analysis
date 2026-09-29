"""Report snapshot assembly (M-4, step 8).

The AEAD source's ``save_report_snapshot_tool`` renders a case's decision into
an S0-S3 report shape. This module is that assembly, ported as a pure
formatting layer over an already-computed :class:`~ps06.rules.semaphore.SemaphoreDecision`
and its source :class:`~ps06.rules.schemas.CaseBundle` -- it never re-derives
lineage or severity, it only reshapes fields those modules already computed
into snapshot rows.

S4 (the rendered document itself) is explicitly out of scope here -- it is
not LLM output and is rendered deterministically downstream in M-5.

No I/O, no case/document identity beyond what's already on the inputs.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from ps06.rules.lineage import RoleResolution
from ps06.rules.schemas import CaseBundle, ROLE_BY_GENERATION
from ps06.rules.semaphore import Estado, SemaphoreDecision

_SIN_PROBLEMAS = ("Ninguno",)


class S1Row(BaseModel):
    """Doc-inventory row: Tipo | Descripción y Función | Estado."""

    model_config = ConfigDict(frozen=True)

    tipo: str
    descripcion: str
    estado: str


class S2Row(BaseModel):
    """Per-document row: Documento | Evaluación de credencial | Campos
    faltantes | Consistencia entre documentos | Conflictos detectados."""

    model_config = ConfigDict(frozen=True)

    documento: str
    evaluacion_credencial: str
    campos_faltantes: tuple[str, ...]
    consistencia: str
    conflictos: tuple[str, ...]


class S3Row(BaseModel):
    """Lineage row: Generación | Nombre Completo | Fecha y Lugar de
    Nacimiento | Relación."""

    model_config = ConfigDict(frozen=True)

    generacion: str
    nombre_completo: str
    fecha_lugar_nacimiento: str
    relacion: str


class ReportSnapshot(BaseModel):
    """The full S0-S3 snapshot for one case's evaluation."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    estado: Estado
    justificacion: str
    problemas: tuple[str, ...]
    s1_rows: tuple[S1Row, ...]
    s2_rows: tuple[S2Row, ...]
    s3_rows: tuple[S3Row, ...]
    s3_footnote: Optional[str] = None


def _s1_rows(decision: SemaphoreDecision) -> tuple[S1Row, ...]:
    rows = []
    for check in decision.expected_document_checks:
        generacion = f" ({check.generation})" if check.generation else ""
        rows.append(
            S1Row(
                tipo=check.label,
                descripcion=f"{check.document_type.value}{generacion}",
                estado="Presente" if check.present else "Faltante",
            )
        )
    return tuple(rows)


def _s2_rows(decision: SemaphoreDecision, bundle: CaseBundle) -> tuple[S2Row, ...]:
    docs_by_id = {doc.document_id: doc for doc in bundle.documents}
    rows = []
    for evaluation in decision.document_evaluations:
        doc = docs_by_id.get(evaluation.document_id)
        missing_fields = doc.missing_fields if doc is not None else ()
        rows.append(
            S2Row(
                documento=evaluation.document_id,
                evaluacion_credencial="Con observaciones" if evaluation.flags else "Sin observaciones",
                campos_faltantes=missing_fields,
                consistencia="Con conflictos" if evaluation.flags else "Consistente",
                conflictos=evaluation.flags,
            )
        )
    return tuple(rows)


def _person_fields(role: RoleResolution) -> tuple[str, str]:
    """(nombre_completo, fecha_lugar_nacimiento) for one resolved role."""
    person = role.person
    if person is None:
        return "No determinado", "No determinado"

    nombre = person.nombre or "No determinado"
    fecha = person.fecha_nacimiento.isoformat() if person.fecha_nacimiento else "No determinado"
    lugar = person.lugar_nacimiento or "No determinado"
    return nombre, f"{fecha} / {lugar}"


def _s3_rows(lineage) -> tuple[S3Row, ...]:
    rows = []
    for role in lineage.roles:
        nombre_completo, fecha_lugar = _person_fields(role)
        relacion = ROLE_BY_GENERATION[role.generation].value
        rows.append(
            S3Row(
                generacion=role.generation,
                nombre_completo=nombre_completo,
                fecha_lugar_nacimiento=fecha_lugar,
                relacion=relacion,
            )
        )
    return tuple(rows)


def _s3_footnote(lineage) -> Optional[str]:
    notes = []
    for role in lineage.roles:
        if role.ambiguous:
            notes.append(f"{role.generation}: nombre ambiguo, no se pudo resolver a una única persona.")
        elif role.rejected:
            notes.append(f"{role.generation}: rechazado por brecha generacional fuera de rango.")
    return " ".join(notes) if notes else None


def build_snapshot(decision: SemaphoreDecision, bundle: CaseBundle) -> ReportSnapshot:
    """Assemble the S0-S3 report snapshot from a decision and its case bundle."""
    return ReportSnapshot(
        case_id=decision.case_id,
        estado=decision.estado,
        justificacion=decision.justificacion,
        problemas=decision.problemas if decision.problemas else _SIN_PROBLEMAS,
        s1_rows=_s1_rows(decision),
        s2_rows=_s2_rows(decision, bundle),
        s3_rows=_s3_rows(decision.lineage),
        s3_footnote=_s3_footnote(decision.lineage),
    )
