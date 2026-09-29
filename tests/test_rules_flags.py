"""Tests for the Conflict Flag detectors (M-4, step 5)."""

from __future__ import annotations

from datetime import date

from ps06.rules.flags import (
    brecha_generacional,
    documento_cruzado,
    documento_vencido,
    document_incomplete,
    fecha_conflicto,
    id_inconsistente,
    nacionalidad_inconsistente,
    nombre_ambiguo,
    nombre_inconsistente,
)
from ps06.rules.rules_config import GapBand


# --- DOCUMENT_INCOMPLETE ------------------------------------------------------


def test_document_incomplete_true_when_fields_missing():
    assert document_incomplete(("nombre",)) is True


def test_document_incomplete_false_when_nothing_missing():
    assert document_incomplete(()) is False


# --- DOCUMENTO_VENCIDO ---------------------------------------------------------


def test_documento_vencido_true_when_expiry_before_evaluation_date():
    assert documento_vencido(date(2020, 1, 1), date(2026, 9, 29)) is True


def test_documento_vencido_false_when_expiry_in_future():
    assert documento_vencido(date(2030, 1, 1), date(2026, 9, 29)) is False


def test_documento_vencido_false_when_expiry_absent():
    assert documento_vencido(None, date(2026, 9, 29)) is False


# --- NOMBRE_AMBIGUO ------------------------------------------------------------


def test_nombre_ambiguo_true_when_multiple_candidates():
    assert nombre_ambiguo(2) is True


def test_nombre_ambiguo_false_when_single_candidate():
    assert nombre_ambiguo(1) is False


def test_nombre_ambiguo_false_when_zero_candidates():
    assert nombre_ambiguo(0) is False


# --- NOMBRE_INCONSISTENTE ------------------------------------------------------


def test_nombre_inconsistente_false_when_names_fuzzy_match():
    assert nombre_inconsistente(["Ana García", "ana garcia"]) is False


def test_nombre_inconsistente_true_when_names_differ():
    assert nombre_inconsistente(["Ana García", "Maria López"]) is True


def test_nombre_inconsistente_false_with_fewer_than_two_names():
    assert nombre_inconsistente(["Ana García"]) is False
    assert nombre_inconsistente([None, "Ana García"]) is False


# --- ID_INCONSISTENTE ------------------------------------------------------


def test_id_inconsistente_false_when_numbers_match_ignoring_formatting():
    assert id_inconsistente(["12-345.678", "12345678"]) is False


def test_id_inconsistente_true_when_numbers_differ():
    assert id_inconsistente(["12345678", "87654321"]) is True


def test_id_inconsistente_false_with_fewer_than_two_numbers():
    assert id_inconsistente(["12345678", None]) is False


# --- DOCUMENTO_CRUZADO ------------------------------------------------------


def test_documento_cruzado_false_when_one_name():
    assert documento_cruzado(["Ana García", "ana  garcia"]) is False


def test_documento_cruzado_true_when_two_distinct_names():
    assert documento_cruzado(["Ana García", "Maria López"]) is True


# --- FECHA_CONFLICTO ------------------------------------------------------


def test_fecha_conflicto_false_when_dates_agree():
    assert fecha_conflicto([date(1990, 1, 1), date(1990, 1, 1)]) is False


def test_fecha_conflicto_true_when_dates_differ():
    assert fecha_conflicto([date(1990, 1, 1), date(1991, 1, 1)]) is True


def test_fecha_conflicto_false_with_fewer_than_two_dates():
    assert fecha_conflicto([date(1990, 1, 1), None]) is False


# --- NACIONALIDAD_INCONSISTENTE ------------------------------------------------------


def test_nacionalidad_inconsistente_false_when_folded_equal():
    assert nacionalidad_inconsistente(["Boliviana", "boliviana"]) is False


def test_nacionalidad_inconsistente_true_when_differ():
    assert nacionalidad_inconsistente(["Boliviana", "Española"]) is True


# --- BRECHA_GENERACIONAL ------------------------------------------------------


def test_brecha_generacional_false_within_hard_band():
    band = GapBand(min_years=14, max_years=60)
    assert brecha_generacional(date(1970, 1, 1), date(2000, 1, 1), band) is False


def test_brecha_generacional_true_outside_hard_band():
    band = GapBand(min_years=14, max_years=60)
    assert brecha_generacional(date(1970, 1, 1), date(1980, 1, 1), band) is True


def test_brecha_generacional_order_independent():
    band = GapBand(min_years=14, max_years=60)
    assert brecha_generacional(date(2000, 1, 1), date(1970, 1, 1), band) is False
