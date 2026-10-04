"""Tests for text hygiene / normalization primitives (M-4, step 2)."""

from __future__ import annotations

from datetime import date

from ps06.rules.normalize import (
    dates_within_years,
    add_months,
    doc_numbers_match,
    fold_doc_number,
    fold_name,
    fold_nationality,
    names_fuzzy_match,
    names_match,
    nfkc,
    normalize_extracted_text,
    parse_date,
    strip_vlm_artifacts,
    years_between,
)


# --- VLM artifact stripping --------------------------------------------------


def test_strip_vlm_artifacts_removes_code_fence_markers_keeps_content():
    raw = "```markdown\nCertificado de Nacimiento\nNombre: Ana\n```"
    out = strip_vlm_artifacts(raw)
    assert "```" not in out
    assert "Certificado de Nacimiento" in out
    assert "Nombre: Ana" in out


def test_strip_vlm_artifacts_removes_invented_image_link():
    raw = "Texto real del documento\n![](https://example.com/image.png)\nMas texto"
    out = strip_vlm_artifacts(raw)
    assert "example.com" not in out
    assert "![" not in out
    assert "Texto real del documento" in out
    assert "Mas texto" in out


def test_strip_vlm_artifacts_noop_on_plain_text_layer_output():
    raw = "--- Page 1 ---\nSOLICITUD DE NACIONALIDAD\nNombre: Ana"
    assert strip_vlm_artifacts(raw) == raw


def test_strip_vlm_artifacts_idempotent():
    raw = "```\nCertificado\n```\n![](http://x/y.png)"
    once = strip_vlm_artifacts(raw)
    twice = strip_vlm_artifacts(once)
    assert once == twice


# --- NFKC / ligatures ---------------------------------------------------------


def test_nfkc_decomposes_ligature():
    # U+FB01 LATIN SMALL LIGATURE FI
    ligatured = "Certiﬁcado"
    assert "ﬁ" in ligatured
    assert nfkc(ligatured) == "Certificado"


def test_normalize_extracted_text_combines_fence_strip_and_nfkc():
    raw = "```markdown\nCertiﬁcado de Nacimiento\n```"
    out = normalize_extracted_text(raw)
    assert "```" not in out
    assert "Certificado de Nacimiento" in out


# --- Name folding -------------------------------------------------------------


def test_fold_name_case_and_accent_insensitive():
    assert fold_name("García") == fold_name("garcia")
    assert fold_name("ANA GARCÍA") == fold_name("ana garcia")


def test_fold_name_collapses_whitespace():
    assert fold_name("Ana   García") == fold_name("Ana García")


def test_fold_name_empty_and_none_return_empty_string():
    assert fold_name(None) == ""
    assert fold_name("") == ""


def test_fold_nationality_resolves_iso_code_to_name():
    assert fold_nationality("ESP") == fold_nationality("Española")


def test_fold_nationality_passes_through_unmapped_values():
    assert fold_nationality("Boliviana") == fold_name("Boliviana")


def test_names_match_requires_nonempty_and_equal():
    assert names_match("Ana García", "ana garcia")
    assert not names_match(None, None)
    assert not names_match("", "")
    assert not names_match("Ana", "Luis")


def test_names_fuzzy_match_handles_subset_tokens():
    assert names_fuzzy_match("Ana García López", "Ana García")
    assert names_fuzzy_match("Ana García", "Ana García López")
    assert not names_fuzzy_match("Ana García", "Luis García")
    assert not names_fuzzy_match(None, "Ana")


def test_names_fuzzy_match_exact_after_fold_is_still_a_match():
    assert names_fuzzy_match("García", "garcia")


# --- Document number folding --------------------------------------------------


def test_fold_doc_number_strips_formatting_and_uppercases():
    assert fold_doc_number("x-123.456 789") == "X123456789"
    assert fold_doc_number("X123456789") == "X123456789"


def test_fold_doc_number_empty_and_none():
    assert fold_doc_number(None) == ""
    assert fold_doc_number("") == ""


def test_doc_numbers_match():
    assert doc_numbers_match("X-1234567", "x 1234567")
    assert not doc_numbers_match(None, None)
    assert not doc_numbers_match("X1234567", "X7654321")


# --- Date parsing --------------------------------------------------------------


def test_parse_date_supports_slash_and_iso_and_dash_and_dot_formats():
    assert parse_date("01/05/1990") == date(1990, 5, 1)
    assert parse_date("1990-05-01") == date(1990, 5, 1)
    assert parse_date("01-05-1990") == date(1990, 5, 1)
    assert parse_date("01.05.1990") == date(1990, 5, 1)


def test_parse_date_supports_space_separated_format():
    assert parse_date("05 07 2026") == date(2026, 7, 5)


def test_parse_date_supports_spanish_month_abbreviation_with_slashes():
    assert parse_date("30/Nov/2015") == date(2015, 11, 30)


def test_parse_date_supports_spanish_de_date_words():
    assert parse_date("30 de noviembre de 2015") == date(2015, 11, 30)


def test_parse_date_supports_fully_spelled_out_spanish_date():
    assert parse_date("27 SEPTIEMBRE MIL NOVECIENTOS SESENTA") == date(1960, 9, 27)


def test_parse_date_returns_none_for_unparseable_or_absent():
    assert parse_date(None) is None
    assert parse_date("") is None
    assert parse_date("not a date") is None
    assert parse_date("May 1, 1990") is None


def test_parse_date_never_raises_on_garbage():
    assert parse_date("32/13/9999") is None


# --- Year math -----------------------------------------------------------------


def test_years_between_whole_years_before_and_after_birthday():
    assert years_between(date(1990, 5, 1), date(2020, 5, 1)) == 30
    assert years_between(date(1990, 5, 1), date(2020, 4, 30)) == 29
    assert years_between(date(1990, 5, 1), date(2020, 5, 2)) == 30


def test_dates_within_years_true_and_false():
    assert dates_within_years(date(1990, 1, 1), date(1990, 6, 1), tolerance_years=1)
    assert dates_within_years(date(1990, 1, 1), date(1991, 1, 1), tolerance_years=1)
    assert not dates_within_years(date(1990, 1, 1), date(1992, 1, 2), tolerance_years=1)


def test_dates_within_years_none_never_matches():
    assert not dates_within_years(None, date(1990, 1, 1), tolerance_years=1)
    assert not dates_within_years(None, None, tolerance_years=1)


def test_dates_within_years_order_independent():
    a, b = date(1990, 1, 1), date(1988, 6, 1)
    assert dates_within_years(a, b, tolerance_years=2) == dates_within_years(b, a, tolerance_years=2)


# --- Month arithmetic ------------------------------------------------------


def test_add_months_within_same_year():
    assert add_months(date(2025, 10, 1), 6) == date(2026, 4, 1)


def test_add_months_rolls_over_year_boundary():
    assert add_months(date(2025, 10, 15), 3) == date(2026, 1, 15)


def test_add_months_clamps_day_to_shorter_month():
    assert add_months(date(2025, 1, 31), 1) == date(2025, 2, 28)
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)


def test_add_months_zero_is_identity():
    assert add_months(date(2025, 10, 4), 0) == date(2025, 10, 4)
