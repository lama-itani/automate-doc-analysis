"""Tests for the deterministic rules configuration (M-4, step 3)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ps06.classification.classifier import DocumentType
from ps06.rules.rules_config import (
    EntityResolutionConfig,
    ExpectedDocument,
    GapBand,
    GapBandsConfig,
    RulesConfig,
)

RULES_YAML_PATH = Path(__file__).resolve().parent.parent / "ps06" / "rules" / "rules.yaml"


# --- Defaults, no file involved ---------------------------------------------


def test_default_config_constructs_with_baked_in_values():
    cfg = RulesConfig()
    assert cfg.gap_bands.typical == GapBand(min_years=20, max_years=35)
    assert cfg.gap_bands.hard == GapBand(min_years=14, max_years=60)
    assert cfg.entity_resolution.min_criteria == 2
    assert cfg.entity_resolution.total_criteria == 3
    assert cfg.entity_resolution.dob_tolerance_years == 1


def test_default_config_is_frozen():
    cfg = RulesConfig()
    with pytest.raises(ValidationError):
        cfg.entity_resolution = EntityResolutionConfig()  # type: ignore[misc]


# --- from_yaml matches defaults ---------------------------------------------


def test_from_yaml_loads_checked_in_file_matches_defaults():
    cfg = RulesConfig.from_yaml(RULES_YAML_PATH)
    assert cfg == RulesConfig()


def test_from_yaml_accepts_str_path():
    cfg = RulesConfig.from_yaml(str(RULES_YAML_PATH))
    assert cfg == RulesConfig()


# --- 5-expected-documents structure ------------------------------------------


def test_expected_documents_has_five_entries_in_order():
    cfg = RulesConfig()
    labels = [d.label for d in cfg.expected_documents]
    assert labels == [
        "Solicitud Principal",
        "Identificación del Solicitante",
        "Certificado de Nacimiento del Solicitante",
        "Certificado de Nacimiento del Progenitor",
        "Certificado de Nacimiento Español de origen",
    ]


def test_expected_documents_generation_mapping_is_correct():
    cfg = RulesConfig()
    app, id_doc, g1, g2, g3 = cfg.expected_documents

    assert app.document_type is DocumentType.APPLICATION
    assert app.generation is None

    assert id_doc.document_type is DocumentType.ID_DOCUMENT
    assert id_doc.generation is None

    assert g1.document_type is DocumentType.BIRTH_CERT
    assert g1.generation == "G1"
    assert g2.document_type is DocumentType.BIRTH_CERT
    assert g2.generation == "G2"
    assert g3.document_type is DocumentType.BIRTH_CERT
    assert g3.generation == "G3"


def test_expected_document_generation_rejects_invalid_value():
    with pytest.raises(ValidationError):
        ExpectedDocument(
            label="x", document_type=DocumentType.BIRTH_CERT, generation="G9"
        )  # type: ignore[arg-type]


# --- Fail-fast on malformed/missing YAML (no silent partial config) --------


def test_from_yaml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RulesConfig.from_yaml(tmp_path / "nope.yaml")


def test_from_yaml_empty_file_raises(tmp_path):
    bad = tmp_path / "empty.yaml"
    bad.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        RulesConfig.from_yaml(bad)


def test_from_yaml_malformed_yaml_syntax_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("gap_bands: {typical: [unclosed\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        RulesConfig.from_yaml(bad)


def test_from_yaml_unknown_top_level_key_raises(tmp_path):
    bad = tmp_path / "unknown_key.yaml"
    bad.write_text(yaml.safe_dump({"bogus_key": 1}), encoding="utf-8")
    with pytest.raises(ValidationError):
        RulesConfig.from_yaml(bad)


def test_from_yaml_wrong_type_for_known_field_raises(tmp_path):
    bad = tmp_path / "wrong_type.yaml"
    bad.write_text(
        yaml.safe_dump({"entity_resolution": {"min_criteria": "not-a-number"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        RulesConfig.from_yaml(bad)


def test_from_yaml_missing_nested_key_uses_default_not_error(tmp_path):
    # Omitting a nested section/field entirely is a supported partial
    # override (falls back to the Python default), not a "no silent
    # failures" violation — this test pins that distinction down explicitly
    # so it isn't mistaken for an oversight later.
    partial = tmp_path / "partial.yaml"
    partial.write_text(
        yaml.safe_dump({"entity_resolution": {"min_criteria": 3}}), encoding="utf-8"
    )
    cfg = RulesConfig.from_yaml(partial)
    assert cfg.entity_resolution.min_criteria == 3
    assert cfg.entity_resolution.dob_tolerance_years == 1  # fell back to default
    assert cfg.gap_bands == GapBandsConfig()  # entire section fell back to default


# --- Severity mapping sanity --------------------------------------------------


def test_severity_rojo_contains_all_seven_binding_flags():
    cfg = RulesConfig()
    assert set(cfg.severity.rojo) == {
        "DOCUMENTO_VENCIDO",
        "NOMBRE_AMBIGUO",
        "NOMBRE_INCONSISTENTE",
        "ID_INCONSISTENTE",
        "DOCUMENTO_CRUZADO",
        "FECHA_CONFLICTO",
        "BRECHA_GENERACIONAL",
    }


def test_severity_rojo_and_amarillo_are_disjoint():
    cfg = RulesConfig()
    assert set(cfg.severity.rojo).isdisjoint(cfg.severity.amarillo)
