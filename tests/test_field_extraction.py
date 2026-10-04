"""Tests for ps06.classification.field_extraction.

Exercises extract_fields() entirely against a fake LLM client — no network,
no live Cloudera/Workbench credentials.
"""

from __future__ import annotations

import json

from ps06.classification.classifier import DocumentType
from ps06.classification.field_extraction import (
    FieldExtractionConfig,
    extract_fields,
)
from tests.conftest import FakeFieldExtractionClient


def _config(**overrides) -> FieldExtractionConfig:
    return FieldExtractionConfig(**overrides)


# ---------------------------------------------------------------------------
# Routing: only ID_DOCUMENT / BIRTH_CERT call the LLM
# ---------------------------------------------------------------------------


def test_application_short_circuits_without_calling_llm():
    client = FakeFieldExtractionClient()
    result = extract_fields("some text", DocumentType.APPLICATION, _config(), client)

    assert result.fields == {}
    assert result.raw_response == ""
    assert client.calls == []


def test_other_short_circuits_without_calling_llm():
    client = FakeFieldExtractionClient()
    result = extract_fields("some text", DocumentType.OTHER, _config(), client)

    assert result.fields == {}
    assert client.calls == []


def test_empty_short_circuits_without_calling_llm():
    client = FakeFieldExtractionClient()
    result = extract_fields("", DocumentType.EMPTY, _config(), client)

    assert result.fields == {}
    assert client.calls == []


def test_id_document_calls_llm_with_id_prompt():
    client = FakeFieldExtractionClient(response="{}")
    extract_fields("PASAPORTE", DocumentType.ID_DOCUMENT, _config(), client)

    assert len(client.calls) == 1
    assert "Tipo de ID" in client.calls[0]["prompt"]


def test_birth_cert_calls_llm_with_certificado_prompt():
    client = FakeFieldExtractionClient(response="{}")
    extract_fields("CERTIFICADO DE NACIMIENTO", DocumentType.BIRTH_CERT, _config(), client)

    assert len(client.calls) == 1
    assert "grado_certificado" in client.calls[0]["prompt"]


# ---------------------------------------------------------------------------
# Happy-path parsing
# ---------------------------------------------------------------------------


def test_id_fields_resolved_from_spanish_labels():
    payload = {
        "Tipo de ID": "Pasaporte",
        "Número de ID": "XDD882743",
        "Apellidos": "Minaya Sainz",
        "Nombres": "Ricardo",
        "Nacionalidad": "Española",
        "Sexo": "M",
        "Fecha de nacimiento": "01/02/1990",
        "Fecha de emisión": "01/01/2020",
        "Fecha de vencimiento": "01/01/2030",
    }
    client = FakeFieldExtractionClient(response=json.dumps(payload))
    result = extract_fields("PASAPORTE text", DocumentType.ID_DOCUMENT, _config(), client)

    assert result.fields == {
        "tipo_id": "Pasaporte",
        "numero_id": "XDD882743",
        "apellidos": "Minaya Sainz",
        "nombres": "Ricardo",
        "nacionalidad": "Española",
        "sexo": "M",
        "fecha_nacimiento": "01/02/1990",
        "fecha_emision": "01/01/2020",
        "fecha_vencimiento": "01/01/2030",
    }
    assert result.raw_response == json.dumps(payload)


def test_certificado_fields_resolved_from_spanish_labels():
    payload = {
        "grado_certificado": "G_2",
        "Nombre G_X": "Maria Lopez",
        "Fecha de nacimiento G_X": "05/05/1965",
        "sexo G_X": "F",
        "Lugar de inscripción G_X": "Madrid",
        "Fecha de Inscripción G_X": "10/05/1965",
        "Nombre progenitor 1 G_X": "Jose Lopez",
        "Nº Documento progenitor 1 G_X": "12345678",
        "Nacionalidad progenitor 1 G_X": "Española",
        "Nombre progenitor 2 G_X": "Carmen Ruiz",
        "Nº Documento progenitor 2 G_X": "87654321",
        "Nacionalidad progenitor 2 G_X": "Española",
        "Apostillado G_X": "APOSTILLE / La Haya",
    }
    client = FakeFieldExtractionClient(response=json.dumps(payload))
    result = extract_fields("CERTIFICADO text", DocumentType.BIRTH_CERT, _config(), client)

    assert result.fields["grado_certificado"] == "G2"
    assert result.fields["nombre"] == "Maria Lopez"
    assert result.fields["fecha_nacimiento"] == "05/05/1965"
    assert result.fields["sexo"] == "F"
    assert result.fields["lugar_inscripcion"] == "Madrid"
    assert result.fields["fecha_inscripcion"] == "10/05/1965"
    assert result.fields["progenitor1_nombre"] == "Jose Lopez"
    assert result.fields["progenitor1_num_doc"] == "12345678"
    assert result.fields["progenitor1_nacionalidad"] == "Española"
    assert result.fields["progenitor2_nombre"] == "Carmen Ruiz"
    assert result.fields["progenitor2_num_doc"] == "87654321"
    assert result.fields["progenitor2_nacionalidad"] == "Española"
    assert result.fields["apostillado"] == "APOSTILLE / La Haya"


def test_label_matching_is_case_and_whitespace_insensitive():
    payload = {"  tipo de id  ": "Cedula"}
    client = FakeFieldExtractionClient(response=json.dumps(payload))
    result = extract_fields("text", DocumentType.ID_DOCUMENT, _config(), client)

    assert result.fields == {"tipo_id": "Cedula"}


# ---------------------------------------------------------------------------
# Parsing resilience — never raises, degrades to missing
# ---------------------------------------------------------------------------


def test_non_json_response_yields_empty_fields():
    client = FakeFieldExtractionClient(response="I cannot extract this document's fields.")
    result = extract_fields("text", DocumentType.ID_DOCUMENT, _config(), client)

    assert result.fields == {}
    assert result.raw_response == "I cannot extract this document's fields."


def test_fenced_json_response_is_parsed():
    payload = {"Tipo de ID": "Pasaporte", "Número de ID": "XDD882743"}
    client = FakeFieldExtractionClient(response=f"```json\n{json.dumps(payload)}\n```")
    result = extract_fields("text", DocumentType.ID_DOCUMENT, _config(), client)

    assert result.fields == {"tipo_id": "Pasaporte", "numero_id": "XDD882743"}


def test_json_array_response_yields_empty_fields():
    client = FakeFieldExtractionClient(response=json.dumps(["Pasaporte", "XDD882743"]))
    result = extract_fields("text", DocumentType.ID_DOCUMENT, _config(), client)

    assert result.fields == {}


def test_json_scalar_response_yields_empty_fields():
    client = FakeFieldExtractionClient(response="42")
    result = extract_fields("text", DocumentType.ID_DOCUMENT, _config(), client)

    assert result.fields == {}


def test_empty_string_values_treated_as_missing():
    payload = {"Tipo de ID": "   ", "Nombres": "Ricardo"}
    client = FakeFieldExtractionClient(response=json.dumps(payload))
    result = extract_fields("text", DocumentType.ID_DOCUMENT, _config(), client)

    assert result.fields == {"nombres": "Ricardo"}


def test_unknown_label_keys_are_ignored():
    payload = {"Not A Real Field": "value", "Nombres": "Ricardo"}
    client = FakeFieldExtractionClient(response=json.dumps(payload))
    result = extract_fields("text", DocumentType.ID_DOCUMENT, _config(), client)

    assert result.fields == {"nombres": "Ricardo"}


def test_non_string_value_is_ignored():
    payload = {"Nombres": 123, "Apellidos": "Minaya"}
    client = FakeFieldExtractionClient(response=json.dumps(payload))
    result = extract_fields("text", DocumentType.ID_DOCUMENT, _config(), client)

    assert result.fields == {"apellidos": "Minaya"}


# ---------------------------------------------------------------------------
# grado_certificado edge cases
# ---------------------------------------------------------------------------


def test_grado_certificado_accepts_g1_g2_g3_variants():
    for raw_value, expected in [("G1", "G1"), ("G_1", "G1"), ("g 1", "G1"), ("G3", "G3")]:
        payload = {"grado_certificado": raw_value}
        client = FakeFieldExtractionClient(response=json.dumps(payload))
        result = extract_fields("text", DocumentType.BIRTH_CERT, _config(), client)
        assert result.fields.get("grado_certificado") == expected, raw_value


def test_grado_certificado_unrecognized_value_is_dropped():
    payload = {"grado_certificado": "unclear", "Nombre G_X": "Maria"}
    client = FakeFieldExtractionClient(response=json.dumps(payload))
    result = extract_fields("text", DocumentType.BIRTH_CERT, _config(), client)

    assert "grado_certificado" not in result.fields
    assert result.fields["nombre"] == "Maria"


# ---------------------------------------------------------------------------
# Config contract
# ---------------------------------------------------------------------------


def test_field_extraction_config_frozen_and_defaults():
    config = _config()

    assert config.max_tokens == 500
    assert config.temperature == 0.0
    assert config.max_text_chars == 4000
    try:
        config.max_tokens = 999  # type: ignore[misc]
        assert False, "expected frozen model to reject attribute assignment"
    except Exception:
        pass


def test_max_text_chars_truncates_prompt_input():
    client = FakeFieldExtractionClient(response="{}")
    long_text = "PASAPORTE " * 1000

    extract_fields(long_text, DocumentType.ID_DOCUMENT, _config(max_text_chars=50), client)

    sent_prompt = client.calls[0]["prompt"]
    assert long_text[:50] in sent_prompt
    assert long_text not in sent_prompt
