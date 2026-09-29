"""Tests for ps06.classification.classifier.

Exercises classify() entirely against a fake LLM client — no network, no
live Cloudera/Workbench credentials.
"""

from __future__ import annotations

from ps06.classification.classifier import (
    ClassificationConfig,
    DocumentType,
    classify,
)
from tests.conftest import FakeClassificationClient


def _config(**overrides) -> ClassificationConfig:
    return ClassificationConfig(**overrides)


# ---------------------------------------------------------------------------
# Happy-path classification per category
# ---------------------------------------------------------------------------


def test_classifies_application_from_solicitud_keywords():
    client = FakeClassificationClient(response="APPLICATION")
    text = "SOLICITUD DE NACIONALIDAD\nNombre del solicitante: Juan Perez"

    result = classify(text, _config(), client)

    assert result.document_type is DocumentType.APPLICATION
    assert result.generation is None
    assert len(client.calls) == 1


def test_classifies_id_document_from_dni_cedula_keywords():
    client = FakeClassificationClient(response="ID_DOCUMENT")
    text = "PASAPORTE\nTipo de ID: Cedula\nNumero de ID: 12345678"

    result = classify(text, _config(), client)

    assert result.document_type is DocumentType.ID_DOCUMENT
    assert result.generation is None


def test_classifies_birth_cert_with_explicit_generation():
    client = FakeClassificationClient(response="BIRTH_CERT\nG2")
    text = "CERTIFICADO DE NACIMIENTO\ngrado_certificado: G_2"

    result = classify(text, _config(), client)

    assert result.document_type is DocumentType.BIRTH_CERT
    assert result.generation == "G2"


def test_classifies_birth_cert_without_explicit_generation():
    client = FakeClassificationClient(response="BIRTH_CERT")
    text = "CERTIFICADO DE NACIMIENTO\nNombre progenitor 1: Maria Lopez"

    result = classify(text, _config(), client)

    assert result.document_type is DocumentType.BIRTH_CERT
    assert result.generation is None


# ---------------------------------------------------------------------------
# Parsing resilience
# ---------------------------------------------------------------------------


def test_fuzzy_match_resolves_noisy_llm_output():
    client = FakeClassificationClient(response="the category is APPLICATION.")
    result = classify("some real document content here", _config(), client)

    assert result.document_type is DocumentType.APPLICATION


def test_unparseable_response_falls_back_to_other():
    client = FakeClassificationClient(response="I cannot determine this document's type.")
    result = classify("some real document content here", _config(), client)

    assert result.document_type is DocumentType.OTHER
    assert result.raw_response == "I cannot determine this document's type."


def test_generation_never_set_for_non_birth_cert_types():
    # Even if the model hallucinates a second line, it's only consulted
    # when the first line resolves to BIRTH_CERT.
    client = FakeClassificationClient(response="APPLICATION\nG1")
    result = classify("some real document content here", _config(), client)

    assert result.document_type is DocumentType.APPLICATION
    assert result.generation is None


def test_unresolvable_generation_line_leaves_generation_none():
    client = FakeClassificationClient(response="BIRTH_CERT\nunclear")
    result = classify("CERTIFICADO DE NACIMIENTO", _config(), client)

    assert result.document_type is DocumentType.BIRTH_CERT
    assert result.generation is None


# ---------------------------------------------------------------------------
# EMPTY short-circuit
# ---------------------------------------------------------------------------


def test_empty_extracted_text_short_circuits_to_empty_without_calling_llm():
    client = FakeClassificationClient(response="OTHER")
    text = "--- Page 1 ---\n"

    result = classify(text, _config(), client)

    assert result.document_type is DocumentType.EMPTY
    assert result.generation is None
    assert result.raw_response == ""
    assert client.calls == []


def test_near_blank_text_below_threshold_is_empty():
    client = FakeClassificationClient(response="OTHER")
    result = classify("--- Page 1 ---\n. ", _config(), client)

    assert result.document_type is DocumentType.EMPTY
    assert client.calls == []


def test_content_above_threshold_calls_llm():
    client = FakeClassificationClient(response="OTHER")
    text = "--- Page 1 ---\nThis has clearly more than ten real characters of content."

    result = classify(text, _config(), client)

    assert result.document_type is DocumentType.OTHER
    assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# Config / result contract
# ---------------------------------------------------------------------------


def test_confidence_defaults_to_placeholder_one():
    client = FakeClassificationClient(response="APPLICATION")
    result = classify("some real document content here", _config(), client)

    assert result.confidence == 1.0


def test_classification_config_frozen_and_defaults():
    config = _config()

    assert config.max_tokens == 32
    assert config.temperature == 0.0
    assert config.max_text_chars == 4000
    try:
        config.max_tokens = 999  # type: ignore[misc]
        assert False, "expected frozen model to reject attribute assignment"
    except Exception:
        pass


def test_max_text_chars_truncates_prompt_input():
    client = FakeClassificationClient(response="OTHER")
    long_text = "SOLICITUD " * 1000  # far more than the default max_text_chars

    classify(long_text, _config(max_text_chars=50), client)

    sent_prompt = client.calls[0]["prompt"]
    # The prompt template itself is long; just confirm the injected text
    # region was cut down, not the full 10,000-character string.
    assert long_text[:50] in sent_prompt
    assert long_text not in sent_prompt
