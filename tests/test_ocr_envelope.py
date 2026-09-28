"""Tests for ps06.ocr.envelope.OcrResult (M-2, step 6)."""

from __future__ import annotations

import pytest

from ps06.ocr.envelope import OcrResult
from ps06.ocr.extraction import ExtractionResult


def _extraction() -> ExtractionResult:
    return ExtractionResult(
        file_type="pdf",
        extracted_text="--- Page 1 ---\nHello",
        page_count=1,
        pages_via_text_layer=1,
        pages_via_vlm=0,
        orientation_corrections={2: "ROTATE_90"},
        acroform_fields={"name": "Jane Doe"},
        xfa_fields={},
    )


def test_build_derives_basename_and_timestamp():
    result = OcrResult.build(
        case_id="case-1",
        document_id="doc-1",
        file_path="/some/local/dir/passport.pdf",
        extraction=_extraction(),
        processing_seconds=1.5,
        model_name="qwen-vl",
    )

    assert result.case_id == "case-1"
    assert result.document_id == "doc-1"
    assert result.source_file == "passport.pdf"
    assert result.processing_seconds == 1.5
    assert result.model_name == "qwen-vl"
    assert result.extracted_at.endswith("Z")


def test_json_round_trip_preserves_nested_extraction():
    result = OcrResult.build(
        case_id="case-1",
        document_id="doc-1",
        file_path="passport.pdf",
        extraction=_extraction(),
        processing_seconds=1.5,
        model_name="qwen-vl",
    )

    restored = OcrResult.model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored.extraction.orientation_corrections == {2: "ROTATE_90"}
    assert restored.extraction.acroform_fields == {"name": "Jane Doe"}
    assert restored.extraction.xfa_fields == {}


def test_frozen():
    result = OcrResult.build(
        case_id="case-1",
        document_id="doc-1",
        file_path="passport.pdf",
        extraction=_extraction(),
        processing_seconds=1.5,
        model_name="qwen-vl",
    )

    with pytest.raises(Exception):
        result.model_name = "changed"
