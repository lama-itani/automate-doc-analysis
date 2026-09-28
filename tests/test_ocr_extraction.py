"""Tests for ps06.ocr.extraction.

Exercises the ported OCR/extraction logic entirely against pymupdf-built PDF
fixtures and a fake VLM client — no network, no live Cloudera/Workbench
credentials.
"""

from __future__ import annotations

import base64
import io

import pymupdf
import pytest
from PIL import Image, ImageOps

from ps06.ocr.extraction import (
    EmptyDocumentError,
    EmptyExtractionError,
    FileTooLargeError,
    OcrJobConfig,
    UnsupportedFileTypeError,
    _parse_xfa_xml,
    extract,
)
from tests.conftest import FakeVlmClient


def _config(**overrides) -> OcrJobConfig:
    defaults = dict(endpoint_url="https://example.test/v1", model_name="test-model")
    defaults.update(overrides)
    return OcrJobConfig(**defaults)


def _b64_to_image(b64_data_url: str) -> Image.Image:
    b64 = b64_data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ---------------------------------------------------------------------------
# Fast path vs VLM path
# ---------------------------------------------------------------------------


def test_fast_path_selected_when_text_layer_sufficient(pdf_with_text_layer, fake_vlm_client):
    result = extract(str(pdf_with_text_layer), _config(min_text_layer_chars=60), fake_vlm_client)

    assert result.pages_via_text_layer == 1
    assert result.pages_via_vlm == 0
    assert fake_vlm_client.calls == []
    assert "[text layer]" in result.extracted_text


def test_fast_path_disabled_forces_vlm_even_with_text_layer(pdf_with_text_layer, fake_vlm_client):
    result = extract(
        str(pdf_with_text_layer),
        _config(use_text_layer_fast_path=False, min_text_layer_chars=60),
        fake_vlm_client,
    )

    assert result.pages_via_vlm == 1
    assert result.pages_via_text_layer == 0
    assert len(fake_vlm_client.ocr_calls) == 1


def test_vlm_path_with_orientation_correction(pdf_without_text_layer):
    client = FakeVlmClient(orientation_label="ROTATE_180", ocr_text="rotated text")

    result = extract(str(pdf_without_text_layer), _config(), client)

    assert result.pages_via_vlm == 1
    assert result.orientation_corrections == {1: "ROTATE_180"}
    assert "rotated text" in result.extracted_text
    assert "[orientation corrected: ROTATE_180]" in result.extracted_text

    # The OCR call must have received the *corrected* image bytes, not a raw
    # render — there is only ever one OCR call per page (correction happens
    # before it, not as a second call).
    doc = pymupdf.open(str(pdf_without_text_layer))
    mat = pymupdf.Matrix(300 / 72.0, 300 / 72.0)
    raw_png = doc[0].get_pixmap(matrix=mat, alpha=False).tobytes("png")
    doc.close()
    raw_b64_url = f"data:image/png;base64,{base64.b64encode(raw_png).decode()}"

    assert len(client.ocr_calls) == 1
    assert client.ocr_calls[0]["image_url"] != raw_b64_url


def test_orientation_none_not_recorded(pdf_without_text_layer):
    client = FakeVlmClient(orientation_label="NONE", ocr_text="upright text")

    result = extract(str(pdf_without_text_layer), _config(), client)

    assert result.orientation_corrections == {}
    assert "[orientation corrected" not in result.extracted_text


@pytest.mark.parametrize(
    ("label", "transform"),
    [
        ("FLIP_HORIZONTAL", ImageOps.mirror),
        ("FLIP_VERTICAL", ImageOps.flip),
        ("ROTATE_180", lambda img: img.rotate(180)),
    ],
)
def test_each_orientation_label_applies_expected_transform(pdf_without_text_layer, label, transform):
    client = FakeVlmClient(orientation_label=label, ocr_text="text")

    result = extract(str(pdf_without_text_layer), _config(), client)

    assert result.orientation_corrections == {1: label}

    doc = pymupdf.open(str(pdf_without_text_layer))
    mat = pymupdf.Matrix(300 / 72.0, 300 / 72.0)
    raw_png = doc[0].get_pixmap(matrix=mat, alpha=False).tobytes("png")
    doc.close()
    expected = transform(Image.open(io.BytesIO(raw_png)))

    sent = _b64_to_image(client.ocr_calls[0]["image_url"])
    assert sent.tobytes() == expected.tobytes()


def test_fuzzy_orientation_response_resolves(pdf_without_text_layer):
    client = FakeVlmClient(orientation_label="the answer is ROTATE_180.", ocr_text="text")

    result = extract(str(pdf_without_text_layer), _config(), client)

    assert result.orientation_corrections == {1: "ROTATE_180"}


def test_unrecognized_orientation_response_defaults_to_none(pdf_without_text_layer):
    client = FakeVlmClient(orientation_label="gibberish", ocr_text="text")

    result = extract(str(pdf_without_text_layer), _config(), client)

    assert result.orientation_corrections == {}


def test_multipage_pdf_mixes_fast_path_and_vlm_path(tmp_path):
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "This page has a real text layer well over sixty characters long.")
    doc.new_page()  # page 2: blank, no text layer
    path = tmp_path / "mixed.pdf"
    doc.save(str(path))
    doc.close()

    client = FakeVlmClient(orientation_label="NONE", ocr_text="page two ocr text")

    result = extract(str(path), _config(min_text_layer_chars=60), client)

    assert result.page_count == 2
    assert result.pages_via_text_layer == 1
    assert result.pages_via_vlm == 1
    text_pos = result.extracted_text.index("--- Page 1 [text layer]")
    vlm_pos = result.extracted_text.index("--- Page 2")
    assert text_pos < vlm_pos


# ---------------------------------------------------------------------------
# Form field extraction
# ---------------------------------------------------------------------------


def test_acroform_fields_extracted(pdf_with_acroform):
    client = FakeVlmClient(orientation_label="NONE", ocr_text="form scan text")

    result = extract(str(pdf_with_acroform), _config(), client)

    assert result.acroform_fields == {"applicant_name": "Jane Q. Applicant"}
    form_pos = result.extracted_text.index("Fillable Form Fields")
    ocr_pos = result.extracted_text.index("form scan text")
    assert form_pos < ocr_pos


def test_xfa_fields_parsed_directly():
    raw_xml = (
        b'<xdp:datasets xmlns:xdp="http://ns.adobe.com/xdp/">'
        b"<data><applicant_name>Jane</applicant_name></data>"
        b"</xdp:datasets>"
    )
    fields = _parse_xfa_xml(raw_xml)
    assert fields == {"applicant_name": "Jane"}


def test_xfa_fields_extracted_via_mocked_document(pdf_without_text_layer, monkeypatch):
    class _FakeDoc:
        def pdf_catalog(self):
            return 1

        def xref_object(self, xref, compressed=False):
            if xref == 1:
                return "<< /AcroForm 2 0 R >>"
            if xref == 2:
                return "<< /XFA 3 0 R >>"
            raise AssertionError(xref)

        def xref_stream(self, xref):
            assert xref == 3
            return (
                b'<xdp:datasets xmlns:xdp="http://ns.adobe.com/xdp/">'
                b"<data><applicant_name>Jane</applicant_name></data>"
                b"</xdp:datasets>"
            )

        def close(self):
            pass

    import ps06.ocr.extraction as extraction_module

    monkeypatch.setattr(extraction_module.pymupdf, "open", lambda *_a, **_kw: _FakeDoc())

    fields = extraction_module._extract_xfa_fields("irrelevant.pdf")
    assert fields == {"applicant_name": "Jane"}


def test_no_form_fields_present(pdf_without_text_layer):
    client = FakeVlmClient(orientation_label="NONE", ocr_text="plain scan text")

    result = extract(str(pdf_without_text_layer), _config(), client)

    assert result.acroform_fields == {}
    assert result.xfa_fields == {}
    assert "Fillable Form Fields" not in result.extracted_text


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def test_image_file_extraction(tmp_path):
    img = Image.new("RGB", (100, 100), color="white")
    path = tmp_path / "scan.png"
    img.save(path, format="PNG")

    client = FakeVlmClient(orientation_label="NONE", ocr_text="scanned image text")

    result = extract(str(path), _config(), client)

    assert result.file_type == "image"
    assert result.page_count == 1
    assert result.pages_via_text_layer == 0
    assert result.pages_via_vlm == 1
    assert "scanned image text" in result.extracted_text


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "doc.docx"
    path.write_bytes(b"irrelevant")

    with pytest.raises(UnsupportedFileTypeError):
        extract(str(path), _config(), FakeVlmClient())


def test_oversized_file_raises_before_any_vlm_call(pdf_with_text_layer, monkeypatch):
    client = FakeVlmClient()
    monkeypatch.setattr("ps06.ocr.extraction.os.path.getsize", lambda _path: 51 * 1024 * 1024)

    with pytest.raises(FileTooLargeError):
        extract(str(pdf_with_text_layer), _config(), client)

    assert client.calls == []


def test_empty_model_response_raises(tmp_path):
    # The PDF path always prepends a "--- Page N ---" header, so an empty OCR
    # response never yields a truly empty extracted_text there (matches the
    # port source's behavior exactly). The image path has no such header, so
    # it's the one that actually exercises this raise.
    img = Image.new("RGB", (10, 10))
    path = tmp_path / "blank.png"
    img.save(path, format="PNG")
    client = FakeVlmClient(orientation_label="NONE", ocr_text="")

    with pytest.raises(EmptyExtractionError):
        extract(str(path), _config(), client)


def test_zero_page_pdf_raises(empty_pdf, monkeypatch):
    # pymupdf refuses to save a real zero-page document to disk, so the
    # zero-page condition is exercised via a mocked Document double instead.
    class _ZeroPageDoc:
        page_count = 0

        def close(self):
            pass

    import ps06.ocr.extraction as extraction_module

    monkeypatch.setattr(extraction_module.pymupdf, "open", lambda *_a, **_kw: _ZeroPageDoc())

    with pytest.raises(EmptyDocumentError):
        extract(str(empty_pdf), _config(), FakeVlmClient())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_ocr_job_config_defaults_and_frozen():
    config = _config()

    assert config.pdf_dpi == 300
    assert config.min_text_layer_chars == 60
    assert config.use_text_layer_fast_path is True
    assert config.temperature == 0.0

    with pytest.raises(Exception):
        config.model_name = "changed"
