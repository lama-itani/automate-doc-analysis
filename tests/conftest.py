"""Shared test fixtures for the M-2 OCR extraction tests.

PDF fixtures are built at test time with ``pymupdf`` — no checked-in binaries.
:class:`FakeVlmClient` is a duck-typed test double for
:class:`ps06.ocr.http_client.ReauthHTTPClient`; only ``.chat_completion`` is
used by extraction code, so that's all it implements.
"""

from __future__ import annotations

import pymupdf
import pytest

from ps06.ocr.extraction import ORIENTATION_PROMPT

TEXT_LAYER_CONTENT = (
    "This page has a genuine embedded text layer with far more than sixty "
    "characters so the fast path is exercised instead of the VLM path."
)


@pytest.fixture
def pdf_with_text_layer(tmp_path):
    """Single-page PDF with a real text layer (>= 60 chars) — exercises the fast path."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), TEXT_LAYER_CONTENT)
    path = tmp_path / "text_layer.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def pdf_without_text_layer(tmp_path):
    """Single-page PDF with no text layer — forces the VLM path."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(10, 10, 100, 100))
    path = tmp_path / "no_text_layer.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def pdf_with_acroform(tmp_path):
    """Single-page PDF with a real AcroForm text widget, no text layer."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(10, 10, 100, 100))
    widget = pymupdf.Widget()
    widget.field_name = "applicant_name"
    widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
    widget.rect = pymupdf.Rect(72, 150, 300, 180)
    widget.field_value = "Jane Q. Applicant"
    page.add_widget(widget)
    path = tmp_path / "acroform.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def empty_pdf(tmp_path):
    """A .pdf-suffixed file whose path exists on disk (for the size/extension
    checks) but whose page count is mocked to 0 — pymupdf itself refuses to
    save a zero-page document, so a real on-disk zero-page fixture isn't
    constructible; the caller must monkeypatch pymupdf.open to return a
    Document double with page_count == 0."""
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    return path


class FakeVlmClient:
    """Test double for ReauthHTTPClient.chat_completion.

    Distinguishes the orientation-classification call from the OCR call by
    matching the prompt text against ORIENTATION_PROMPT. Scripts a fixed
    orientation label and OCR text (or a queue of per-call responses), and
    records every call for assertions.
    """

    def __init__(
        self,
        orientation_label: str = "NONE",
        ocr_text: str = "extracted text",
        orientation_responses: list[str] | None = None,
    ) -> None:
        self.orientation_label = orientation_label
        self.ocr_text = ocr_text
        self._orientation_responses = list(orientation_responses) if orientation_responses else None
        self.calls: list[dict] = []

    def chat_completion(self, *, messages, max_tokens, temperature=0.0, extra_body=None):
        content = messages[0]["content"]
        prompt = content[0]["text"]
        image_url = content[1]["image_url"]["url"]
        call = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": extra_body,
            "image_url": image_url,
        }
        self.calls.append(call)

        if prompt == ORIENTATION_PROMPT:
            if self._orientation_responses:
                return self._orientation_responses.pop(0)
            return self.orientation_label
        return self.ocr_text

    @property
    def orientation_calls(self) -> list[dict]:
        return [c for c in self.calls if c["prompt"] == ORIENTATION_PROMPT]

    @property
    def ocr_calls(self) -> list[dict]:
        return [c for c in self.calls if c["prompt"] != ORIENTATION_PROMPT]


@pytest.fixture
def fake_vlm_client():
    return FakeVlmClient()
