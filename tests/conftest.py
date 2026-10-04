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


class FakeClassificationClient:
    """Test double for ReauthHTTPClient.chat_completion, for classification.

    Classification calls are text-only (no image_url block), unlike
    FakeVlmClient's OCR/orientation calls — distinguished structurally here
    rather than by prompt text, since the classification prompt interpolates
    the full extracted text and isn't a stable string to match on.
    """

    def __init__(self, response: str = "OTHER") -> None:
        self.response = response
        self.calls: list[dict] = []

    def chat_completion(self, *, messages, max_tokens, temperature=0.0, extra_body=None):
        prompt = messages[0]["content"][0]["text"]
        call = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": extra_body,
        }
        self.calls.append(call)
        return self.response


@pytest.fixture
def fake_classification_client():
    return FakeClassificationClient()


class FakeFieldExtractionClient:
    """Test double for ReauthHTTPClient.chat_completion, for field extraction.

    Same shape as FakeClassificationClient — text-only calls, one scripted
    response. Field-extraction calls are distinguished from classification
    calls by the caller choosing which fake to inject (job-level fakes that
    need to serve both call types route by prompt content instead, as
    FakeOpenAIClient does).
    """

    def __init__(self, response: str = "{}") -> None:
        self.response = response
        self.calls: list[dict] = []

    def chat_completion(self, *, messages, max_tokens, temperature=0.0, extra_body=None):
        prompt = messages[0]["content"][0]["text"]
        call = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": extra_body,
        }
        self.calls.append(call)
        return self.response


@pytest.fixture
def fake_field_extraction_client():
    return FakeFieldExtractionClient()


class _FakeMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMsg(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """Returns ``ocr_text`` for an OCR/orientation-style call (image block
    present), ``field_extraction_text`` for a field-extraction call (text-only,
    "structured-data extractor" instruction phrase present), and
    ``classification_text`` for a classification call (text-only, everything
    else) — distinguished structurally, same convention as
    FakeClassificationClient/FakeFieldExtractionClient.
    """

    def __init__(
        self,
        ocr_text: str,
        classification_text: str,
        fail_on_call: int | None,
        field_extraction_text: str = "{}",
    ) -> None:
        self._ocr_text = ocr_text
        self._classification_text = classification_text
        self._field_extraction_text = field_extraction_text
        self._fail_on_call = fail_on_call
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        if self._fail_on_call == self.call_count:
            raise RuntimeError(f"simulated failure on call {self.call_count}")
        messages = kwargs["messages"]
        content_blocks = messages[0]["content"]
        has_image = len(content_blocks) > 1
        if has_image:
            content = self._ocr_text
        elif "structured-data extractor" in content_blocks[0]["text"]:
            content = self._field_extraction_text
        else:
            content = self._classification_text
        return _FakeCompletion(content)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class FakeOpenAIClient:
    """Test double standing in for ``openai.OpenAI`` at the ``client_factory``
    seam (:class:`~ps06.ocr.http_client.ReauthHTTPClient`,
    :func:`ps06.ocr.job.run`, :class:`~ps06.orchestrator.launcher.LocalThreadJobLauncher`).

    Needed because classification (M-2.5) unconditionally calls the LLM once
    extracted text is above the EMPTY threshold — CLI/orchestrator tests that
    previously relied on the text-layer fast path making zero LLM calls (and
    so could safely construct a *real* ``openai.OpenAI`` against a fake URL)
    now need a fake client at this seam too, or the classification call
    attempts a real network request.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        ocr_text: str = "unused",
        classification_text: str = "OTHER",
        fail_on_call: int | None = None,
        field_extraction_text: str = "{}",
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.chat = _FakeChat(
            _FakeCompletions(ocr_text, classification_text, fail_on_call, field_extraction_text)
        )


@pytest.fixture
def fake_openai_client_factory():
    """A zero-arg factory producing a fresh :class:`FakeOpenAIClient`, ready to
    pass as ``client_factory=`` to ``job.run``/``LocalThreadJobLauncher``/the
    CLIs' ``client_factory`` param."""

    def factory(*, base_url: str, api_key: str) -> FakeOpenAIClient:
        return FakeOpenAIClient(base_url=base_url, api_key=api_key)

    return factory
