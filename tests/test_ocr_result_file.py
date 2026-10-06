"""Tests for ps06.ocr.result_file (job -> API result handoff over NFS)."""

from __future__ import annotations

import pytest

from ps06.ocr.envelope import OcrResult, get_extraction
from ps06.ocr.extraction import ExtractionResult
from ps06.ocr.result_file import (
    ResultFile,
    ResultFileError,
    ingest,
    read_result_file,
    result_path,
    write_failure,
    write_success,
)
from ps06.status import db
from ps06.status.states import DocumentStage
from ps06.status.store import DocumentNotFound, InvalidTransition, StatusStore


def _result(case_id: str = "C1", document_id: str = "a") -> OcrResult:
    return OcrResult.build(
        case_id=case_id,
        document_id=document_id,
        file_path="/x/a.pdf",
        extraction=ExtractionResult(
            file_type="pdf",
            extracted_text="--- Page 1 ---\nHola",
            page_count=1,
            pages_via_text_layer=1,
            pages_via_vlm=0,
            orientation_corrections={},
            acroform_fields={},
            xfa_fields={},
        ),
        processing_seconds=1.0,
        model_name="m",
    )


@pytest.fixture()
def store() -> StatusStore:
    s = StatusStore(db.connect())
    s.register_document("C1", "a")
    return s


def test_result_path_layout(tmp_path):
    assert result_path(tmp_path, "C1", "a") == tmp_path / "C1" / "a.json"


@pytest.mark.parametrize("bad", ["", ".", "..", "x/y", "x\\y"])
def test_result_path_rejects_path_like_ids(tmp_path, bad):
    with pytest.raises(ValueError):
        result_path(tmp_path, bad, "a")
    with pytest.raises(ValueError):
        result_path(tmp_path, "C1", bad)


def test_success_roundtrip_and_no_temp_left(tmp_path):
    path = result_path(tmp_path, "C1", "a")
    write_success(path, _result())
    rf = read_result_file(path)
    assert rf.ok and rf.result == _result().model_copy(
        update={"extracted_at": rf.result.extracted_at}
    )
    assert [p.name for p in path.parent.iterdir()] == ["a.json"]


def test_failure_roundtrip(tmp_path):
    path = result_path(tmp_path, "C1", "a")
    write_failure(path, "C1", "a", "FileNotFoundError: x")
    rf = read_result_file(path)
    assert not rf.ok and rf.error_detail == "FileNotFoundError: x"


def test_failure_requires_error_detail(tmp_path):
    with pytest.raises(ValueError):
        write_failure(tmp_path / "f.json", "C1", "a", "  ")
    assert not (tmp_path / "f.json").exists()


def test_missing_file_returns_none(tmp_path):
    assert read_result_file(tmp_path / "nope.json") is None


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "a.json"
    path.write_text("{not json")
    with pytest.raises(ResultFileError):
        read_result_file(path)


def test_shape_rules():
    with pytest.raises(ValueError):
        ResultFile(ok=True, case_id="C1", document_id="a")
    with pytest.raises(ValueError):
        ResultFile(ok=True, case_id="C1", document_id="b", result=_result())
    with pytest.raises(ValueError):
        ResultFile(ok=False, case_id="C1", document_id="a", result=_result(),
                   error_detail="e")


def test_ingest_success(store):
    rf = ResultFile(ok=True, case_id="C1", document_id="a", result=_result())
    assert ingest(store, "C1", "a", rf) is DocumentStage.OCR_DONE
    assert store.get_or_raise("C1", "a").stage is DocumentStage.OCR_DONE
    assert get_extraction(store, "C1", "a") == rf.result


def test_ingest_failure_sets_error_detail(store):
    rf = ResultFile(ok=False, case_id="C1", document_id="a", error_detail="boom")
    assert ingest(store, "C1", "a", rf) is DocumentStage.FAILED
    status = store.get_or_raise("C1", "a")
    assert status.stage is DocumentStage.FAILED
    assert status.error_detail == "boom"


def test_ingest_rejects_mismatched_ids(store):
    rf = ResultFile(ok=False, case_id="C1", document_id="zz", error_detail="e")
    with pytest.raises(ResultFileError):
        ingest(store, "C1", "a", rf)
    assert store.get_or_raise("C1", "a").stage is DocumentStage.RECEIVED


def test_ingest_unknown_document(store):
    rf = ResultFile(ok=False, case_id="C1", document_id="b", error_detail="e")
    with pytest.raises(DocumentNotFound):
        ingest(store, "C1", "b", rf)


def test_ingest_twice_is_rejected(store):
    rf = ResultFile(ok=False, case_id="C1", document_id="a", error_detail="e")
    ingest(store, "C1", "a", rf)
    with pytest.raises(InvalidTransition):
        ingest(store, "C1", "a", rf)