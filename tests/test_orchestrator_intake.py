"""Unit tests for ps06/orchestrator/intake.py (pure folder scan, no store)."""

from __future__ import annotations

import pytest

from ps06.orchestrator.intake import (
    DuplicateDocumentIdError,
    EmptyFolderError,
    scan_folder,
)


def test_scan_folder_returns_sorted_supported_files(tmp_path):
    (tmp_path / "b.pdf").write_bytes(b"")
    (tmp_path / "a.png").write_bytes(b"")
    documents = scan_folder(tmp_path)
    assert documents == {
        "a": str(tmp_path / "a.png"),
        "b": str(tmp_path / "b.pdf"),
    }


def test_scan_folder_skips_unsupported_extensions(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignored")
    documents = scan_folder(tmp_path)
    assert documents == {"a": str(tmp_path / "a.pdf")}


def test_scan_folder_skips_dotfiles(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / ".DS_Store").write_bytes(b"")
    documents = scan_folder(tmp_path)
    assert documents == {"a": str(tmp_path / "a.pdf")}


def test_scan_folder_skips_subdirectories(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"")
    sub = tmp_path / "sub.pdf"
    sub.mkdir()
    (sub / "b.pdf").write_bytes(b"")
    documents = scan_folder(tmp_path)
    assert documents == {"a": str(tmp_path / "a.pdf")}


def test_scan_folder_raises_on_empty_folder(tmp_path):
    with pytest.raises(EmptyFolderError):
        scan_folder(tmp_path)


def test_scan_folder_raises_on_only_unsupported_files(tmp_path):
    (tmp_path / "notes.txt").write_text("ignored")
    with pytest.raises(EmptyFolderError):
        scan_folder(tmp_path)


def test_scan_folder_raises_on_duplicate_stem(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "a.png").write_bytes(b"")
    with pytest.raises(DuplicateDocumentIdError):
        scan_folder(tmp_path)
