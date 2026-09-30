"""Folder-scan intake: a local/dev-facing convenience to build the same
``document_id -> path`` map ``ps06.orchestrator.cli._parse_documents`` produces,
without an explicit ``--doc ID=PATH`` per file.

Kept pure (no store/side effects) so it's trivially unit-testable, matching the
project's pattern of pure functions + a thin composing CLI layer.
"""

from __future__ import annotations

from pathlib import Path

from ps06.ocr.extraction import SUPPORTED_EXTENSIONS


class EmptyFolderError(ValueError):
    """Raised when a folder has no top-level files with a supported extension."""


class DuplicateDocumentIdError(ValueError):
    """Raised when two files in a folder resolve to the same document id
    (same stem, different extension) — ambiguous, so this fails loud rather
    than silently overwriting one."""


def scan_folder(folder: str | Path) -> dict[str, str]:
    """Scan ``folder``'s top-level entries (non-recursive) and build a
    ``document_id -> path`` map, using each file's stem as its id.

    Skips dotfiles and subdirectories; only keeps files whose suffix is in
    ``SUPPORTED_EXTENSIONS``. Raises :class:`EmptyFolderError` if nothing
    matches, and :class:`DuplicateDocumentIdError` on a duplicate stem.
    """
    folder = Path(folder)
    documents: dict[str, str] = {}
    for path in sorted(folder.iterdir()):
        if path.name.startswith("."):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        doc_id = path.stem
        if doc_id in documents:
            raise DuplicateDocumentIdError(
                f"duplicate document id {doc_id!r}: {documents[doc_id]!r} and "
                f"{str(path)!r} both resolve to it"
            )
        documents[doc_id] = str(path)

    if not documents:
        raise EmptyFolderError(
            f"no supported documents found in {str(folder)!r} "
            f"(supported extensions: {sorted(SUPPORTED_EXTENSIONS)})"
        )
    return documents
