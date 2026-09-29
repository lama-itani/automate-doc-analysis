"""Small CLI to drive the PS-06 app-layer orchestrator by hand (M-3).

Not part of the production surface — a convenience for local inspection,
mirroring ``ps06/ocr/cli.py`` and ``ps06/status/cli.py``. It drives documents
through the orchestrator using the in-process :class:`LocalThreadJobLauncher`
(the real Workbench API v2 launcher is a stub — Build Handoff open item #4).
``CDSWAuthProvider`` is likewise an honest stub, so today ``--auth fake`` is the
only way an invocation actually processes a document; ``--auth cdsw`` will drive
every document to ``FAILED`` (with a ``NotImplementedError`` error_detail),
which is the honest behaviour, not a crash.

Usage (no reinstall needed)::

    python -m ps06.status.cli --db ps06.db add-doc --case C1 --doc d1

    python -m ps06.orchestrator.cli --db ps06.db run-case \\
        --case C1 --doc d1=/path/a.pdf --doc d2=/path/b.pdf \\
        --endpoint-url http://fake --model-name test-model \\
        --auth fake --fake-token tok --auto-register

    python -m ps06.orchestrator.cli --db ps06.db status --case C1

    python -m ps06.orchestrator.cli --db ps06.db retry-failed \\
        --case C1 --doc d1=/path/a.pdf --max-attempts 5 \\
        --endpoint-url http://fake --model-name test-model --auth fake --fake-token tok
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from ps06.ocr.auth import AuthProvider, CDSWAuthProvider, FakeAuthProvider
from ps06.ocr.extraction import DEFAULT_PDF_DPI, OcrJobConfig
from ps06.orchestrator.launcher import LocalThreadJobLauncher
from ps06.orchestrator.orchestrator import (
    CaseSummary,
    OrchestratorConfig,
    process_case,
)
from ps06.status import db
from ps06.status.states import DocumentStage
from ps06.status.store import DocumentAlreadyExists, StatusStore, StatusStoreError

DEFAULT_DB = "ps06_status.db"


def _parse_documents(pairs: Sequence[str]) -> dict[str, str]:
    """Parse repeated ``--doc ID=PATH`` values into a ``document_id -> path`` map."""
    documents: dict[str, str] = {}
    for pair in pairs:
        doc_id, sep, path = pair.partition("=")
        if not sep or not doc_id or not path:
            raise SystemExit(f"error: --doc must be ID=PATH, got {pair!r}")
        if doc_id in documents:
            raise SystemExit(f"error: duplicate --doc id {doc_id!r}")
        documents[doc_id] = path
    return documents


def _build_auth_factory(args: argparse.Namespace) -> Callable[[], AuthProvider]:
    """A factory the launcher calls once per invocation (fresh provider each time,
    mirroring per-invocation token isolation)."""
    if args.auth == "fake":
        if not args.fake_token:
            raise SystemExit("error: --auth fake requires at least one --fake-token")
        tokens = list(args.fake_token)
        return lambda: FakeAuthProvider(tokens=list(tokens))
    return lambda: CDSWAuthProvider()


def _build_ocr_config(args: argparse.Namespace) -> OcrJobConfig:
    return OcrJobConfig(
        endpoint_url=args.endpoint_url,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        pdf_dpi=args.pdf_dpi,
        temperature=args.temperature,
        use_text_layer_fast_path=args.use_text_layer_fast_path,
        min_text_layer_chars=args.min_text_layer_chars,
        enable_orientation_correction=args.enable_orientation_correction,
    )


def _build_orchestrator_config(args: argparse.Namespace) -> OrchestratorConfig:
    return OrchestratorConfig(
        max_concurrency=args.max_concurrency,
        max_attempts=args.max_attempts,
        poll_interval_seconds=args.poll_interval,
    )


def _print_summary(summary: CaseSummary) -> None:
    status = summary.case_status.value if summary.case_status else "(no documents)"
    print(
        f"case {summary.case_id!r}: {status}  "
        f"[{summary.succeeded}/{summary.total} OCR_DONE, {summary.failed} FAILED]"
    )
    for o in summary.documents:
        err = f"  error={o.error_detail!r}" if o.error_detail else ""
        print(f"  {o.document_id}: {o.stage.value}  attempt={o.attempt_count}{err}")


def _drive(store: StatusStore, args: argparse.Namespace, documents: dict[str, str]) -> int:
    """Run one orchestrator pass over ``documents`` and report. Returns a nonzero
    exit code if any document ended ``FAILED`` (surface failures, never silent)."""
    launcher = LocalThreadJobLauncher(
        args.db,
        _build_ocr_config(args),
        _build_auth_factory(args),
        max_workers=args.max_concurrency,
    )
    with launcher:
        summary = process_case(
            args.case, documents, launcher, store, _build_orchestrator_config(args)
        )
    _print_summary(summary)
    if args.json:
        print(summary.model_dump_json(indent=2))
    return 0 if summary.failed == 0 else 1


def _cmd_run_case(store: StatusStore, args: argparse.Namespace) -> int:
    documents = _parse_documents(args.doc)
    if args.auto_register:
        for doc_id in documents:
            try:
                store.register_document(args.case, doc_id)
            except DocumentAlreadyExists:
                pass
    return _drive(store, args, documents)


def _cmd_retry_failed(store: StatusStore, args: argparse.Namespace) -> int:
    documents = _parse_documents(args.doc)
    failed = {
        doc_id: path
        for doc_id, path in documents.items()
        if (s := store.get(args.case, doc_id)) is not None
        and s.stage is DocumentStage.FAILED
    }
    if not failed:
        print(f"case {args.case!r}: no FAILED documents among the supplied set")
        return 0
    # Raise --max-attempts above the stored attempt_count to re-enable retries.
    return _drive(store, args, failed)


def _cmd_status(store: StatusStore, args: argparse.Namespace) -> int:
    docs = store.list_for_case(args.case)
    if not docs:
        print(f"case {args.case!r}: no documents")
        return 0
    print(f"case {args.case!r}: {store.case_status(args.case).value}")
    for s in docs:
        err = f"  error={s.error_detail!r}" if s.error_detail else ""
        print(
            f"  {s.document_id}: {s.stage.value}  attempt={s.attempt_count}  "
            f"updated={s.last_updated}{err}"
        )
    return 0


def _add_drive_args(p: argparse.ArgumentParser) -> None:
    """Args shared by ``run-case`` and ``retry-failed`` (OCR config + auth +
    orchestrator tunables)."""
    p.add_argument("--case", required=True)
    p.add_argument(
        "--doc",
        action="append",
        required=True,
        metavar="ID=PATH",
        help="document id and source file path (repeatable)",
    )
    p.add_argument("--endpoint-url", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--max-tokens", type=int, default=2500)
    p.add_argument("--pdf-dpi", type=int, default=DEFAULT_PDF_DPI)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--no-text-layer-fast-path",
        dest="use_text_layer_fast_path",
        action="store_false",
        default=True,
        help="disable the PDF text-layer fast path (always use the VLM)",
    )
    p.add_argument("--min-text-layer-chars", type=int, default=60)
    p.add_argument(
        "--enable-orientation-correction",
        action="store_true",
        default=False,
        help="detect/correct page rotation via the VLM before OCR (off by "
        "default: unreliable on sparse/portrait content, see Tier-1 findings)",
    )
    p.add_argument(
        "--auth",
        choices=["cdsw", "fake"],
        default="cdsw",
        help="token provider: 'cdsw' (real, not yet implemented -> all FAILED) "
        "or 'fake' (--fake-token, for local hand-testing)",
    )
    p.add_argument(
        "--fake-token",
        action="append",
        default=None,
        help="bearer token to hand FakeAuthProvider (repeatable; --auth fake only)",
    )
    p.add_argument("--max-concurrency", type=int, default=3)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--poll-interval", type=float, default=1.0)
    p.add_argument("--json", action="store_true", help="also print the CaseSummary as JSON")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ps06-orchestrator",
        description="Drive/inspect the PS-06 app-layer orchestrator.",
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB, help=f"SQLite path (default: {DEFAULT_DB})"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run-case", help="launch/poll/retry all of a case's documents")
    _add_drive_args(p_run)
    p_run.add_argument(
        "--auto-register",
        action="store_true",
        help="register each document at RECEIVED first if not already present",
    )
    p_run.set_defaults(func=_cmd_run_case)

    p_retry = sub.add_parser(
        "retry-failed",
        help="re-drive only the currently-FAILED documents among the supplied set",
    )
    _add_drive_args(p_retry)
    p_retry.set_defaults(func=_cmd_retry_failed)

    p_status = sub.add_parser("status", help="show a case's documents + derived status")
    p_status.add_argument("--case", required=True)
    p_status.set_defaults(func=_cmd_status)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    conn = db.connect(args.db)
    try:
        store = StatusStore(conn)
        return args.func(store, args)
    except StatusStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
