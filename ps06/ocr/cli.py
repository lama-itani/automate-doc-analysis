"""Small CLI to exercise the PS-06 per-document OCR job by hand (M-2).

Not part of the production surface — a convenience for local inspection,
mirroring ``ps06/status/cli.py``'s style. ``CDSWAuthProvider`` (see
``ps06.ocr.auth``) is an honest stub that always raises ``NotImplementedError``
until live Cloudera/Workbench credentials land (Build Handoff open item #4), so
today the only way to actually exercise a ``run`` is ``--auth fake``.

Usage (no reinstall needed)::

    python -m ps06.status.cli --db ps06.db add-doc --case C1 --doc a

    python -m ps06.ocr.cli --db ps06.db run \\
        --case C1 --doc a --file /path/to/doc.pdf \\
        --endpoint-url http://fake --model-name test-model \\
        --auth fake --fake-token tok1 --fake-token tok2

    python -m ps06.ocr.cli --db ps06.db show --case C1 --doc a --json
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Any

import openai

from ps06.classification.classifier import ClassificationConfig
from ps06.ocr.auth import AuthProvider, CDSWAuthProvider, FakeAuthProvider
from ps06.ocr.envelope import OcrResult, get_extraction
from ps06.ocr.extraction import DEFAULT_PDF_DPI, ExtractionError, OcrJobConfig
from ps06.ocr.job import run as run_job
from ps06.status import db
from ps06.status.store import DocumentAlreadyExists, StatusStore, StatusStoreError

DEFAULT_DB = "ps06_status.db"


def _print_result(result: OcrResult) -> None:
    extraction = result.extraction
    print(
        f"{result.case_id}/{result.document_id}: "
        f"page_count={extraction.page_count} "
        f"pages_via_text_layer={extraction.pages_via_text_layer} "
        f"pages_via_vlm={extraction.pages_via_vlm} "
        f"processing_seconds={result.processing_seconds:.3f} "
        f"model_name={result.model_name}"
    )
    if result.classification is not None:
        print(
            f"  document_type={result.classification.document_type.value} "
            f"generation={result.classification.generation}"
        )


def _build_auth(args: argparse.Namespace) -> AuthProvider:
    if args.auth == "fake":
        if not args.fake_token:
            raise SystemExit("error: --auth fake requires at least one --fake-token")
        return FakeAuthProvider(tokens=args.fake_token)
    return CDSWAuthProvider()


def _build_classification_config(args: argparse.Namespace) -> ClassificationConfig:
    # prompt_template is intentionally not exposed as a CLI flag — it's a
    # multi-line Spanish-aware prompt, impractical as a single-line arg.
    # Stays code-default-only.
    return ClassificationConfig(
        max_tokens=args.classification_max_tokens,
        temperature=args.classification_temperature,
        max_text_chars=args.classification_max_text_chars,
        min_chars_for_classification=args.classification_min_chars,
    )


def _build_config(args: argparse.Namespace) -> OcrJobConfig:
    return OcrJobConfig(
        endpoint_url=args.endpoint_url,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        pdf_dpi=args.pdf_dpi,
        temperature=args.temperature,
        use_text_layer_fast_path=args.use_text_layer_fast_path,
        min_text_layer_chars=args.min_text_layer_chars,
        enable_orientation_correction=args.enable_orientation_correction,
        classification=_build_classification_config(args),
    )


def _cmd_run(
    store: StatusStore,
    args: argparse.Namespace,
    client_factory: Callable[..., Any],
) -> int:
    if args.auto_register:
        try:
            store.register_document(args.case, args.doc)
        except DocumentAlreadyExists:
            pass

    auth = _build_auth(args)
    config = _build_config(args)
    result = run_job(
        args.case, args.doc, args.file, config, store, auth, client_factory=client_factory
    )

    _print_result(result)
    if args.json:
        print(result.model_dump_json(indent=2))
    return 0


def _cmd_show(
    store: StatusStore,
    args: argparse.Namespace,
    client_factory: Callable[..., Any],
) -> int:
    result = get_extraction(store, args.case, args.doc)
    if result is None:
        print(
            f"error: no extraction found for case={args.case!r} document={args.doc!r}",
            file=sys.stderr,
        )
        return 1

    _print_result(result)
    if args.json:
        print(result.model_dump_json(indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ps06-ocr", description="Run/inspect the PS-06 per-document OCR job."
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB, help=f"SQLite path (default: {DEFAULT_DB})"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run OCR extraction for one document")
    p_run.add_argument("--case", required=True)
    p_run.add_argument("--doc", required=True)
    p_run.add_argument("--file", required=True, help="path to the source document")
    p_run.add_argument("--endpoint-url", required=True)
    p_run.add_argument("--model-name", required=True)
    p_run.add_argument("--max-tokens", type=int, default=2500)
    p_run.add_argument("--pdf-dpi", type=int, default=DEFAULT_PDF_DPI)
    p_run.add_argument("--temperature", type=float, default=0.0)
    p_run.add_argument(
        "--no-text-layer-fast-path",
        dest="use_text_layer_fast_path",
        action="store_false",
        default=True,
        help="disable the PDF text-layer fast path (always use the VLM)",
    )
    p_run.add_argument("--min-text-layer-chars", type=int, default=60)
    p_run.add_argument(
        "--enable-orientation-correction",
        action="store_true",
        default=False,
        help="detect and correct page rotation via the VLM before OCR "
        "(off by default: unreliable on sparse/portrait content, see Tier-1 findings)",
    )
    p_run.add_argument("--classification-max-tokens", type=int, default=32)
    p_run.add_argument("--classification-temperature", type=float, default=0.0)
    p_run.add_argument("--classification-max-text-chars", type=int, default=4000)
    p_run.add_argument("--classification-min-chars", type=int, default=10)
    p_run.add_argument(
        "--auth", choices=["cdsw", "fake"], default="cdsw",
        help="token provider: 'cdsw' (real, not yet implemented) or 'fake' "
        "(--fake-token, for local hand-testing)",
    )
    p_run.add_argument(
        "--fake-token", action="append", default=None,
        help="bearer token to hand FakeAuthProvider (repeatable; --auth fake only)",
    )
    p_run.add_argument(
        "--auto-register", action="store_true",
        help="register the document at RECEIVED first if not already present",
    )
    p_run.add_argument("--json", action="store_true", help="also print the full OcrResult as JSON")
    p_run.set_defaults(func=_cmd_run)

    p_show = sub.add_parser("show", help="show a persisted OcrResult for one document")
    p_show.add_argument("--case", required=True)
    p_show.add_argument("--doc", required=True)
    p_show.add_argument("--json", action="store_true", help="print the full OcrResult as JSON")
    p_show.set_defaults(func=_cmd_show)

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[..., Any] = openai.OpenAI,
) -> int:
    args = _build_parser().parse_args(argv)
    conn = db.connect(args.db)
    try:
        store = StatusStore(conn)
        return args.func(store, args, client_factory)
    except (StatusStoreError, ExtractionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
