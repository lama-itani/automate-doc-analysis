"""Small CLI to exercise the PS-06 status table by hand (M-1).

Not part of the production surface — a convenience for local inspection and for
the M-1 end-to-end verification described in the build plan.

Usage (no reinstall needed)::

    python -m ps06.status.cli --db status.db init-db
    python -m ps06.status.cli --db status.db add-doc   --case C1 --doc a
    python -m ps06.status.cli --db status.db add-doc   --case C1 --doc b
    python -m ps06.status.cli --db status.db set-stage --case C1 --doc a --stage OCR_DONE
    python -m ps06.status.cli --db status.db set-stage --case C1 --doc b --stage FAILED --error "knox 401"
    python -m ps06.status.cli --db status.db show-case --case C1
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from ps06.status import db
from ps06.status.states import DocumentStage
from ps06.status.store import DocumentStatus, StatusStore, StatusStoreError

DEFAULT_DB = "ps06_status.db"


def _print_status(s: DocumentStatus) -> None:
    err = f"  error={s.error_detail!r}" if s.error_detail else ""
    print(
        f"{s.case_id}/{s.document_id}: {s.stage.value}"
        f"  attempt={s.attempt_count}  updated={s.last_updated}{err}"
    )


def _cmd_init_db(store: StatusStore, args: argparse.Namespace) -> int:
    version = db.migrate(store._conn)  # connect() already migrated; report version
    print(f"database ready at {args.db!r} (schema version {version})")
    return 0


def _cmd_add_doc(store: StatusStore, args: argparse.Namespace) -> int:
    _print_status(store.register_document(args.case, args.doc))
    return 0


def _cmd_set_stage(store: StatusStore, args: argparse.Namespace) -> int:
    target = DocumentStage(args.stage)
    _print_status(
        store.transition(args.case, args.doc, target, error_detail=args.error)
    )
    return 0


def _cmd_show_case(store: StatusStore, args: argparse.Namespace) -> int:
    docs = store.list_for_case(args.case)
    if not docs:
        print(f"case {args.case!r}: no documents")
        return 0
    print(f"case {args.case!r}: {store.case_status(args.case).value}")
    for s in docs:
        print("  ", end="")
        _print_status(s)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ps06-status", description="Inspect/exercise the PS-06 status table."
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB, help=f"SQLite path (default: {DEFAULT_DB})"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create/migrate the database").set_defaults(
        func=_cmd_init_db
    )

    p_add = sub.add_parser("add-doc", help="register a document at RECEIVED")
    p_add.add_argument("--case", required=True)
    p_add.add_argument("--doc", required=True)
    p_add.set_defaults(func=_cmd_add_doc)

    p_set = sub.add_parser("set-stage", help="transition a document to a new stage")
    p_set.add_argument("--case", required=True)
    p_set.add_argument("--doc", required=True)
    p_set.add_argument(
        "--stage", required=True, choices=[s.value for s in DocumentStage]
    )
    p_set.add_argument("--error", default=None, help="error detail (for FAILED)")
    p_set.set_defaults(func=_cmd_set_stage)

    p_show = sub.add_parser("show-case", help="list a case's documents + derived status")
    p_show.add_argument("--case", required=True)
    p_show.set_defaults(func=_cmd_show_case)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
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
