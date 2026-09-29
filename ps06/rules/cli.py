"""Small CLI to exercise the PS-06 rules/aggregation engine by hand (M-4).

Not part of the production surface — a convenience for local inspection,
mirroring ``ps06/ocr/cli.py``, ``ps06/orchestrator/cli.py``, and
``ps06/status/cli.py``'s style. Readiness (whether every document in the case
has reached ``OCR_DONE``) is not checked here — that's the orchestrator's job;
``run-case`` simply calls :func:`ps06.rules.engine.evaluate_case`, which raises
:class:`~ps06.rules.adapter.MissingExtractionError` if a registered document
has no persisted extraction yet.

Usage (no reinstall needed)::

    python -m ps06.status.cli --db ps06.db add-doc --case C1 --doc a
    python -m ps06.ocr.cli --db ps06.db run --case C1 --doc a --file /path/a.pdf ...

    python -m ps06.rules.cli --db ps06.db run-case --case C1
    python -m ps06.rules.cli --db ps06.db run-case --case C1 --rules-config ps06/rules/rules.yaml --json

    python -m ps06.rules.cli --db ps06.db show-case --case C1 --json
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Optional, Sequence

from ps06.rules.adapter import MissingExtractionError
from ps06.rules.engine import evaluate_case, get_snapshot
from ps06.rules.rules_config import RulesConfig
from ps06.rules.snapshot import ReportSnapshot
from ps06.status import db
from ps06.status.store import StatusStore, StatusStoreError

DEFAULT_DB = "ps06_status.db"


def _print_snapshot(snapshot: ReportSnapshot) -> None:
    print(f"case {snapshot.case_id!r}: {snapshot.estado}")
    print(f"  justificacion: {snapshot.justificacion}")
    print(f"  problemas: {', '.join(snapshot.problemas)}")
    print("  s1 (inventario):")
    for row in snapshot.s1_rows:
        print(f"    {row.tipo}: {row.estado}")
    print("  s2 (evaluacion por documento):")
    for row in snapshot.s2_rows:
        print(f"    {row.documento}: {row.evaluacion_credencial} / {row.consistencia}")
    print("  s3 (linaje):")
    for row in snapshot.s3_rows:
        print(f"    {row.generacion} ({row.relacion}): {row.nombre_completo}")
    if snapshot.s3_footnote:
        print(f"  s3 footnote: {snapshot.s3_footnote}")


def _build_rules_config(args: argparse.Namespace) -> RulesConfig:
    if args.rules_config is None:
        return RulesConfig()
    return RulesConfig.from_yaml(args.rules_config)


def _cmd_run_case(store: StatusStore, args: argparse.Namespace) -> int:
    config = _build_rules_config(args)
    evaluation_date = date.fromisoformat(args.evaluation_date) if args.evaluation_date else date.today()
    snapshot = evaluate_case(args.case, store, config, evaluation_date)

    _print_snapshot(snapshot)
    if args.json:
        print(snapshot.model_dump_json(indent=2))
    return 0


def _cmd_show_case(store: StatusStore, args: argparse.Namespace) -> int:
    snapshot = get_snapshot(store, args.case)
    if snapshot is None:
        print(f"error: no rule evaluation found for case={args.case!r}", file=sys.stderr)
        return 1

    _print_snapshot(snapshot)
    if args.json:
        print(snapshot.model_dump_json(indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ps06-rules", description="Run/inspect the PS-06 rules/aggregation engine."
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB, help=f"SQLite path (default: {DEFAULT_DB})"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run-case", help="evaluate a case's rules and persist the snapshot")
    p_run.add_argument("--case", required=True)
    p_run.add_argument(
        "--rules-config", default=None,
        help="path to a rules.yaml override (default: built-in RulesConfig() defaults)",
    )
    p_run.add_argument(
        "--evaluation-date", default=None,
        help="ISO-8601 date to evaluate document expiry against (default: today)",
    )
    p_run.add_argument("--json", action="store_true", help="also print the full ReportSnapshot as JSON")
    p_run.set_defaults(func=_cmd_run_case)

    p_show = sub.add_parser("show-case", help="show a persisted ReportSnapshot for one case")
    p_show.add_argument("--case", required=True)
    p_show.add_argument("--json", action="store_true", help="print the full ReportSnapshot as JSON")
    p_show.set_defaults(func=_cmd_show_case)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    conn = db.connect(args.db)
    try:
        store = StatusStore(conn)
        return args.func(store, args)
    except (StatusStoreError, MissingExtractionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
