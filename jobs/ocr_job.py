"""Cloudera AI job entry point for the PS-06 per-document OCR command.

Set the job's script to ``jobs/ocr_job.py`` and pass the ``ps06.ocr.cli``
arguments as the job arguments, e.g.::

    --db ps06_status.db run --case C1 --doc a --file /home/cdsw/in/a.pdf
    --endpoint-url <url> --model-name <model> --auth cdsw

Why this file exists: the PBJ runtime executes a job script inside an IPython
kernel, one top-level statement ("chunk") at a time.

* Any ``SystemExit`` fails the job, even ``sys.exit(0)``. So this file never
  exits. Success = the script ends. Failure = a normal exception.
* Job arguments are in the ``JOB_ARGUMENTS`` env var, not ``sys.argv`` (which
  holds the kernel's own ``-f <kernel.json>``).

Auth: jobs must use ``--auth cdsw`` (the default), which reads the platform's
token file. ``--auth fake`` and ``--fake-token`` are refused here: a token in
the job arguments can leak into logs and the job's stored settings.

Status-table handling is not repeated here: ``ps06.ocr.job.run`` already moves
the document to FAILED with ``error_detail`` before this file sees the error.

Keep top-level statements few: errors are reported by chunk index.
"""
import os
import shlex
import sys

from ps06.ocr import cli

_KERNEL_FLAG = "-f"
_FAKE_TOKEN_FLAG = "--fake-token"
_AUTH_FLAG = "--auth"
_FAKE_AUTH = "fake"


def _is_fake_token_flag(flag):
    """True for ``--fake-token`` and any abbreviation argparse would accept.

    argparse accepts unique prefixes (``--fake``, ``--fa``...). ``--f`` is
    ambiguous with ``--file`` and ``--fi...`` is not a prefix, so ``--fa`` is
    the shortest case that matters.
    """
    return len(flag) >= 4 and _FAKE_TOKEN_FLAG.startswith(flag)


def _redact(args):
    """Copy of ``args`` with fake-token values hidden, safe for error text."""
    safe = []
    hide_next = False
    for arg in args:
        flag, sep, _ = arg.partition("=")
        if hide_next:
            safe.append("***")
            hide_next = False
        elif _is_fake_token_flag(flag):
            if sep:
                safe.append(f"{flag}=***")
            else:
                safe.append(arg)
                hide_next = True
        else:
            safe.append(arg)
    return safe


def _check_real_auth(args):
    """Raise ``RuntimeError`` if ``args`` ask for fake auth or carry a token."""
    uses_fake = False
    for i, arg in enumerate(args):
        flag, sep, value = arg.partition("=")
        if _is_fake_token_flag(flag):
            uses_fake = True
        elif flag == _AUTH_FLAG:
            if not sep and i + 1 < len(args):
                value = args[i + 1]
            if value == _FAKE_AUTH:
                uses_fake = True
    if uses_fake:
        raise RuntimeError(
            "ps06 OCR job refuses --auth fake / --fake-token: a token in job "
            "arguments can leak into logs. Use --auth cdsw (the default); "
            f"args={_redact(args)}"
        )


def resolve_args(environ=None, argv=None):
    """Return the CLI arguments for this job.

    ``JOB_ARGUMENTS`` wins when set and non-blank. Otherwise ``argv[1:]``,
    unless ``argv`` is the IPython kernel's (``... -f <kernel.json>``). Then
    there are no arguments, and the CLI rejects the empty command.
    """
    environ = os.environ if environ is None else environ
    argv = sys.argv if argv is None else argv
    raw = environ.get("JOB_ARGUMENTS", "")
    if raw.strip():
        return shlex.split(raw)
    if len(argv) > 1 and argv[1] == _KERNEL_FLAG:
        return []
    return list(argv[1:])


def run_job(args=None):
    """Run ``ps06.ocr.cli.main``. Return on success, raise on any failure."""
    if args is None:
        args = resolve_args()
    _check_real_auth(args)
    try:
        rc = cli.main(args)
    except SystemExit as exc:
        # argparse errors, --help, and the CLI's own exit on bad auth flags.
        raise RuntimeError(
            f"ps06 OCR job exited early (SystemExit {exc.code!r}); "
            f"args={_redact(args)}"
        ) from exc
    if rc != 0:
        raise RuntimeError(
            f"ps06 OCR job failed (exit code {rc}); args={_redact(args)}"
        )


run_job()