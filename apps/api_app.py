"""Cloudera AI Application entry point for the PS-06 API and UI.

Set the Application's script to ``apps/api_app.py``. Environment variables:

* ``PS06_ENDPOINT_URL`` (required): the model endpoint's OpenAI base URL.
* ``PS06_DATA_DIR`` (default ``/home/cdsw/ps06_data``), ``PS06_UI_DIST``
  (default ``<repo>/ui/dist``) and the other ``PS06_*`` settings read by
  :meth:`ApiSettings.from_env`.
* ``CDSW_APP_PORT``: set by the platform. Cloudera AI 1.5.5 docs: bind to
  ``127.0.0.1`` on this port.

PBJ runtime: like ``jobs/ocr_job.py``, this file runs inside an IPython kernel,
one top-level statement at a time. So:

* It never exits. Any failure is a normal ``RuntimeError`` with a clear message.
* The kernel may already run an asyncio loop on the main thread, and
  ``uvicorn`` starts its own with ``asyncio.run``. So the server runs on a
  separate thread and the main thread waits for it. Off the main thread,
  uvicorn installs no signal handlers; the platform stops the Application by
  ending the pod.

Keep top-level statements few: errors are reported by chunk index.
"""
import logging
import os
import threading

HOST = "127.0.0.1"
PORT_VAR = "CDSW_APP_PORT"
ENDPOINT_VAR = "PS06_ENDPOINT_URL"

logger = logging.getLogger("ps06.apps.api_app")


def resolve_port(environ=None):
    """Return ``CDSW_APP_PORT`` as an int. Raise ``RuntimeError`` if missing or bad."""
    environ = os.environ if environ is None else environ
    raw = environ.get(PORT_VAR, "").strip()
    if not raw:
        raise RuntimeError(
            f"{PORT_VAR} is not set. Run this file as a Cloudera AI Application; "
            "the platform sets the port."
        )
    try:
        port = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{PORT_VAR} must be an integer, got {raw!r}") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError(f"{PORT_VAR} must be 1-65535, got {port}")
    return port


def load_settings(environ=None):
    """Return :class:`ApiSettings` from the environment, or raise ``RuntimeError``."""
    from ps06.api.app import ApiSettings

    environ = os.environ if environ is None else environ
    if not environ.get(ENDPOINT_VAR, "").strip():
        raise RuntimeError(
            f"{ENDPOINT_VAR} is not set. Add it to the Application's environment "
            "variables (the model endpoint's OpenAI base URL, ending in /v1)."
        )
    try:
        return ApiSettings.from_env(environ)
    except ValueError as exc:
        raise RuntimeError(f"invalid PS06 settings: {exc}") from exc


def check_ui(settings):
    """Warn if the built UI is missing. The API still serves ``/api/*``."""
    index = settings.ui_dist / "index.html"
    if index.is_file():
        logger.info("serving UI from %s", settings.ui_dist)
        return True
    logger.warning(
        "UI not built: %s is missing. Only /api/* is served. Build ui/ locally "
        "(npm run build) and commit ui/dist/.",
        index,
    )
    return False


def check_workbench(environ=None):
    """Fail fast on what every job launch needs: ``cmlapi`` and ``CDSW_PROJECT_ID``."""
    environ = os.environ if environ is None else environ
    if not environ.get("CDSW_PROJECT_ID", "").strip():
        raise RuntimeError("CDSW_PROJECT_ID is not set; jobs cannot be launched")
    try:
        import cmlapi  # noqa: F401  (only on the Cloudera AI workbench)
    except ImportError as exc:
        raise RuntimeError(
            "the 'cmlapi' package is not importable; jobs cannot be launched"
        ) from exc


def serve(app, port, host=HOST, server_cls=None):
    """Run uvicorn on its own thread and block. Always ends with ``RuntimeError``.

    The server only returns if it failed to start or stopped on its own; both
    are errors for an Application, so neither ends silently.
    """
    import uvicorn

    server_cls = uvicorn.Server if server_cls is None else server_cls
    server = server_cls(uvicorn.Config(app, host=host, port=port, log_level="info"))
    errors = []

    def run():
        try:
            server.run()
        except BaseException as exc:  # noqa: BLE001  includes uvicorn's sys.exit(1) on bind errors
            errors.append(exc)

    thread = threading.Thread(target=run, name="ps06-uvicorn")
    thread.start()
    thread.join()
    if errors:
        raise RuntimeError(
            f"API server on {host}:{port} failed: {type(errors[0]).__name__}: "
            f"{errors[0]} (see the log above)"
        ) from errors[0]
    if not getattr(server, "started", False):
        raise RuntimeError(f"API server did not start on {host}:{port} (see the log above)")
    raise RuntimeError(f"API server on {host}:{port} stopped unexpectedly")


def main(environ=None):
    """Check the environment, build the app and serve it. Never returns normally."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    environ = os.environ if environ is None else environ
    problems = []
    port = settings = None
    try:
        port = resolve_port(environ)
    except RuntimeError as exc:
        problems.append(str(exc))
    try:
        settings = load_settings(environ)
    except RuntimeError as exc:
        problems.append(str(exc))
    if problems:
        raise RuntimeError("ps06 Application cannot start: " + " | ".join(problems))

    check_workbench(environ)
    check_ui(settings)

    from ps06.api.app import create_app

    app = create_app(settings)
    logger.info(
        "ps06 API on %s:%s | data %s | endpoint %s | model %s | concurrency %d | "
        "job runtime %s | job cpu %d, memory %d GB, timeout %d s",
        HOST, port, settings.db_path.parent, settings.endpoint_url,
        settings.model_name, settings.max_concurrency, settings.job_runtime,
        settings.job_cpu, settings.job_memory_gb, settings.job_timeout_seconds,
    )
    serve(app, port)


main()