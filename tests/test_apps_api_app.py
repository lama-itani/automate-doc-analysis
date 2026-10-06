"""Tests for apps/api_app.py, the Cloudera AI Application entry point.

The entry file calls ``main()`` as its last statement. ``_load`` executes every
statement except that call, so the helpers can be tested on their own.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import socket
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from tests.test_jobs_ocr_job import _failure, _run_like_pbj  # noqa: E402

APP_FILE = Path(__file__).resolve().parent.parent / "apps" / "api_app.py"
ENDPOINT = "http://fake/v1"


def _load():
    source = APP_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    last = tree.body[-1]
    assert ast.unparse(last) == "main()", "entry file must end with main()"
    tree.body = tree.body[:-1]
    module = types.ModuleType("api_app_under_test")
    module.__file__ = str(APP_FILE)
    exec(compile(tree, str(APP_FILE), "exec"), module.__dict__)
    return module


@pytest.fixture
def entry():
    return _load()


@pytest.fixture
def env(tmp_path):
    ui = tmp_path / "dist"
    ui.mkdir()
    (ui / "index.html").write_text("<html></html>", encoding="utf-8")
    return {
        "PS06_ENDPOINT_URL": ENDPOINT,
        "PS06_DATA_DIR": str(tmp_path / "data"),
        "PS06_UI_DIST": str(ui),
        "CDSW_APP_PORT": "8100",
        "CDSW_PROJECT_ID": "proj-1",
    }


@pytest.fixture
def fake_cmlapi(monkeypatch):
    monkeypatch.setitem(sys.modules, "cmlapi", types.ModuleType("cmlapi"))


class FakeServer:
    """Records its config. ``run`` starts a loop the way uvicorn does."""

    instances: list = []

    def __init__(self, config):
        self.config = config
        self.started = False
        FakeServer.instances.append(self)

    def run(self):
        async def serve():
            self.started = True

        asyncio.run(serve())


@pytest.fixture
def fake_server(monkeypatch):
    FakeServer.instances = []
    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    return FakeServer


# --- Environment checks -----------------------------------------------------


class TestResolvePort:
    def test_reads_port(self, entry):
        assert entry.resolve_port({"CDSW_APP_PORT": " 8100 "}) == 8100

    @pytest.mark.parametrize("value", [None, "", "  "])
    def test_missing_port_fails_loudly(self, entry, value):
        environ = {} if value is None else {"CDSW_APP_PORT": value}
        with pytest.raises(RuntimeError, match="CDSW_APP_PORT is not set"):
            entry.resolve_port(environ)

    @pytest.mark.parametrize("value", ["abc", "0", "70000"])
    def test_bad_port_fails_loudly(self, entry, value):
        with pytest.raises(RuntimeError, match="CDSW_APP_PORT must be"):
            entry.resolve_port({"CDSW_APP_PORT": value})


class TestLoadSettings:
    def test_missing_endpoint_fails_loudly(self, entry, env):
        del env["PS06_ENDPOINT_URL"]
        with pytest.raises(RuntimeError, match="PS06_ENDPOINT_URL is not set"):
            entry.load_settings(env)

    def test_invalid_setting_is_a_runtime_error(self, entry, env):
        env["PS06_MAX_CONCURRENCY"] = "many"
        with pytest.raises(RuntimeError, match="PS06_MAX_CONCURRENCY"):
            entry.load_settings(env)

    def test_defaults(self, entry):
        settings = entry.load_settings({"PS06_ENDPOINT_URL": ENDPOINT})
        assert settings.db_path == Path("/home/cdsw/ps06_data/ps06.db")
        assert settings.ui_dist == APP_FILE.parent.parent / "ui" / "dist"


class TestCheckUi:
    def test_built_ui(self, entry, env):
        assert entry.check_ui(entry.load_settings(env)) is True

    def test_missing_ui_warns(self, entry, env, tmp_path, caplog):
        env["PS06_UI_DIST"] = str(tmp_path / "nope")
        with caplog.at_level(logging.WARNING):
            assert entry.check_ui(entry.load_settings(env)) is False
        assert "UI not built" in caplog.text


class TestCheckWorkbench:
    def test_missing_project_id(self, entry, fake_cmlapi):
        with pytest.raises(RuntimeError, match="CDSW_PROJECT_ID"):
            entry.check_workbench({})

    def test_missing_cmlapi(self, entry, monkeypatch):
        monkeypatch.setitem(sys.modules, "cmlapi", None)
        with pytest.raises(RuntimeError, match="cmlapi"):
            entry.check_workbench({"CDSW_PROJECT_ID": "p"})


# --- main and serve ----------------------------------------------------------


class TestMain:
    def test_reports_every_missing_variable_at_once(self, entry):
        with pytest.raises(RuntimeError) as info:
            entry.main({})
        assert "CDSW_APP_PORT" in str(info.value)
        assert "PS06_ENDPOINT_URL" in str(info.value)

    def test_serves_on_loopback_and_app_port(self, entry, env, fake_cmlapi, fake_server):
        with pytest.raises(RuntimeError, match="stopped unexpectedly"):
            entry.main(env)
        (server,) = fake_server.instances
        assert server.config.host == "127.0.0.1"
        assert server.config.port == 8100
        assert (Path(env["PS06_DATA_DIR"]) / "ps06.db").exists()


class TestServe:
    def test_startup_exit_becomes_runtime_error(self, entry):
        class ExitServer(FakeServer):
            def run(self):
                sys.exit(3)  # what uvicorn does when the port is taken

        with pytest.raises(RuntimeError, match="failed: SystemExit"):
            entry.serve(object(), 8100, server_cls=ExitServer)

    def test_never_started(self, entry):
        class IdleServer(FakeServer):
            def run(self):
                return None

        with pytest.raises(RuntimeError, match="did not start"):
            entry.serve(object(), 8100, server_cls=IdleServer)

    def test_works_inside_a_running_event_loop(self, entry):
        """The PBJ kernel may run an asyncio loop on the main thread."""

        async def kernel_cell():
            entry.serve(object(), 8100, server_cls=FakeServer)

        with pytest.raises(RuntimeError, match="stopped unexpectedly"):
            asyncio.run(kernel_cell())

    def test_real_uvicorn_serves_health(self, entry, env, fake_cmlapi):
        """Real uvicorn and the real app, started from inside a running loop."""
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        env["CDSW_APP_PORT"] = str(port)
        servers = []

        class StoppableServer(uvicorn.Server):
            def __init__(self, config):
                super().__init__(config)
                servers.append(self)

        bodies = []

        def probe():
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if servers and servers[0].started:
                    url = f"http://127.0.0.1:{port}/api/health"
                    with urllib.request.urlopen(url, timeout=5) as r:
                        bodies.append(json.loads(r.read()))
                    break
                time.sleep(0.05)
            if servers:
                servers[0].should_exit = True

        from ps06.api.app import create_app

        app = create_app(entry.load_settings(env))
        threading.Thread(target=probe, daemon=True).start()

        async def kernel_cell():
            entry.serve(app, port, server_cls=StoppableServer)

        with pytest.raises(RuntimeError, match="stopped unexpectedly"):
            asyncio.run(kernel_cell())
        assert bodies and bodies[0]["ui_built"] is True


# --- Whole file in a PBJ-like kernel ----------------------------------------


@pytest.fixture
def kernel(monkeypatch, tmp_path):
    pytest.importorskip("IPython")
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    return _run_like_pbj


def test_kernel_missing_env_fails_with_normal_exception(kernel, monkeypatch):
    for name in ("CDSW_APP_PORT", "PS06_ENDPOINT_URL"):
        monkeypatch.delenv(name, raising=False)
    results = kernel(APP_FILE.read_text(encoding="utf-8"))
    assert all(_failure(r) is None for r in results[:-1])
    error = _failure(results[-1])
    assert isinstance(error, RuntimeError)
    assert "CDSW_APP_PORT" in str(error) and "PS06_ENDPOINT_URL" in str(error)


def test_kernel_full_start(kernel, monkeypatch, env, fake_cmlapi, fake_server):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    results = kernel(APP_FILE.read_text(encoding="utf-8"))
    error = _failure(results[-1])
    assert isinstance(error, RuntimeError)
    assert "stopped unexpectedly" in str(error)
    assert fake_server.instances[0].config.port == 8100
