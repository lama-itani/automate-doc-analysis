"""Tests for ps06.ocr.auth."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx2 as httpx
import openai
import pytest

from ps06.ocr.auth import (
    DEFAULT_JWT_PATH,
    AuthTokenError,
    CDSWAuthProvider,
    FakeAuthProvider,
)
from ps06.ocr.http_client import ReauthHTTPClient


class TestFakeAuthProvider:
    def test_returns_tokens_in_order(self):
        auth = FakeAuthProvider(["tok1", "tok2", "tok3"])
        assert auth.mint_token() == "tok1"
        assert auth.mint_token() == "tok2"
        assert auth.mint_token() == "tok3"

    def test_tracks_call_count(self):
        auth = FakeAuthProvider(["tok1", "tok2"])
        assert auth.call_count == 0
        auth.mint_token()
        assert auth.call_count == 1

    def test_raises_once_exhausted(self):
        auth = FakeAuthProvider(["tok1"])
        auth.mint_token()
        with pytest.raises(RuntimeError, match="only 1 token"):
            auth.mint_token()

    def test_requires_at_least_one_token(self):
        with pytest.raises(ValueError):
            FakeAuthProvider([])


SECRET = "eyJSECRET.TOKEN.VALUE"


def _write(path, content):
    path.write_text(content, encoding="utf-8")
    return path


class TestCDSWAuthProvider:
    def test_default_path_is_tmp_jwt(self):
        assert DEFAULT_JWT_PATH == "/tmp/jwt"

    def test_returns_access_token(self, tmp_path):
        jwt = _write(tmp_path / "jwt", json.dumps({"access_token": SECRET}))
        assert CDSWAuthProvider(str(jwt)).mint_token() == SECRET

    def test_strips_whitespace(self, tmp_path):
        jwt = _write(tmp_path / "jwt", json.dumps({"access_token": f" {SECRET}\n"}))
        assert CDSWAuthProvider(str(jwt)).mint_token() == SECRET

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(AuthTokenError, match="not found"):
            CDSWAuthProvider(str(tmp_path / "nope")).mint_token()

    def test_bad_json_raises_without_echoing_contents(self, tmp_path):
        jwt = _write(tmp_path / "jwt", f"not json {SECRET}")
        with pytest.raises(AuthTokenError, match="not valid JSON") as err:
            CDSWAuthProvider(str(jwt)).mint_token()
        assert SECRET not in str(err.value)
        assert err.value.__cause__ is None

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"access_token": ""},
            {"access_token": "   "},
            {"access_token": None},
            {"access_token": 123},
            {"token": SECRET},
            [SECRET],
            SECRET,
        ],
    )
    def test_no_usable_access_token_raises(self, tmp_path, payload):
        jwt = _write(tmp_path / "jwt", json.dumps(payload))
        with pytest.raises(AuthTokenError, match="no usable") as err:
            CDSWAuthProvider(str(jwt)).mint_token()
        assert SECRET not in str(err.value)

    def test_rereads_file_on_every_call(self, tmp_path):
        jwt = _write(tmp_path / "jwt", json.dumps({"access_token": "old"}))
        auth = CDSWAuthProvider(str(jwt))
        assert auth.mint_token() == "old"
        _write(jwt, json.dumps({"access_token": "new"}))
        assert auth.mint_token() == "new"

    def test_401_remint_uses_refreshed_token(self, tmp_path):
        jwt = _write(tmp_path / "jwt", json.dumps({"access_token": "old"}))
        keys = []

        class _Client:
            def __init__(self, *, base_url, api_key):
                keys.append(api_key)
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                if self is clients[0]:
                    _write(jwt, json.dumps({"access_token": "new"}))
                    response = httpx.Response(401, request=httpx.Request("POST", "http://t"))
                    raise openai.AuthenticationError("401", response=response, body=None)
                message = SimpleNamespace(content="ok")
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        clients = []

        def factory(**kwargs):
            clients.append(_Client(**kwargs))
            return clients[-1]

        client = ReauthHTTPClient(CDSWAuthProvider(str(jwt)), "http://t", "m", client_factory=factory)
        assert client.chat_completion(messages=[], max_tokens=1) == "ok"
        assert keys == ["old", "new"]