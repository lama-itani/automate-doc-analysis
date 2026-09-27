"""Tests for ps06.ocr.http_client.ReauthHTTPClient.

Exercises the 401-retry control flow entirely against a fake OpenAI-shaped
client — no network, no live Cloudera/Workbench credentials.
"""

from __future__ import annotations

import httpx2 as httpx
import openai
import pytest

from ps06.ocr.auth import FakeAuthProvider
from ps06.ocr.http_client import AuthRetryExhausted, ReauthHTTPClient


def _auth_error(message: str = "unauthorized") -> openai.AuthenticationError:
    request = httpx.Request("POST", "http://test")
    response = httpx.Response(401, request=request)
    return openai.AuthenticationError(message, response=response, body=None)


class _FakeCompletions:
    """Records the api_key it was built with and replays scripted results."""

    def __init__(self, results: list[object]) -> None:
        self._results = list(results)
        self.calls: list[None] = []

    def create(self, **kwargs):
        self.calls.append(None)
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeOpenAIClient:
    """Stands in for openai.OpenAI: only .chat.completions.create is used."""

    instances: list["_FakeOpenAIClient"] = []

    def __init__(self, *, base_url: str, api_key: str, results: list[object]) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.chat = _FakeChat(_FakeCompletions(results))
        _FakeOpenAIClient.instances.append(self)


def _make_completion(text: str):
    class _Msg:
        content = text

    class _Choice:
        message = _Msg()

    class _Completion:
        choices = [_Choice()]

    return _Completion()


@pytest.fixture(autouse=True)
def _reset_fake_instances():
    _FakeOpenAIClient.instances = []
    yield
    _FakeOpenAIClient.instances = []


def _factory_for(results_by_call: list[list[object]]):
    """Build a client_factory that hands out `results_by_call[i]` results to
    the i-th constructed client, in the order clients are constructed."""
    remaining = list(results_by_call)

    def factory(*, base_url: str, api_key: str):
        results = remaining.pop(0)
        return _FakeOpenAIClient(base_url=base_url, api_key=api_key, results=results)

    return factory


class TestReauthHTTPClient:
    def test_success_no_retry(self):
        auth = FakeAuthProvider(["tok1"])
        factory = _factory_for([[_make_completion("hello")]])
        client = ReauthHTTPClient(
            auth, "http://inference", "model-x", client_factory=factory
        )
        result = client.chat_completion(messages=[], max_tokens=10)
        assert result == "hello"
        assert auth.call_count == 1

    def test_401_then_success_retries_once_with_new_token(self):
        auth = FakeAuthProvider(["tok1", "tok2"])
        factory = _factory_for(
            [[_auth_error()], [_make_completion("recovered")]]
        )
        client = ReauthHTTPClient(
            auth, "http://inference", "model-x", client_factory=factory
        )
        result = client.chat_completion(messages=[], max_tokens=10)
        assert result == "recovered"
        assert auth.call_count == 2
        # The second underlying client was built with the freshly-minted token.
        assert [c.api_key for c in _FakeOpenAIClient.instances] == ["tok1", "tok2"]

    def test_401_twice_raises_auth_retry_exhausted(self):
        auth = FakeAuthProvider(["tok1", "tok2"])
        factory = _factory_for([[_auth_error()], [_auth_error()]])
        client = ReauthHTTPClient(
            auth, "http://inference", "model-x", client_factory=factory
        )
        with pytest.raises(AuthRetryExhausted):
            client.chat_completion(messages=[], max_tokens=10)
        assert auth.call_count == 2

    def test_non_401_error_not_retried(self):
        auth = FakeAuthProvider(["tok1"])
        boom = RuntimeError("boom")
        factory = _factory_for([[boom]])
        client = ReauthHTTPClient(
            auth, "http://inference", "model-x", client_factory=factory
        )
        with pytest.raises(RuntimeError, match="boom"):
            client.chat_completion(messages=[], max_tokens=10)
        # No re-auth attempted for a non-auth error.
        assert auth.call_count == 1
        assert len(_FakeOpenAIClient.instances) == 1

    def test_empty_content_returns_empty_string(self):
        auth = FakeAuthProvider(["tok1"])
        factory = _factory_for([[_make_completion(None)]])
        client = ReauthHTTPClient(
            auth, "http://inference", "model-x", client_factory=factory
        )
        assert client.chat_completion(messages=[], max_tokens=10) == ""
