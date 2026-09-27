"""Token minting for the per-document OCR job (M-2).

Per the No-TTL architecture, each job invocation mints its own fresh token at
start, bounding the token's exposure window to one document's processing time
instead of a whole batch. :class:`AuthProvider` is the seam that lets
:class:`ps06.ocr.http_client.ReauthHTTPClient` mint (and re-mint, on a 401)
without depending on a live Cloudera/Workbench topology in tests.
"""

from __future__ import annotations

import os
from typing import Protocol


class AuthProvider(Protocol):
    """Something that can mint a fresh bearer token on demand."""

    def mint_token(self) -> str:
        """Return a fresh token. Called once at job start and again on a 401."""
        ...


class CDSWAuthProvider:
    """Real Cloudera AI Inference (Knox-gated) token provider.

    Not yet implemented: minting a fresh Knox JWT against the live topology
    requires the Cloudera/Workbench credentials tracked as Build Handoff open
    item #4, which are not available yet. Raises rather than returning a fake
    token, so a caller that forgets to inject a test double in tests fails
    loudly instead of silently "working" against no real auth.
    """

    def __init__(self, api_key_env_var: str = "CDSW_APIV2_KEY") -> None:
        self._api_key_env_var = api_key_env_var

    def mint_token(self) -> str:
        if not os.environ.get(self._api_key_env_var):
            raise NotImplementedError(
                f"CDSWAuthProvider requires live Cloudera/Workbench credentials "
                f"(env var {self._api_key_env_var!r} is unset) — blocked on "
                f"Build Handoff open item #4. Use FakeAuthProvider in tests."
            )
        raise NotImplementedError(
            "Live Knox JWT minting against the Cloudera AI Inference topology "
            "is not yet implemented — blocked on Build Handoff open item #4 "
            "(live Cloudera/Workbench credentials)."
        )


class FakeAuthProvider:
    """Test double: returns tokens from a fixed sequence, one per call.

    Lets tests assert that a re-auth actually happened after a 401 — e.g. the
    second :class:`~ps06.ocr.http_client.ReauthHTTPClient` call used a token
    different from the first. Raises once the sequence is exhausted rather than
    looping or returning a stale token, so a test that over-calls it fails
    loudly instead of masking a bug with a silently-reused token.
    """

    def __init__(self, tokens: list[str]) -> None:
        if not tokens:
            raise ValueError("FakeAuthProvider requires at least one token")
        self._tokens = list(tokens)
        self._calls = 0

    def mint_token(self) -> str:
        if self._calls >= len(self._tokens):
            raise RuntimeError(
                f"FakeAuthProvider.mint_token() called {self._calls + 1} times "
                f"but only {len(self._tokens)} token(s) were configured"
            )
        token = self._tokens[self._calls]
        self._calls += 1
        return token

    @property
    def call_count(self) -> int:
        return self._calls
