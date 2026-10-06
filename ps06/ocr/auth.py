"""Token minting for the per-document OCR job (M-2).

Per the No-TTL architecture, each job invocation mints its own fresh token at
start, bounding the token's exposure window to one document's processing time
instead of a whole batch. :class:`AuthProvider` is the seam that lets
:class:`ps06.ocr.http_client.ReauthHTTPClient` mint (and re-mint, on a 401)
without depending on a live Cloudera/Workbench topology in tests.
"""

from __future__ import annotations

import json
from typing import Protocol


class AuthProvider(Protocol):
    """Something that can mint a fresh bearer token on demand."""

    def mint_token(self) -> str:
        """Return a fresh token. Called once at job start and again on a 401."""
        ...


DEFAULT_JWT_PATH = "/tmp/jwt"


class AuthTokenError(RuntimeError):
    """The job's token file is missing or unusable.

    Messages name the file path and error type only. They never contain the
    token or the file's contents.
    """


class CDSWAuthProvider:
    """Real token provider: reads the access token Cloudera AI places in every
    session and job at ``/tmp/jwt`` (confirmed in step-0 workbench tests,
    2026-10-06).

    Re-reads the file on every call, so
    :class:`~ps06.ocr.http_client.ReauthHTTPClient`'s 401 re-mint picks up a
    refreshed token if the platform rotated it. Raises
    :class:`AuthTokenError` (never returns an empty or stale value) if the
    file is missing, not JSON, or has no usable ``access_token``.
    """

    def __init__(self, jwt_path: str = DEFAULT_JWT_PATH) -> None:
        self._jwt_path = jwt_path

    def mint_token(self) -> str:
        try:
            with open(self._jwt_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError as exc:
            raise AuthTokenError(
                f"token file {self._jwt_path} not found (it only exists inside "
                f"Cloudera AI sessions and jobs)"
            ) from exc
        except (OSError, ValueError) as exc:
            # from None: a chained JSON error could echo file contents.
            raise AuthTokenError(
                f"token file {self._jwt_path} is unreadable or not valid JSON "
                f"({type(exc).__name__})"
            ) from None
        token = data.get("access_token") if isinstance(data, dict) else None
        if not isinstance(token, str) or not token.strip():
            raise AuthTokenError(
                f"token file {self._jwt_path} has no usable 'access_token'"
            )
        return token.strip()


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