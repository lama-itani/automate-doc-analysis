"""401-triggered re-auth-and-retry wrapper around the Inference HTTP client (M-2).

This is the auth chokepoint named in the No-TTL architecture: the OCR job mints
a fresh token at start (bounding its exposure to one document's processing
time), but as cheap insurance against even a single document's runtime being
marginal against the Knox TTL, every completions call is wrapped so that a 401
triggers exactly one re-mint-and-retry before giving up loudly.

This module is the load-bearing, fully-tested piece of the auth story — the
retry control flow itself is real logic, not a stub, and is exercised in tests
via an injected fake client (see :class:`ReauthHTTPClient`'s ``client_factory``
parameter), so no live Cloudera/Workbench credentials are needed to test it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import openai

from ps06.ocr.auth import AuthProvider


class AuthRetryExhausted(Exception):
    """Raised when a completions call still 401s after a fresh re-mint."""


class ReauthHTTPClient:
    """Wraps an OpenAI-compatible client with fresh-JWT + 401-retry.

    Mints a token at construction (one JWT per job invocation, per the
    architecture) and builds the underlying client. On
    :class:`openai.AuthenticationError` from a completions call, mints a new
    token, rebuilds the client, and retries the call exactly once. A second
    401 raises :class:`AuthRetryExhausted` rather than retrying indefinitely —
    unlimited retries would silently mask a genuinely broken credential, which
    is exactly the kind of silent failure this build is meant to eliminate.
    Any other exception propagates immediately, uncaught: only 401s are
    treated as an auth problem worth retrying.
    """

    def __init__(
        self,
        auth: AuthProvider,
        base_url: str,
        model_name: str,
        client_factory: Callable[..., Any] = openai.OpenAI,
    ) -> None:
        self._auth = auth
        self._base_url = base_url
        self._model_name = model_name
        self._client_factory = client_factory
        self._client = self._build_client()

    def _build_client(self) -> Any:
        token = self._auth.mint_token()
        return self._client_factory(base_url=self._base_url, api_key=token)

    def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Run one chat-completions call, retrying once on a 401.

        Returns the stripped text content of the first choice's message.
        """
        try:
            return self._call(messages, max_tokens, temperature, extra_body)
        except openai.AuthenticationError:
            self._client = self._build_client()
            try:
                return self._call(messages, max_tokens, temperature, extra_body)
            except openai.AuthenticationError as exc:
                raise AuthRetryExhausted(
                    "chat completion still returned 401 after re-minting a "
                    "fresh token — the credential is likely invalid, not "
                    "merely expired"
                ) from exc

    def _call(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        extra_body: dict[str, Any] | None,
    ) -> str:
        completion = self._client.chat.completions.create(
            model=self._model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            extra_body=extra_body or None,
        )
        content = completion.choices[0].message.content
        return content.strip() if content else ""
