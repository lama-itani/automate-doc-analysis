"""Tests for ps06.ocr.auth."""

from __future__ import annotations

import pytest

from ps06.ocr.auth import CDSWAuthProvider, FakeAuthProvider


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


class TestCDSWAuthProvider:
    def test_raises_not_implemented_without_env_var(self, monkeypatch):
        monkeypatch.delenv("CDSW_APIV2_KEY", raising=False)
        auth = CDSWAuthProvider()
        with pytest.raises(NotImplementedError, match="open item #4"):
            auth.mint_token()

    def test_raises_not_implemented_even_with_env_var(self, monkeypatch):
        # Presence of the key doesn't make live minting work — it's still
        # unimplemented pending the real Cloudera/Workbench integration.
        monkeypatch.setenv("CDSW_APIV2_KEY", "some-key")
        auth = CDSWAuthProvider()
        with pytest.raises(NotImplementedError):
            auth.mint_token()
