"""Tests for tcode error handling and retry logic."""
from __future__ import annotations
import asyncio
import pytest
from tcode.providers.errors import (
    ProviderError, APIError, RateLimitError, ContextOverflowError,
    AuthError, TimeoutError_, AbortedError, ServerError,
    map_provider_error, is_retryable, retry_delay, retry_sleep,
    RETRY_INITIAL_DELAY, RETRY_BACKOFF_FACTOR, MAX_RETRIES,
)


# ---- Error class hierarchy ----

def test_provider_error_base():
    err = ProviderError("test error", retryable=True)
    assert str(err) == "test error"
    assert err.retryable is True


def test_rate_limit_error():
    err = RateLimitError("Too many requests")
    assert err.retryable is True
    assert isinstance(err, ProviderError)


def test_context_overflow_error():
    err = ContextOverflowError("Too many tokens")
    assert err.retryable is False


def test_auth_error():
    err = AuthError("Invalid API key")
    assert err.retryable is False


def test_timeout_error():
    err = TimeoutError_("Connection timed out")
    assert err.retryable is True


def test_server_error():
    err = ServerError("Internal server error", status_code=503)
    assert err.retryable is True
    assert err.status_code == 503


def test_aborted_error():
    err = AbortedError()
    assert err.retryable is False


# ---- map_provider_error ----

def test_map_rate_limit_by_status():
    class FakeResp:
        status_code = 429
        headers = {}
    exc = Exception("rate limited")
    exc.response = FakeResp()
    result = map_provider_error(exc)
    assert result["type"] == "rate_limit"
    assert result["retryable"] is True


def test_map_server_error_by_status():
    class FakeResp:
        status_code = 503
        headers = {}
    exc = Exception("service unavailable")
    exc.response = FakeResp()
    result = map_provider_error(exc)
    assert result["type"] == "server_error"
    assert result["retryable"] is True


def test_map_auth_error_by_status():
    class FakeResp:
        status_code = 401
        headers = {}
    exc = Exception("unauthorized")
    exc.response = FakeResp()
    result = map_provider_error(exc)
    assert result["type"] == "auth_error"
    assert result["retryable"] is False


def test_map_context_overflow_by_message():
    exc = Exception("This model's context length is too long")
    result = map_provider_error(exc)
    assert result["type"] == "context_overflow"
    assert result["retryable"] is False


def test_map_timeout_by_message():
    exc = Exception("Request timed out after 30s")
    result = map_provider_error(exc)
    assert result["type"] == "timeout"
    assert result["retryable"] is True


def test_map_rate_limit_by_message():
    exc = Exception("Rate limit exceeded")
    result = map_provider_error(exc)
    assert result["type"] == "rate_limit"
    assert result["retryable"] is True


def test_map_overloaded_by_message():
    exc = Exception("The server is currently overloaded")
    result = map_provider_error(exc)
    assert result["type"] == "server_error"
    assert result["retryable"] is True


def test_map_unknown_error():
    exc = Exception("Something weird happened")
    result = map_provider_error(exc)
    assert result["type"] == "api_error"
    assert result["retryable"] is False


def test_map_already_provider_error():
    exc = RateLimitError("Throttled")
    result = map_provider_error(exc)
    assert result["type"] == "RateLimitError"
    assert result["retryable"] is True


# ---- is_retryable ----

def test_is_retryable_true():
    assert is_retryable({"retryable": True}) is True


def test_is_retryable_false():
    assert is_retryable({"retryable": False}) is False


def test_is_retryable_missing():
    assert is_retryable({}) is False


# ---- retry_delay ----

def test_retry_delay_exponential():
    d1 = retry_delay(1)
    d2 = retry_delay(2)
    d3 = retry_delay(3)
    assert d1 == RETRY_INITIAL_DELAY
    assert d2 == RETRY_INITIAL_DELAY * RETRY_BACKOFF_FACTOR
    assert d3 == RETRY_INITIAL_DELAY * RETRY_BACKOFF_FACTOR ** 2


def test_retry_delay_capped():
    d = retry_delay(100)
    assert d <= 30.0  # RETRY_MAX_DELAY_NO_HEADERS


def test_retry_delay_from_headers():
    err = RateLimitError("throttled", headers={"retry-after": "5"})
    d = retry_delay(1, {"error": err})
    assert d == 5.0


def test_retry_delay_from_ms_headers():
    err = RateLimitError("throttled", headers={"retry-after-ms": "3000"})
    d = retry_delay(1, {"error": err})
    assert d == 3.0


# ---- retry_sleep ----

@pytest.mark.asyncio
async def test_retry_sleep_completes():
    await retry_sleep(0.01)


@pytest.mark.asyncio
async def test_retry_sleep_aborted():
    abort = asyncio.Event()
    abort.set()  # Already set
    with pytest.raises(AbortedError):
        await retry_sleep(10.0, abort_event=abort)


@pytest.mark.asyncio
async def test_retry_sleep_aborted_during_wait():
    abort = asyncio.Event()

    async def _set_later():
        await asyncio.sleep(0.05)
        abort.set()

    asyncio.create_task(_set_later())
    with pytest.raises(AbortedError):
        await retry_sleep(10.0, abort_event=abort)
