"""Provider error hierarchy and retry logic for tcode.

Reference: opencode session/retry.ts, processor.ts:350-378
"""
from __future__ import annotations
import asyncio
import time
from typing import Dict, Any, Optional


# ---- Error class hierarchy ----

class ProviderError(Exception):
    """Base class for all provider errors."""
    def __init__(self, message: str = "", retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class APIError(ProviderError):
    """General API error with optional HTTP status code."""
    def __init__(self, message: str = "", status_code: int = 0, retryable: bool = False,
                 headers: Optional[Dict[str, str]] = None):
        super().__init__(message, retryable=retryable)
        self.status_code = status_code
        self.headers = headers or {}


class RateLimitError(ProviderError):
    """Rate limit (429) error."""
    def __init__(self, message: str = "Rate limited", headers: Optional[Dict[str, str]] = None):
        super().__init__(message, retryable=True)
        self.headers = headers or {}


class ContextOverflowError(ProviderError):
    """Context window exceeded."""
    def __init__(self, message: str = "Context overflow"):
        super().__init__(message, retryable=False)


class AuthError(ProviderError):
    """Authentication/authorization error (401/403)."""
    def __init__(self, message: str = "Authentication error"):
        super().__init__(message, retryable=False)


class TimeoutError_(ProviderError):
    """Request timeout."""
    def __init__(self, message: str = "Request timed out"):
        super().__init__(message, retryable=True)


class AbortedError(ProviderError):
    """Request was aborted by user."""
    def __init__(self, message: str = "Aborted"):
        super().__init__(message, retryable=False)


class ServerError(ProviderError):
    """Server error (5xx)."""
    def __init__(self, message: str = "Server error", status_code: int = 500):
        super().__init__(message, retryable=True)
        self.status_code = status_code


# ---- Error classification ----

def map_provider_error(exc: Exception) -> Dict[str, Any]:
    """Map provider exceptions to structured error dicts.

    Returns: {"type": str, "message": str, "retryable": bool, "error": ProviderError}
    """
    msg = str(exc)
    lower = msg.lower()

    # Already a ProviderError
    if isinstance(exc, ProviderError):
        return {
            "type": type(exc).__name__,
            "message": msg,
            "retryable": exc.retryable,
            "error": exc,
        }

    # Extract HTTP status from SDK exceptions
    status = 0
    headers: Dict[str, str] = {}
    if hasattr(exc, 'response'):
        resp = exc.response
        status = getattr(resp, 'status_code', 0) or getattr(resp, 'status', 0)
        if hasattr(resp, 'headers'):
            try:
                headers = dict(resp.headers)
            except Exception:
                pass
    elif hasattr(exc, 'status_code'):
        status = exc.status_code
    elif hasattr(exc, 'http_status'):
        status = exc.http_status

    # Classify by status code
    if status == 429:
        err = RateLimitError(msg, headers=headers)
        return {"type": "rate_limit", "message": msg, "retryable": True, "error": err}
    if status >= 500:
        err = ServerError(msg, status_code=status)
        return {"type": "server_error", "message": msg, "retryable": True, "error": err}
    if status in (401, 403):
        err = AuthError(msg)
        return {"type": "auth_error", "message": msg, "retryable": False, "error": err}

    # Classify by error message content
    if 'context' in lower and ('overflow' in lower or 'length' in lower or 'too long' in lower or 'too many tokens' in lower):
        err = ContextOverflowError(msg)
        return {"type": "context_overflow", "message": msg, "retryable": False, "error": err}
    if 'timeout' in lower or 'timed out' in lower:
        err = TimeoutError_(msg)
        return {"type": "timeout", "message": msg, "retryable": True, "error": err}
    if 'rate' in lower and 'limit' in lower:
        err = RateLimitError(msg, headers=headers)
        return {"type": "rate_limit", "message": msg, "retryable": True, "error": err}
    if '429' in msg:
        err = RateLimitError(msg, headers=headers)
        return {"type": "rate_limit", "message": msg, "retryable": True, "error": err}
    if 'overloaded' in lower or 'unavailable' in lower or 'exhausted' in lower:
        err = ServerError(msg)
        return {"type": "server_error", "message": msg, "retryable": True, "error": err}
    if 'connection' in lower and ('error' in lower or 'refused' in lower or 'reset' in lower):
        err = ServerError(msg)
        return {"type": "connection_error", "message": msg, "retryable": True, "error": err}

    err = APIError(msg, status_code=status, retryable=False, headers=headers)
    return {"type": "api_error", "message": msg, "retryable": False, "error": err}


def is_retryable(error_dict: Dict[str, Any]) -> bool:
    """Check if a mapped error is retryable."""
    return bool(error_dict.get('retryable', False))


# ---- Retry logic ----

RETRY_INITIAL_DELAY = 2.0  # seconds
RETRY_BACKOFF_FACTOR = 2
RETRY_MAX_DELAY_NO_HEADERS = 30.0  # seconds
RETRY_MAX_DELAY = 300.0  # 5 minutes max
MAX_RETRIES = 5


def retry_delay(attempt: int, error_dict: Optional[Dict[str, Any]] = None) -> float:
    """Calculate retry delay for a given attempt.

    Checks for retry-after headers first, then uses exponential backoff.
    """
    # Check retry-after headers if available
    if error_dict:
        err = error_dict.get('error')
        headers = {}
        if hasattr(err, 'headers'):
            headers = err.headers or {}

        # retry-after-ms header (milliseconds)
        retry_ms = headers.get('retry-after-ms')
        if retry_ms:
            try:
                return float(retry_ms) / 1000.0
            except ValueError:
                pass

        # retry-after header (seconds or HTTP date)
        retry_after = headers.get('retry-after')
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass

    # Exponential backoff
    delay = RETRY_INITIAL_DELAY * (RETRY_BACKOFF_FACTOR ** (attempt - 1))
    max_delay = RETRY_MAX_DELAY_NO_HEADERS
    if error_dict and error_dict.get('error') and hasattr(error_dict['error'], 'headers'):
        if error_dict['error'].headers:
            max_delay = RETRY_MAX_DELAY
    return min(delay, max_delay)


async def retry_sleep(delay: float, abort_event: Optional[asyncio.Event] = None):
    """Sleep for a delay, aborting early if abort_event is set."""
    if abort_event:
        try:
            await asyncio.wait_for(
                _wait_for_event(abort_event),
                timeout=delay
            )
            # Event was set — abort
            raise AbortedError("Aborted during retry wait")
        except asyncio.TimeoutError:
            # Normal timeout — delay elapsed, continue
            pass
    else:
        await asyncio.sleep(delay)


async def _wait_for_event(event: asyncio.Event):
    """Wait until an asyncio.Event is set."""
    while not event.is_set():
        await asyncio.sleep(0.1)
