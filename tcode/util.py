import time
import os
import uuid
import asyncio
from typing import Tuple, Optional, Callable, Awaitable


def next_id(prefix: str = "id") -> str:
    # Use UUID4-based ids to avoid collisions across process restarts
    return f"{prefix}-{uuid.uuid4().hex}"


def truncate_text(text: str, limit: int) -> Tuple[str, bool, Optional[str]]:
    """Truncate text to `limit` characters. If truncated, return path where remainder is stored."""
    if len(text) <= limit:
        return text, False, None
    head = text[:limit]
    remainder = text[limit:]
    path = os.path.join(os.getcwd(), ".tcode_truncated_" + str(int(time.time())) + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(remainder)
    return head, True, path


async def retry_async(func: Callable[[], Awaitable], retries: int = 3, delay: float = 0.5, backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """Run async func() with retries and exponential backoff."""
    current = 0
    d = delay
    while True:
        try:
            return await func()
        except exceptions as e:
            current += 1
            if current > retries:
                raise
            await asyncio.sleep(d)
            d *= backoff
