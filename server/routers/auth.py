"""
Optional bearer-token authentication for LAN deployments.

If TTS_API_KEY is not set every request passes through.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from .. import config


async def verify_api_key(authorization: str | None = Header(default=None)) -> None:
    if config.API_KEY is None:
        return  # auth disabled
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header.",
        )
    token = authorization.removeprefix("Bearer ").strip()
    if token != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
