from __future__ import annotations

from typing import Any

import pytest

from apps.api.app.middleware import BodySizeLimitMiddleware


async def _consume_body_app(scope, receive, send):
    while True:
        message = await receive()
        if message["type"] != "http.request" or not message.get("more_body", False):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


@pytest.mark.asyncio
async def test_streamed_body_without_content_length_is_limited():
    messages = iter(
        [
            {"type": "http.request", "body": b"12345", "more_body": True},
            {"type": "http.request", "body": b"67890", "more_body": False},
        ]
    )
    sent: list[dict[str, Any]] = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    middleware = BodySizeLimitMiddleware(_consume_body_app, max_bytes=8)
    await middleware(
        {"type": "http", "headers": [], "method": "POST", "path": "/"},
        receive,
        send,
    )

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_declared_oversize_body_is_rejected_before_reading():
    sent: list[dict[str, Any]] = []
    read_called = False

    async def receive():
        nonlocal read_called
        read_called = True
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    middleware = BodySizeLimitMiddleware(_consume_body_app, max_bytes=8)
    await middleware(
        {
            "type": "http",
            "headers": [(b"content-length", b"9")],
            "method": "POST",
            "path": "/",
        },
        receive,
        send,
    )

    assert sent[0]["status"] == 413
    assert read_called is False
