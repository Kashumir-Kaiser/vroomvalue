from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Collection
from typing import Any

from starlette.concurrency import run_in_threadpool

ASGIReceive = Callable[[], Awaitable[dict[str, Any]]]
ASGISend = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[dict[str, Any], ASGIReceive, ASGISend], Awaitable[None]]
MetricRecorder = Callable[[str, int, float], None]

logger = logging.getLogger("vroomvalue.api")


class RequestBodyTooLarge(RuntimeError):
    pass


class BodySizeLimitMiddleware:
    """Enforce a request-body limit for both fixed-length and streamed requests."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(
        self,
        send: ASGISend,
        status_code: int,
        detail: str,
    ) -> None:
        body = json.dumps({"detail": detail}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                await self._reject(send, 400, "Invalid Content-Length header.")
                return
            if declared < 0:
                await self._reject(send, 400, "Invalid Content-Length header.")
                return
            if declared > self.max_bytes:
                await self._reject(
                    send,
                    413,
                    f"Request body exceeds {self.max_bytes} bytes.",
                )
                return

        received = 0
        response_started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if response_started:
                raise
            await self._reject(
                send,
                413,
                f"Request body exceeds {self.max_bytes} bytes.",
            )


class RequestObservabilityMiddleware:
    """Add request metadata and persist service metrics after response delivery."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        metric_recorder: MetricRecorder,
        excluded_paths: Collection[str],
        request_id_pattern: str,
    ) -> None:
        self.app = app
        self.metric_recorder = metric_recorder
        self.excluded_paths = frozenset(excluded_paths)
        self.request_id_pattern = re.compile(request_id_pattern)

    def _request_id(self, scope: dict[str, Any]) -> str:
        for key, value in scope.get("headers", []):
            if key.lower() != b"x-request-id":
                continue
            try:
                candidate = value.decode("ascii")
            except UnicodeDecodeError:
                break
            if self.request_id_pattern.fullmatch(candidate):
                return candidate
            break
        return uuid.uuid4().hex

    @staticmethod
    def _set_header(
        headers: list[tuple[bytes, bytes]],
        name: bytes,
        value: bytes,
    ) -> None:
        lower_name = name.lower()
        headers[:] = [
            (key, existing)
            for key, existing in headers
            if key.lower() != lower_name
        ]
        headers.append((name, value))

    async def _record_metric(
        self,
        path: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        if path in self.excluded_paths:
            return
        try:
            await run_in_threadpool(
                self.metric_recorder,
                path,
                status_code,
                latency_ms,
            )
        except Exception:
            logger.exception(
                json.dumps(
                    {
                        "event": "request.metric_persistence_failed",
                        "route": path,
                        "status_code": status_code,
                    }
                )
            )

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path", ""))
        request_id = self._request_id(scope)
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status_code = 500
        response_started = False
        response_complete = False

        async def observed_send(message: dict[str, Any]) -> None:
            nonlocal status_code, response_started, response_complete
            if message.get("type") == "http.response.start":
                response_started = True
                status_code = int(message["status"])
                headers = list(message.get("headers", []))
                self._set_header(headers, b"x-request-id", request_id.encode("ascii"))
                self._set_header(headers, b"x-content-type-options", b"nosniff")
                self._set_header(headers, b"referrer-policy", b"no-referrer")
                self._set_header(headers, b"x-frame-options", b"DENY")
                message = {**message, "headers": headers}
            elif (
                message.get("type") == "http.response.body"
                and not message.get("more_body", False)
            ):
                response_complete = True
            await send(message)

        try:
            await self.app(scope, receive, observed_send)
        except Exception:
            latency_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                json.dumps(
                    {
                        "event": "request.failed",
                        "request_id": request_id,
                        "route": path,
                        "latency_ms": round(latency_ms, 2),
                    }
                )
            )
            if response_started:
                await self._record_metric(path, status_code, latency_ms)
                raise

            body = json.dumps(
                {
                    "detail": "Internal server error.",
                    "request_id": request_id,
                }
            ).encode("utf-8")
            await observed_send(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await observed_send(
                {
                    "type": "http.response.body",
                    "body": body,
                }
            )

        latency_ms = (time.perf_counter() - started) * 1000
        if response_complete:
            await self._record_metric(path, status_code, latency_ms)

        logger.info(
            json.dumps(
                {
                    "event": "request.completed",
                    "request_id": request_id,
                    "route": path,
                    "status_code": status_code,
                    "latency_ms": round(latency_ms, 2),
                }
            )
        )
