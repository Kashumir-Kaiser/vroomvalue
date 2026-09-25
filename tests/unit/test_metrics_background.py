import pytest

from apps.api.app.database import is_invalid_input_status
from apps.api.app.middleware import (
    BodySizeLimitMiddleware,
    RequestObservabilityMiddleware,
)


@pytest.mark.parametrize("status_code", [400, 413, 422])
def test_malformed_or_oversized_requests_count_as_invalid_input(status_code: int):
    assert is_invalid_input_status(status_code) is True


@pytest.mark.parametrize("status_code", [401, 404, 409, 500, 503])
def test_other_failures_do_not_count_as_invalid_input(status_code: int):
    assert is_invalid_input_status(status_code) is False


@pytest.mark.asyncio
async def test_observability_records_after_response_body_is_sent():
    events: list[str] = []

    async def endpoint(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        events.append("start-sent")
        await send({"type": "http.response.body", "body": b"ok"})
        events.append("body-sent")

    def recorder(path: str, status_code: int, latency_ms: float) -> None:
        assert path == "/v1/predictions"
        assert status_code == 200
        assert latency_ms >= 0
        events.append("metric-recorded")

    middleware = RequestObservabilityMiddleware(
        endpoint,
        metric_recorder=recorder,
        excluded_paths=set(),
        request_id_pattern=r"^[A-Za-z0-9._-]{1,64}$",
    )

    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await middleware(
        {
            "type": "http",
            "path": "/v1/predictions",
            "headers": [],
            "state": {},
        },
        receive,
        send,
    )

    assert events == ["start-sent", "body-sent", "metric-recorded"]
    headers = dict(sent[0]["headers"])
    assert b"x-request-id" in headers
    assert headers[b"x-content-type-options"] == b"nosniff"


@pytest.mark.asyncio
async def test_413_reaches_metric_pipeline_without_response_mutation():
    calls: list[tuple[str, int]] = []

    async def endpoint(scope, receive, send):
        raise AssertionError("Oversized request should not reach endpoint")

    def recorder(path: str, status_code: int, latency_ms: float) -> None:
        calls.append((path, status_code))

    app = BodySizeLimitMiddleware(endpoint, max_bytes=8)
    app = RequestObservabilityMiddleware(
        app,
        metric_recorder=recorder,
        excluded_paths=set(),
        request_id_pattern=r"^[A-Za-z0-9._-]{1,64}$",
    )

    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "path": "/v1/predictions",
            "headers": [(b"content-length", b"9")],
            "state": {},
        },
        receive,
        send,
    )

    assert sent[0]["status"] == 413
    assert calls == [("/v1/predictions", 413)]
