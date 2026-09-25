import pytest
from fastapi.responses import JSONResponse

from apps.api.app import main


@pytest.mark.asyncio
async def test_metric_persistence_is_deferred_to_background(monkeypatch):
    calls: list[tuple[str, int, float]] = []

    def fake_persist(path: str, status_code: int, latency_ms: float) -> None:
        calls.append((path, status_code, latency_ms))

    monkeypatch.setattr(main, "_persist_request_metric", fake_persist)
    response = JSONResponse({"ok": True})

    main._attach_metric_background(response, "/v1/predictions", 200, 12.5)

    assert calls == []
    assert response.background is not None

    await response.background()

    assert calls == [("/v1/predictions", 200, 12.5)]
