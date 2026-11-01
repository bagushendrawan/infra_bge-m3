import httpx
import pytest

from src.encoding.dependencies import get_worker
from src.main import create_app


class FakeWorker:
    async def encode(self, payload):
        assert payload["return_dense"] is True
        assert payload["return_sparse"] is False
        return {"model": "BAAI/bge-m3", "dense_vecs": [[1.0] * 1024]}


async def test_defaults():
    app = create_app()
    app.dependency_overrides[get_worker] = lambda: FakeWorker()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/encode", json={"texts": ["hello"]})
    assert response.status_code == 200
    assert len(response.json()["dense_vecs"][0]) == 1024
    assert "colbert_vecs" not in response.json()


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": []},
        {"texts": ["  "]},
        {"texts": ["ok"], "max_length": 8193},
        {"texts": ["ok"], "return_dense": False},
    ],
)
async def test_invalid(payload):
    app = create_app()
    app.dependency_overrides[get_worker] = lambda: FakeWorker()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        assert (await client.post("/v1/encode", json=payload)).status_code == 422


@pytest.mark.parametrize("status", [413, 500, 503, 504])
async def test_errors(status):
    from src.encoding.exceptions import EncodingError

    class FailedWorker:
        async def encode(self, payload):
            raise EncodingError(status, "failed")

    app = create_app()
    app.dependency_overrides[get_worker] = lambda: FailedWorker()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/encode", json={"texts": ["ok"]})
    assert response.status_code == status
    assert response.headers["X-Request-ID"]


async def test_limits_and_readiness():
    from src.encoding.config import Settings

    app = create_app(Settings(max_items=1))
    app.dependency_overrides[get_worker] = lambda: FakeWorker()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        assert (await client.get("/health/live")).status_code == 200
        assert (await client.get("/health/ready")).status_code == 503
        assert (await client.post("/v1/encode", json={"texts": ["a", "b"]})).status_code == 422
