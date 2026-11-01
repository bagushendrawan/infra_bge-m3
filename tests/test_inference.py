"""Opt-in real model verification: BGE_M3_TEST_INFERENCE=1 uv run pytest tests/test_inference.py."""

import math
import os

import pytest

from src.encoding.config import Settings
from src.encoding.schemas import EncodeRequest
from src.encoding.worker import Worker


@pytest.mark.skipif(os.getenv("BGE_M3_TEST_INFERENCE") != "1", reason="Real model test is opt-in")
async def test_real_inference():
    worker = Worker(Settings())
    await worker.start()
    process = worker.process
    try:
        payload = EncodeRequest(
            texts=["Hello world", "Selamat pagi"], return_sparse=True, return_colbert_vecs=True
        ).model_dump()
        for _ in range(2):
            result = await worker.encode(payload)
            assert worker.process is process
            assert len(result["dense_vecs"]) == 2
            for dense in result["dense_vecs"]:
                assert len(dense) == 1024 and all(math.isfinite(x) for x in dense)
            assert all(result["lexical_weights"])
            assert all(result["colbert_vecs"])
    finally:
        await worker.close()
