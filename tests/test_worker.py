import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from src.encoding.config import Settings
from src.encoding.exceptions import EncodingError
from src.encoding.worker import Worker


def worker():
    value = Worker(Settings())
    value.ready = True
    value.process = Mock()
    value.connection = Mock()
    value.receive = AsyncMock(return_value=("ok", {"model": "test"}))
    return value


async def test_reuse_and_busy():
    value = worker()
    for _ in range(2):
        assert await value.encode({}) == {"model": "test"}
    assert value.connection.send.call_count == 2
    value.busy = True
    with pytest.raises(EncodingError) as error:
        await value.encode({})
    assert error.value.status == 503


@pytest.mark.parametrize(
    "failure",
    [
        EncodingError(504, "timeout"),
        EncodingError(503, "oom"),
        EOFError(),
        asyncio.CancelledError(),
    ],
)
async def test_recovery(failure):
    value = worker()
    value.receive.side_effect = failure
    value.close = AsyncMock()
    value.start = AsyncMock()
    with pytest.raises((EncodingError, asyncio.CancelledError)):
        await value.encode({})
    value.close.assert_awaited_once()
    value.start.assert_awaited_once()
    assert not value.busy


async def test_deadline():
    value = Worker(Settings())
    value.connection = Mock()
    value.connection.poll.return_value = False
    with pytest.raises(EncodingError) as error:
        await value.receive(0.001)
    assert error.value.status == 504


async def test_shutdown():
    value = worker()
    await value.close()
    assert not value.ready
    value.process.terminate.assert_called_once()
    value.connection.close.assert_called_once()
