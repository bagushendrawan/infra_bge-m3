import asyncio
import logging
import multiprocessing as mp

from fastapi.concurrency import run_in_threadpool

from src.encoding.exceptions import EncodingError

logger = logging.getLogger(__name__)


def serve(connection, settings):
    try:
        from src.encoding.service import Encoder

        encoder = Encoder(settings)
        connection.send(("ready", None))
        while True:
            args = connection.recv()
            try:
                connection.send(("ok", encoder.encode(*args)))
            except EncodingError as exc:
                connection.send(("error", (exc.status, exc.detail)))
            except Exception:
                logger.exception("Encoding failed")
                connection.send(("error", (500, "Encoding failed")))
    except Exception:
        logger.exception("Worker stopped")
    finally:
        connection.close()


class Worker:
    def __init__(self, settings):
        self.settings = settings
        self.ready = False
        self.busy = False
        self.process = None

    async def start(self):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=serve, args=(child, self.settings), daemon=True)
        self.process.start()
        child.close()
        try:
            status, _ = await self.receive(self.settings.startup_timeout)
            self.ready = status == "ready"
        except (EncodingError, EOFError, OSError):
            await self.close()
            raise RuntimeError("BGE-M3 worker initialization failed")

    async def restart(self):
        try:
            await self.start()
        except RuntimeError:
            logger.exception("Model reload failed; readiness is false")

    async def receive(self, timeout):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not self.connection.poll():
            if loop.time() >= deadline:
                raise EncodingError(504, "Encoding worker deadline exceeded")
            await asyncio.sleep(0.02)
        return await run_in_threadpool(self.connection.recv)

    async def close(self):
        self.ready = False
        if self.process:
            if self.process.is_alive():
                self.process.terminate()
            await run_in_threadpool(self.process.join, 5)
            if self.process.is_alive():
                self.process.kill()
                await run_in_threadpool(self.process.join)
            self.connection.close()

    async def encode(self, *args):
        if self.busy or not self.ready or not self.process:
            raise EncodingError(503, "Encoding worker unavailable")
        self.busy = True
        try:
            if not self.process.is_alive():
                raise EOFError("Worker exited")
            await run_in_threadpool(self.connection.send, args)
            status, value = await self.receive(self.settings.timeout)
            if status == "error":
                raise EncodingError(*value)
            return value
        except (EOFError, OSError) as exc:
            await self.close()
            await self.restart()
            raise EncodingError(503, "Encoding worker crashed") from exc
        except (EncodingError, asyncio.CancelledError) as exc:
            if isinstance(exc, asyncio.CancelledError) or exc.status in {503, 504}:
                await self.close()
                await self.restart()
            raise
        finally:
            self.busy = False
