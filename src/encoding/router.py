import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from src.encoding.dependencies import get_worker
from src.encoding.exceptions import EncodingError
from src.encoding.schemas import EncodeRequest, EncodeResponse

router = APIRouter()


@router.post(
    "/v1/encode",
    response_model=EncodeResponse,
    response_model_exclude_none=True,
    tags=["encoding"],
    summary="Encode an ordered text batch",
    description="Select dense, sparse, and ColBERT outputs. Long texts are truncated.",
    responses={
        413: {"description": "Result too large"},
        422: {"description": "Invalid input"},
        503: {"description": "Busy or unavailable"},
        504: {"description": "Timeout"},
        500: {"description": "Inference failed"},
    },
)
async def encode(
    payload: EncodeRequest, request: Request, worker: Annotated[object, Depends(get_worker)]
):
    settings = request.app.state.settings
    if (
        len(payload.texts) > settings.max_items
        or any(len(t) > settings.max_text_chars for t in payload.texts)
        or sum(map(len, payload.texts)) > settings.max_total_chars
    ):
        raise EncodingError(422, "Text batch exceeds configured limits")
    task = asyncio.create_task(worker.encode(payload.model_dump()))
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.25)
            if not task.done() and await request.is_disconnected():
                task.cancel()
                break
        return await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
