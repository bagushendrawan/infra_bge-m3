from fastapi import Request


async def get_worker(request: Request):
    return request.app.state.worker
