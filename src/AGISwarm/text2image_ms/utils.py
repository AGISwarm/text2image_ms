"""Utility functions for LLM engines"""

import asyncio
import threading
from abc import abstractmethod
from typing import Dict, Generic, List, Protocol, TypeVar, cast, runtime_checkable

from pydantic import BaseModel

__ABORT_EVENTS = {}
__QUEUE = []


def abort_generation_request(request_id: str):
    """Abort generation request"""
    if request_id in __ABORT_EVENTS:
        __ABORT_EVENTS[request_id].set()


def generation_request_queued_func(func, wait_time=0.2):
    """Decorator for generation requests"""

    def abort_response(request_id: str):
        return {
            "request_id": request_id,
            "type": "abort",
            "msg": "Generation aborted.",
        }

    def waiting_response(request_id: str):
        """Waiting response"""
        return {
            "request_id": request_id,
            "type": "waiting",
            "msg": f"Waiting for {__QUEUE.index(request_id)} requests to finish...\n",
        }

    async def wrapper(*args, **kwargs):
        request_id = args[1]
        __ABORT_EVENTS[request_id] = threading.Event()
        __QUEUE.append(request_id)
        try:
            while __QUEUE[0] != request_id:
                await asyncio.sleep(wait_time)
                if __ABORT_EVENTS[request_id].is_set():
                    yield abort_response(request_id)
                    return
                yield waiting_response(request_id)
            async for response in func(*args, **kwargs):
                if __ABORT_EVENTS[request_id].is_set():
                    yield abort_response(request_id)
                    return
                yield response
        except asyncio.CancelledError as e:
            print(e)
        finally:
            __QUEUE.remove(request_id)
            __ABORT_EVENTS[request_id].clear()
            __ABORT_EVENTS.pop(request_id)

    return wrapper
