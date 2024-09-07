"""
Utility functions for text2image_ms.

"""

import asyncio

import nest_asyncio


def _to_task(future: asyncio.Future, as_task: bool, loop: asyncio.AbstractEventLoop):
    if not as_task or isinstance(future, asyncio.Task):
        return future
    return loop.create_task(future)  # type: ignore


def asyncio_run(future, as_task=True):
    """
    A better implementation of `asyncio.run`.

    :param future: A future or task or call of an async method.
    :param as_task: Forces the future to be scheduled as task (needed for e.g. aiohttp).
    """

    try:  # pylint: disable=no-else-return
        loop = asyncio.get_running_loop()
    except RuntimeError:  # no event loop running:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(_to_task(future, as_task, loop))
    else:
        nest_asyncio.apply(loop)
        return asyncio.run(_to_task(future, as_task, loop))  # type: ignore
