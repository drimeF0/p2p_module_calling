import asyncio
import os
from concurrent.futures import Future
from threading import Thread
from typing import Awaitable, Optional

from hivemind.utils import switch_to_uvloop

import logging

class RemoteModuleWorker:
    """Local thread for managing async tasks related to RemoteModule"""

    _event_thread: Optional[Thread] = None
    _event_loop_fut: Optional[Future] = None
    _pid: Optional[int] = None

    @classmethod
    def _run_event_loop(cls):
        try:
            loop = switch_to_uvloop()
            loop.set_debug(True)
            cls._event_loop_fut.set_result(loop)
        except Exception as e:
            cls._event_loop_fut.set_exception(e)
        loop.run_forever()

    @classmethod
    def run_coroutine(cls, coro: Awaitable, return_future: bool = False):
        if cls._event_thread is None or cls._pid != os.getpid():
            cls._pid = os.getpid()
            cls._event_loop_fut = Future()
            cls._event_thread = Thread(target=cls._run_event_loop, daemon=True)
            cls._event_thread.start()

        try:
            loop = cls._event_loop_fut.result(timeout=5)  # Add a timeout to avoid hanging indefinitely
        except Exception as e:
            raise RuntimeError("Failed to initialize event loop") from e
        
        for task in asyncio.all_tasks(loop=loop):
            # log the task
            logging.debug(task)

        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future if return_future else future.result()