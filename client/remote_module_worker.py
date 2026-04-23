"""Background thread with a dedicated uvloop for running async P2P coroutines.

Since PyTorch's autograd machinery is synchronous but the Hivemind P2P stack
is fully async, this module bridges the gap by running a persistent event-loop
thread and exposing :meth:`run_coroutine` which submits a coroutine and blocks
until the result is available.
"""

import asyncio
import logging
import os
from concurrent.futures import Future
from threading import Thread
from typing import Awaitable, Optional, TypeVar

from hivemind.utils import switch_to_uvloop

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RemoteModuleWorker:
    """Manages a background ``uvloop`` thread for async P2P operations.

    The class is stateless from the caller's perspective — just call
    :meth:`run_coroutine` with any awaitable and it will be executed on the
    shared event loop.  The loop (and its thread) are lazily created on first
    use and reused for subsequent calls within the same process.
    """

    _event_thread: Optional[Thread] = None
    _event_loop_fut: Optional[Future] = None
    _pid: Optional[int] = None

    @classmethod
    def _run_event_loop(cls) -> None:
        """Target function for the background event-loop thread."""
        try:
            loop = switch_to_uvloop()
            loop.set_debug(True)
            cls._event_loop_fut.set_result(loop)
        except Exception as e:
            cls._event_loop_fut.set_exception(e)
            return
        loop.run_forever()

    @classmethod
    def run_coroutine(cls, coro: Awaitable[T], return_future: bool = False):
        """Submit *coro* to the background event loop and return its result.

        Parameters
        ----------
        coro:
            An awaitable to execute on the background uvloop.
        return_future:
            If ``True``, return a :class:`~concurrent.futures.Future` instead
            of blocking until the coroutine completes.

        Returns
        -------
        The coroutine's result, or a ``Future`` if *return_future* is ``True``.

        Raises
        ------
        RuntimeError
            If the background event loop fails to start within the timeout.
        """
        # Re-create the thread if we are in a different process (e.g. after fork).
        if cls._event_thread is None or cls._pid != os.getpid():
            cls._pid = os.getpid()
            cls._event_loop_fut = Future()
            cls._event_thread = Thread(target=cls._run_event_loop, daemon=True)
            cls._event_thread.start()

        try:
            loop = cls._event_loop_fut.result(timeout=5)
        except Exception as e:
            raise RuntimeError("Failed to initialize event loop") from e

        future = asyncio.run_coroutine_threadsafe(coro, loop)

        if return_future:
            return future

        try:
            return future.result()
        except Exception as e:
            # Re-raise with the original traceback for better debugging.
            raise e