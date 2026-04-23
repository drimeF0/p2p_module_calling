"""Server-side servicer that hosts PyTorch modules and serves forward/backward RPCs.

Each :class:`ModuleServicer` runs in its own ``ForkProcess`` so that the heavy
GIL-bound tensor computation does not interfere with the DHT or the caller's
event loop.  Modules are registered by name, and the servicer exposes four P2P
RPC endpoints:

* ``rpc_forward_module`` — unary forward pass
* ``rpc_forward_module_stream`` — streaming forward pass (large payloads)
* ``rpc_backward_module`` — unary backward pass
* ``rpc_backward_module_stream`` — streaming backward pass (large payloads)
"""

import asyncio
import multiprocessing as mp
from typing import Dict, List, Optional, AsyncIterator

import torch
from hivemind.dht import DHT
from hivemind.p2p import ServicerBase, P2PContext
from hivemind.p2p.p2p_daemon import P2P
from hivemind.utils import MPFuture
from hivemind.utils.asyncio import switch_to_uvloop
from hivemind.utils.logging import get_logger

from p2p_module_calling.module_service import (
    ModuleBackwardRequest,
    ModuleBackwardResponse,
    ModuleForwardRequest,
    ModuleForwardResponse,
)
from p2p_module_calling.utils import (
    DEFAULT_ZERO_SAFETENSOR_BYTES,
    deserialize_tensors,
    fill_dict,
    serialize_tensors,
    split_bytes,
    unfill_dict,
)

logger = get_logger(__name__)


class ModuleServicer(ServicerBase, mp.context.ForkProcess):
    """P2P servicer that executes forward/backward passes on registered modules.

    Parameters
    ----------
    dht:
        A running Hivemind DHT instance.
    modules:
        A mapping from module identifiers to the actual ``torch.nn.Module``
        instances that will serve the requests.
    """

    # -- Pre-built error responses (avoid re-allocating on every error) -------

    _ERR_MODULE_NOT_FOUND_FORWARD = ModuleForwardResponse(
        error_message="Module not found",
        success=False,
        output_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES,
    )
    _ERR_MODULE_NOT_FOUND_BACKWARD = ModuleBackwardResponse(
        error_message="Module not found",
        success=False,
        grad_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES,
    )
    _ERR_MISSING_INPUT_OR_GRAD = ModuleBackwardResponse(
        error_message="Input and gradient tensors must be provided",
        success=False,
        grad_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES,
    )

    def __init__(self, dht: DHT, modules: Dict[str, torch.nn.Module]):
        super().__init__()
        self.dht = dht
        self._p2p: Optional[P2P] = None
        self._inner_pipe, self._outer_pipe = mp.Pipe(duplex=False)
        self.ready = MPFuture()
        self.modules = modules

    # -- Process lifecycle ----------------------------------------------------

    def run(self) -> None:
        """Entry point for the forked process — sets up the event loop and P2P handlers."""
        torch.set_num_threads(1)
        asyncio_loop = switch_to_uvloop()
        stop = asyncio.Event()
        asyncio_loop.add_reader(self._inner_pipe.fileno(), stop.set)

        async def _run():
            try:
                self._p2p = await self.dht.replicate_p2p()
                await self.add_p2p_handlers(self._p2p, balanced=True)
                self.ready.set_result(True)
            except Exception as e:
                logger.error("ModuleServicer failed to start", exc_info=True)
                self.ready.set_exception(e)
            try:
                await stop.wait()
            finally:
                await self.remove_p2p_handlers(self._p2p)

        try:
            asyncio_loop.run_until_complete(_run())
        except KeyboardInterrupt:
            pass

    def run_in_background(
        self,
        await_ready: bool = True,
        timeout: Optional[float] = None,
    ):
        """Start the servicer in a background process.

        If *await_ready* is ``True`` (the default), this method blocks until
        the servicer is ready to accept requests or until *timeout* expires.
        """
        self.start()
        if await_ready:
            return self.wait_until_ready(timeout)

    def wait_until_ready(self, timeout: Optional[float] = None):
        """Block until the servicer signals readiness."""
        return self.ready.result(timeout=timeout)

    def shutdown(self) -> None:
        """Gracefully shut down the servicer process."""
        if self.is_alive():
            self._outer_pipe.send("_shutdown")
            self.join(self.shutdown_timeout)
            if self.is_alive():
                logger.warning(
                    "ModuleServicer did not shut down within the grace period; terminating",
                )
                self.terminate()
        else:
            logger.warning("ModuleServicer shutdown called but process is already dead")

    # -- Internal helpers -----------------------------------------------------

    def _backward(
        self,
        tensors: Dict[str, torch.Tensor],
        grad_tensors: List[torch.Tensor],
    ) -> None:
        """Run ``torch.autograd.backward`` on the given tensors."""
        torch.autograd.backward(
            list(tensors.values()),
            grad_tensors,
            retain_graph=False,
            create_graph=False,
        )

    @staticmethod
    def _prepare_inputs_for_backward(
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Detach inputs and enable ``requires_grad`` on floating-point tensors.

        This is necessary because the backward pass re-runs the forward pass
        on the server side under ``torch.enable_grad()`` — we need leaf tensors
        that track gradients.
        """
        return {
            key: (
                tensor.detach().requires_grad_(True)
                if tensor.is_floating_point()
                else tensor.detach()
            )
            for key, tensor in inputs.items()
        }

    def get_module(self, module_id: str) -> Optional[torch.nn.Module]:
        """Look up a registered module by its identifier."""
        return self.modules.get(module_id)

    # -- RPC handlers ---------------------------------------------------------

    async def rpc_forward_module(
        self,
        request: ModuleForwardRequest,
        context: P2PContext,
    ) -> ModuleForwardResponse:
        """Unary forward pass: deserialize inputs, run the module, serialize outputs."""
        logger.info("Received forward request for module %s", request.module_id)

        module = self.get_module(request.module_id)
        if module is None:
            return self._ERR_MODULE_NOT_FOUND_FORWARD

        input_tensors = deserialize_tensors(request.input_tensor_bytes)
        with torch.no_grad():
            output_tensors = module(**input_tensors)

        output_tensor_bytes = serialize_tensors(output_tensors)
        return ModuleForwardResponse(success=True, output_tensor_bytes=output_tensor_bytes)

    async def rpc_forward_module_stream(
        self,
        requests: AsyncIterator[ModuleForwardRequest],
        context: P2PContext,
    ) -> AsyncIterator[ModuleForwardResponse]:
        """Streaming forward pass: accumulate chunks, run the module, stream back."""
        bytes_buffer = bytearray()
        module_id: Optional[str] = None

        async for message in requests:
            bytes_buffer.extend(message.input_tensor_bytes)
            module_id = message.module_id

        if module_id is None:
            # Empty stream — nothing to do.
            return

        tensor_dict = deserialize_tensors(bytes(bytes_buffer))
        module = self.get_module(module_id)
        if module is None:
            yield self._ERR_MODULE_NOT_FOUND_FORWARD
            return

        with torch.no_grad():
            output_tensors = module(**tensor_dict)

        output_tensor_bytes = serialize_tensors(output_tensors)
        for chunk in split_bytes(output_tensor_bytes):
            yield ModuleForwardResponse(success=True, output_tensor_bytes=chunk)

    async def rpc_backward_module(
        self,
        request: ModuleBackwardRequest,
        context: P2PContext,
    ) -> ModuleBackwardResponse:
        """Unary backward pass: re-run forward with gradients enabled, then backward."""
        logger.info("Received backward request for module %s", request.module_id)

        module = self.get_module(request.module_id)
        if module is None:
            return self._ERR_MODULE_NOT_FOUND_BACKWARD

        if request.input_tensor_bytes is None or request.grad_tensor_bytes is None:
            return self._ERR_MISSING_INPUT_OR_GRAD

        input_tensors = deserialize_tensors(request.input_tensor_bytes)
        input_tensors = self._prepare_inputs_for_backward(input_tensors)
        grad_tensors = deserialize_tensors(request.grad_tensor_bytes)
        grads = unfill_dict(grad_tensors)

        with torch.enable_grad():
            output_tensors = module(**input_tensors)
            self._backward(output_tensors, grads)

        grad_tensors_list = [
            tensor.grad if isinstance(tensor, torch.Tensor) else torch.zeros_like(tensor)
            for tensor in input_tensors.values()
        ]
        grad_tensors_dict = fill_dict(grad_tensors_list)
        grad_tensor_bytes = serialize_tensors(grad_tensors_dict)
        return ModuleBackwardResponse(success=True, grad_tensor_bytes=grad_tensor_bytes)

    async def rpc_backward_module_stream(
        self,
        requests: AsyncIterator[ModuleBackwardRequest],
        context: P2PContext,
    ) -> AsyncIterator[ModuleBackwardResponse]:
        """Streaming backward pass: accumulate chunks, compute gradients, stream back."""
        bytes_buffer_inputs = bytearray()
        bytes_buffer_grads = bytearray()
        module_id: Optional[str] = None

        async for request in requests:
            if request.input_tensor_bytes is not None:
                bytes_buffer_inputs.extend(request.input_tensor_bytes)
            if request.grad_tensor_bytes is not None:
                bytes_buffer_grads.extend(request.grad_tensor_bytes)
            module_id = request.module_id

        if module_id is None:
            # Empty stream — nothing to do.
            return

        module = self.get_module(module_id)
        if module is None:
            yield self._ERR_MODULE_NOT_FOUND_BACKWARD
            return

        if not bytes_buffer_inputs or not bytes_buffer_grads:
            yield self._ERR_MISSING_INPUT_OR_GRAD
            return

        input_tensors = deserialize_tensors(bytes_buffer_inputs)
        input_tensors = self._prepare_inputs_for_backward(input_tensors)
        grad_tensors = deserialize_tensors(bytes_buffer_grads)
        grads = unfill_dict(grad_tensors)

        with torch.enable_grad():
            output_tensors = module(**input_tensors)
            self._backward(output_tensors, grads)

        grad_tensors_list = [
            tensor.grad if isinstance(tensor, torch.Tensor) else torch.zeros_like(tensor)
            for tensor in input_tensors.values()
        ]
        grad_tensors_dict = fill_dict(grad_tensors_list)
        grad_tensor_bytes = serialize_tensors(grad_tensors_dict)

        for chunk in split_bytes(grad_tensor_bytes):
            yield ModuleBackwardResponse(success=True, grad_tensor_bytes=chunk)