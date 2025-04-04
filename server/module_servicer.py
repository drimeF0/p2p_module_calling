from numpy import bytes_
from p2p_module_calling.module_service import (
    ModuleForwardRequest,
    ModuleForwardResponse,
    ModuleBackwardRequest,
    ModuleBackwardResponse,
)

from p2p_module_calling.utils import serialize_tensors, deserialize_tensors, DEFAULT_ZERO_SAFETENSOR_BYTES, split_bytes

from hivemind.p2p import ServicerBase, P2PContext #ServicerBase - Base class for P2P RPC servicers (e.g. DHT, Remote Module Calling, MoE server). The interface mimics gRPC servicers.
from hivemind.p2p.p2p_daemon import DEFAULT_MAX_MSG_SIZE, P2P
from hivemind.utils.asyncio import switch_to_uvloop #Switch to faster uvloop event loop, if not available installing it
from hivemind.dht import DHT
from hivemind.utils import MPFuture


import asyncio

import torch

from typing import Dict, List, Optional, AsyncIterator

import multiprocessing as mp

import logging

logger = logging.Logger(name="ModuleServiceServicer")


class ModuleServiceServicer(mp.context.ForkProcess, ServicerBase):

    def __init__(self, dht: DHT, modules: Dict[str,torch.nn.Module]):
        self.dht = dht
        self.modules = modules

        self._p2p: Optional[P2P] = None

        self._inner_pipe, self._outer_pipe = mp.Pipe(duplex=False)

        self.ready = MPFuture()
    

    def run(self):
        torch.set_num_threads(1)    
        asyncio_loop = asyncio.new_event_loop()
        stop = asyncio.Event()
        asyncio_loop.add_reader(self._inner_pipe.fileno(), stop.set)
    
        async def _run():
            try:
                self._p2p = await self.dht.replicate_p2p()
                await self.add_p2p_handlers(self._p2p, balanced=True)
                self.ready.set_result(None)
            except Exception as e:
                self.ready.set_exception(e)

            try:
                await stop.wait()
            finally:
                await self.remove_p2p_handlers(self._p2p)
            
            try:
                asyncio_loop.run_until_complete(_run())
            except KeyboardInterrupt:
                pass

    
    def run_in_background(self, await_ready: bool = True, timeout: Optional[float] = None) -> None:
        """
        Starts ConnectionHandler in a background process. If :await_ready:, this method will wait until
        it is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready:
            self.wait_until_ready(timeout)

    def wait_until_ready(self, timeout: Optional[float] = None) -> None:
        self.ready.result(timeout=timeout)


    def shutdown(self):
        if self.is_alive():
            self._outer_pipe.send("_shutdown")
            self.join(self.shutdown_timeout)
            if self.is_alive():
                logger.warning(
                    "ConnectionHandler did not shut down within the grace period; terminating it the hard way"
                )
                self.terminate()
        else:
            logger.warning("ConnectionHandler shutdown had no effect, the process is already dead")
    
    async def rpc_forward_module(self, request: ModuleForwardRequest, context: P2PContext) -> ModuleForwardResponse:
        module: Optional[torch.nn.Module] = self.modules.get(request.module_id)
        if module is None:
            return ModuleForwardResponse(error="Module not found", success=False, output_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES)
        input_tensors: Dict[str, torch.Tensor] = serialize_tensors(request.input_tensor_bytes)
        output_tensors: Dict[str, torch.Tensor] = module(**input_tensors)
        output_tensor_bytes: bytes = deserialize_tensors(output_tensors)
        return ModuleForwardResponse(success=True, output_tensor_bytes=output_tensor_bytes)
    

    async def rpc_forward_module_stream(self, requests: AsyncIterator[ModuleForwardRequest], context: P2PContext) -> AsyncIterator[ModuleForwardResponse]:
        bytes_buffer = bytearray()
        async for message in requests:
            byte_data = message.input_tensor_bytes
            bytes_buffer.append(byte_data)
        tensor_dict: Dict[str,torch.Tensor] = serialize_tensors(bytes(bytes_buffer))
        module_id = message.module_id
        module = self.modules.get(module_id)
        if module is None:
            yield ModuleForwardResponse(error="Module not found", success=False, output_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES)
        else:
            output_tensors: Dict[str, torch.Tensor] = module(**tensor_dict)
            output_tensor_bytes: bytes = deserialize_tensors(output_tensors)
            tensor_chunks: List[bytes] = split_bytes(output_tensor_bytes)
            for chunk in tensor_chunks:
                yield ModuleForwardResponse(success=True, output_tensor_bytes=chunk)


    async def rpc_backward_module(self, request: ModuleBackwardRequest, context: P2PContext) -> ModuleBackwardResponse:
        module: Optional[torch.nn.Module] = self.modules.get(request.module_id)


        if module in None:
            return ModuleBackwardResponse(error="Module not found", success=False, grad_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES)

        if request.input_tensor_bytes is None or request.grad_tensor_bytes is None:
            return ModuleBackwardResponse(error="Input and gradient tensors must be provided", success=False, grad_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES)
        input_tensors: Dict[str, torch.Tensor] = serialize_tensors(request.input_tensor_bytes)
        grad_tensors: Dict[str, torch.Tensor] = serialize_tensors(request.grad_tensor_bytes)
        
        with torch.enable_grad():
            input_tensors = {input_key: (tensor.detach().requires_grad_(True) if tensor.is_floating_point() else tensor.detach()) for input_key, tensor in input_tensors.items()} #input tokens are not floating point and do not require grads
            output_tensors: Dict[str, torch.Tensor] = module(**input_tensors)
            torch.autograd.backward(output_tensors, grad_tensors, retain_graph=False, create_graph=False)
        
        grad_tensors = {input_key: tensor.grad if isinstance(tensor, torch.Tensor) else torch.zeros_like(tensor) for input_key, tensor in input_tensors.items()}
        grad_tensor_bytes: bytes = deserialize_tensors(grad_tensors)
        return ModuleBackwardResponse(success=True, grad_tensor_bytes=grad_tensor_bytes)
    
    async def rpc_backward_module_stream(self, requests: AsyncIterator[ModuleBackwardRequest], context: P2PContext) -> AsyncIterator[ModuleBackwardResponse]:
        bytes_buffer_inputs: bytes = bytearray()  # Buffer to accumulate input tensor bytes
        bytes_buffer_grads: bytes = bytearray()  # Buffer to accumulate gradient tensor bytes)
        async for request in requests:
            if request.input_tensor_bytes is not None:  # Accumulate input tensor bytes if present
                bytes_buffer_inputs.extend(request.input_tensor_bytes)
            if request.grad_tensor_bytes is not None:  # Accumulate gradient tensor bytes if present
                bytes_buffer_grads.extend(request.grad_tensor_bytes)
        module = self.modules.get(request.module_id)
        if module is None:
            yield ModuleBackwardResponse(error="Module not found", success=False, grad_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES)
            return
        if not bytes_buffer_inputs or not bytes_buffer_grads:  # Check if both buffers are non-empty
            yield ModuleBackwardResponse(error="Missing input or gradient tensors", success=False, grad_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES)
            return
        input_tensors: Dict[str, torch.Tensor] = serialize_tensors(bytes_buffer_inputs)  # Deserialize input tensors from accumulated bytes
        grad_tensors: Dict[str, torch.Tensor] = serialize_tensors(bytes_buffer_grads)  # Deserialize gradient tensors from accumulated bytes
        with torch.enable_grad():
            input_tensors = {input_key: (tensor.detach().requires_grad_(True) if tensor.is_floating_point() else tensor.detach()) for input_key, tensor in input_tensors.items()} #input tokens are not floating point and do not require grads
            output_tensors = module(**input_tensors)

            torch.autograd.backward(output_tensors, grad_tensors=grad_tensors) 
        grad_tensors: Dict[str, torch.Tensor] = {input_key: tensor.grad if isinstance(tensor, torch.Tensor) else torch.zeros_like(tensor) for input_key, tensor in input_tensors.items()}
        grad_tensor_bytes: bytes = serialize_tensors(grad_tensors)  # Serialize gradient tensors to bytes
        grad_tensor_chunks: List[bytes] = split_bytes(grad_tensor_bytes)
        for chunk in grad_tensor_chunks:
            yield ModuleBackwardResponse(success=True, grad_tensor_bytes=chunk)

