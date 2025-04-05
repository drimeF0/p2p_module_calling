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
    
    async def forward_module(self, request: ModuleForwardRequest, context: P2PContext) -> ModuleForwardResponse:
        module: Optional[torch.nn.Module] = self.modules.get(request.module_id)
        if module is None:
            return ModuleForwardResponse(error="Module not found", success=False, output_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES)
        input_tensors: Dict[str, torch.Tensor] = serialize_tensors(request.input_tensor_bytes)
        output_tensors: Dict[str, torch.Tensor] = module(**input_tensors)
        output_tensor_bytes: bytes = deserialize_tensors(output_tensors)
        return ModuleForwardResponse(success=True, output_tensor_bytes=output_tensor_bytes)
    

    async def forward_module_stream(self, requests: AsyncIterator[ModuleForwardRequest], context: P2PContext) -> AsyncIterator[ModuleForwardResponse]:
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


    # async def backward_module(self, request: ModuleBackwardRequest, context: P2PContext) -> ModuleBackwardResponse:
    #     module: Optional[torch.nn.Module] = self.modules.get(request.module_id)
    #     if module in None:
    #         return ModuleBackwardResponse(error="Module not found", success=False, output_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES)
    
    # async def backward_module_stream(self, requests: AsyncIterator[ModuleBackwardRequest], context: P2PContext) -> AsyncIterator[ModuleBackwardResponse]:
    #     bytes_buffer = bytearray()
    #     async for message in requests:
    #         byte_data = message.input_tensor_bytes
    #         bytes_buffer.append(byte_data)
    #     tensor_dict: Dict[str,torch.Tensor] = serialize_tensors(bytes(bytes_buffer))
    #     module_id = message.module_id
    #     module = self.modules.get(module_id)
    #     if module is None:
    #         yield ModuleBackwardResponse(error="Module not found", success=False, output_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES)
    #         return
    #     output_tensors: Dict[str, torch.Tensor] = module(**tensor_dict)
    #     output_tensor_bytes: bytes = deserialize_tensors(output_tensors)
    #     tensor_chunks = split_bytes(output_tensor_bytes)
    #     for chunk in tensor_chunks:
    #         yield ModuleBackwardResponse(success=True, output_tensor_bytes=chunk)
