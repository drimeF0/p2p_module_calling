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


import logging

logger = logging.Logger(name="ModuleServiceServicer")


class ModuleServicer(ServicerBase):

    err_message_module_not_found_forward = ModuleForwardResponse(error_message="Module not found", success=False, output_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES)
    err_message_module_not_found_backward = ModuleBackwardResponse(error_message="Module not found", success=False, grad_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES)
    err_message_input_or_grad_must_be_provided = ModuleBackwardResponse(error_message="Input and gradient tensors must be provided", success=False, grad_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES)

    def __init__(self, dht: DHT, modules: Dict[str,torch.nn.Module]):
        super().__init__()
        self.dht = dht
        self.modules = modules

        self._p2p: Optional[P2P] = None

    

    def run(self):
        torch.set_num_threads(1)    
        asyncio_loop = asyncio.get_event_loop()
        try:
            asyncio_loop.run_until_complete(self.async_run())
        except KeyboardInterrupt:
            asyncio_loop.run_until_complete(self.remove_p2p_handlers(self._p2p))
    

    async def async_run(self):
        try:
            self._p2p = await self.dht.replicate_p2p()
            await self.add_p2p_handlers(self._p2p, balanced=True)
        except Exception as e:
            print(e.with_traceback())


    def _backward(self, tensors: Dict[str, torch.Tensor], grad_tensors: Dict[str, torch.Tensor]):
        for key, tensor in tensors.items():
            torch.autograd.backward(tensor, grad_tensors[key], retain_graph=False, create_graph=False)
    
    def get_module(self, module_id: str) -> Optional[torch.nn.Module]:
        return self.modules.get(module_id)
    
    async def rpc_forward_module(self, request: ModuleForwardRequest, context: P2PContext) -> ModuleForwardResponse:
        module: Optional[torch.nn.Module] = self.get_module(request.module_id)

        if module is None:
            return self.err_message_module_not_found
        
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
            yield self.err_message_module_not_found_forward
        else:
            output_tensors: Dict[str, torch.Tensor] = module(**tensor_dict)
            output_tensor_bytes: bytes = deserialize_tensors(output_tensors)
            tensor_chunks: List[bytes] = split_bytes(output_tensor_bytes)
            for chunk in tensor_chunks:
                yield ModuleForwardResponse(success=True, output_tensor_bytes=chunk)


    async def rpc_backward_module(self, request: ModuleBackwardRequest, context: P2PContext) -> ModuleBackwardResponse:

        module: Optional[torch.nn.Module] = self.get_module(request.module_id)

        if module in None:
            return self.err_message_module_not_found_backward

        if request.input_tensor_bytes is None or request.grad_tensor_bytes is None:
            return self.err_message_input_or_grad_must_be_provided

        input_tensors: Dict[str, torch.Tensor] = serialize_tensors(request.input_tensor_bytes)
        grad_tensors: Dict[str, torch.Tensor] = serialize_tensors(request.grad_tensor_bytes)
        
        with torch.enable_grad():
            input_tensors = {input_key: (tensor.detach().requires_grad_(True) if tensor.is_floating_point() else tensor.detach()) for input_key, tensor in input_tensors.items()} #input tokens are not floating point and do not require grads
            output_tensors: Dict[str, torch.Tensor] = module(**input_tensors)
            self._backward(output_tensors, grad_tensors)
        
        grad_tensors = {input_key: tensor.grad if isinstance(tensor, torch.Tensor) else torch.zeros_like(tensor) for input_key, tensor in input_tensors.items()}
        grad_tensor_bytes: bytes = deserialize_tensors(grad_tensors)
        return ModuleBackwardResponse(success=True, grad_tensor_bytes=grad_tensor_bytes)
    
    async def rpc_backward_module_stream(self, requests: AsyncIterator[ModuleBackwardRequest], context: P2PContext) -> AsyncIterator[ModuleBackwardResponse]:
        bytes_buffer_inputs: bytes = bytearray()
        bytes_buffer_grads: bytes = bytearray()


        async for request in requests:
            if request.input_tensor_bytes is not None:  # Accumulate input tensor bytes if present
                bytes_buffer_inputs.extend(request.input_tensor_bytes)
            if request.grad_tensor_bytes is not None:  # Accumulate gradient tensor bytes if present
                bytes_buffer_grads.extend(request.grad_tensor_bytes)
        
        module = self.get_module(request.module_id)
        if module is None:
            yield self.err_message_module_not_found_backward
            return
        
        if not bytes_buffer_inputs or not bytes_buffer_grads:
            yield self.err_message_input_or_grad_must_be_provided
            return
        
        input_tensors: Dict[str, torch.Tensor] = serialize_tensors(bytes_buffer_inputs)
        grad_tensors: Dict[str, torch.Tensor] = serialize_tensors(bytes_buffer_grads)
        
        with torch.enable_grad():
            input_tensors = {input_key: (tensor.detach().requires_grad_(True) if tensor.is_floating_point() else tensor.detach()) for input_key, tensor in input_tensors.items()} #input tokens are not floating point and do not require grads
            output_tensors = module(**input_tensors)

            self._backward(output_tensors, grad_tensors)
        grad_tensors: Dict[str, torch.Tensor] = {input_key: tensor.grad if isinstance(tensor, torch.Tensor) else torch.zeros_like(tensor) for input_key, tensor in input_tensors.items()}
        grad_tensor_bytes: bytes = serialize_tensors(grad_tensors)
        grad_tensor_chunks: List[bytes] = split_bytes(grad_tensor_bytes)

        for chunk in grad_tensor_chunks:
            yield ModuleBackwardResponse(success=True, grad_tensor_bytes=chunk)

