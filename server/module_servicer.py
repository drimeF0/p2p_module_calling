from p2p_module_calling.module_service import (
    ModuleRegistrationRequest,
    ModuleRegistrationResponse,
    ModuleCallRequest,
    ModuleCallResponse,
    ModuleDeleteRequest,
    ModuleDeleteResponse,
    ListModulesRequest,
    ListModulesResponse,
)

from p2p_module_calling.utils import serialize_tensors, deserialize_tensors, DEFAULT_ZERO_SAFETENSOR_BYTES

from hivemind.p2p.servicers import ServicerBase, P2PContext #ServicerBase - Base class for P2P RPC servicers (e.g. DHT, Remote Module Calling, MoE server). The interface mimics gRPC servicers.
from hivemind.p2p.p2p_daemon import DEFAULT_MAX_MSG_SIZE, P2P
from hivemind.utils.asyncio import switch_to_uvloop #Switch to faster uvloop event loop, if not available installing it
from hivemind.dht import DHT

import asyncio

import torch

from typing import Dict, List, Optional, Any, Tuple


class ModuleServiceServicer(ServicerBase):

    def __init__(self, modules: Dict[str, torch.nn.Module], dht: DHT):
        self.modules = modules
        self.dht = dht
        self._p2p: Optional[P2P] = None
    

    def run(self):
        asyncio_loop = switch_to_uvloop()
        stop = asyncio.Event()
        asyncio_loop.run_until_complete(self._run(stop))
    
    async def _run(self, stop: asyncio.Event):
        self._p2p = await self.dht.replicate_p2p()
        await self.add_p2p_handlers(self._p2p, balanced=True)
        try:
            await stop.wait()
        finally:
            await self.remove_p2p_handlers(self._p2p)
    
    async def rpc_call_module(self, request: ModuleCallRequest, context: P2PContext) -> ModuleCallResponse:
        module: Optional[torch.nn.Module] = self.modules.get(request.module_id)
        if module is None:
            return ModuleCallResponse(error="Module not found", success=False, output_tensor_bytes=DEFAULT_ZERO_SAFETENSOR_BYTES)        
        input_tensors: Dict[str, torch.Tensor] = serialize_tensors(request.input_tensor_bytes)
        output_tensors: Dict[str, torch.Tensor] = module(**input_tensors)
        output_tensor_bytes: bytes = deserialize_tensors(output_tensors)
        return ModuleCallResponse(error="", success=True, output_tensor_bytes=output_tensor_bytes)
    
    async def rpc_list_modules(self, request: ListModulesRequest, context: P2PContext) -> ListModulesResponse:
        filter = request.filter
        modules_found = []

        for module_id in self.modules.keys():
            if filter in module_id:
                modules_found.append(module_id)
        if len(modules_found) == 0:
            return ListModulesResponse(error="No modules found", success=False, module_ids=["No modules found"])
        return ListModulesResponse(error="", success=True, module_ids=modules_found)