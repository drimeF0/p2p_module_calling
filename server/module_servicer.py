from p2p_module_calling.module_service import (
    TestRequest,
    TestResponse
)

from p2p_module_calling.utils import serialize_tensors, deserialize_tensors, DEFAULT_ZERO_SAFETENSOR_BYTES

from hivemind.p2p import ServicerBase, P2PContext #ServicerBase - Base class for P2P RPC servicers (e.g. DHT, Remote Module Calling, MoE server). The interface mimics gRPC servicers.
from hivemind.p2p.p2p_daemon import DEFAULT_MAX_MSG_SIZE, P2P
from hivemind.utils.asyncio import switch_to_uvloop #Switch to faster uvloop event loop, if not available installing it
from hivemind.dht import DHT

import asyncio

import torch

from typing import Dict, List, Optional, Any, Tuple


class ModuleServiceServicer(ServicerBase):

    def __init__(self, dht: DHT):
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
    
    async def rpc_call_module(self, request: TestRequest, context: P2PContext) -> TestResponse:
        tensor_dict: Dict[str, torch.Tensor] = serialize_tensors(request.input_tensor_bytes)
        print(f"got dict: {tensor_dict}")
        return TestResponse(output_tensor_bytes=request.input_tensor_bytes)