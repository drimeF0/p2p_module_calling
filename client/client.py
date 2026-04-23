from doctest import debug
from sympy import Dummy
from hivemind import DHT
from hivemind.p2p import P2P, PeerID, StubBase
from hivemind.utils.logging import get_logger


from p2p_module_calling.server.module_servicer import ModuleServicer

from p2p_module_calling.utils import serialize_tensors, deserialize_tensors, flatten, fill_dict, unfill_dict, flatten


from p2p_module_calling.module_service import (
    ModuleForwardRequest, ModuleForwardResponse, ModuleBackwardRequest, ModuleBackwardResponse
)

from p2p_module_calling.client.remote_module_worker import RemoteModuleWorker


import torch
from torch.autograd.function import once_differentiable

from typing import Dict, List, Union, Optional, Tuple, Any, Awaitable

import asyncio


DUMMY = torch.empty(0, requires_grad=True)
DUMMY.requires_grad = True

logger = get_logger(__name__)

def get_server_stub(p2p: P2P, server_peer_id: PeerID) -> StubBase:
    return ModuleServicer.get_stub(p2p, server_peer_id)




class Client:

    def __init__(self, dht: DHT, server_peer_id: PeerID):
        self.p2p = RemoteModuleWorker.run_coroutine(dht.replicate_p2p()) #replica now running on a different loop
        self.stub = get_server_stub(self.p2p, server_peer_id)
    

    def forward(self, module_id: str, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data_bytes: bytes = serialize_tensors(data)
        message = ModuleForwardRequest(module_id=module_id, input_tensor_bytes=data_bytes)
        result: ModuleForwardResponse = RemoteModuleWorker.run_coroutine(self.stub.rpc_forward_module(message))
        return deserialize_tensors(result.output_tensor_bytes)
    
    def backward(self, module_id: str, data: Dict[str, torch.Tensor], grad: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data_bytes: bytes = serialize_tensors(data)
        grad_bytes: bytes = serialize_tensors(grad)
        message = ModuleBackwardRequest(module_id=module_id, input_tensor_bytes=data_bytes, grad_tensor_bytes=grad_bytes)
        result: ModuleBackwardResponse = RemoteModuleWorker.run_coroutine(self.stub.rpc_backward_module(message))
        return deserialize_tensors(result.grad_tensor_bytes)
    
    def __del__(self):
        RemoteModuleWorker.run_coroutine(self.p2p.shutdown())
    

class RemoteModule(torch.nn.Module):

    def __init__(self, client: Client, module_id: str):
        super().__init__()
        self.client = client
        self.module_id = module_id
    

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tensors, keys = flatten(x)
        return _RemoteModuleCall.apply(DUMMY, self.module_id, self.client, keys, *tensors)

class _RemoteModuleCall(torch.autograd.Function): #https://github.com/learning-at-home/hivemind/blob/master/hivemind/moe/client/expert.py#L194
    """Internal autograd-friendly call of a remote module."""

    @staticmethod
    def forward(
        ctx,
        dummy: torch.Tensor,
        uid: str,
        client: Client,
        inputs_keys: List[str],
        *inputs_tensors: torch.Tensor,
    ) -> torch.Tensor:
        

        inputs = {key: tensor.cpu().detach() for key, tensor in zip(inputs_keys, inputs_tensors)}
        ctx.uid, ctx.client, ctx.keys  = uid, client, inputs_keys
        ctx.save_for_backward(*inputs_tensors)
        result = client.forward(uid, inputs)["result"]
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:

        grad_outputs_cpu = [tensor.cpu().contiguous() for tensor in grad_outputs]
        restored_inputs = {key: tensor for key, tensor in zip(ctx.keys, ctx.saved_tensors)}
        grad_inputs = fill_dict(grad_outputs_cpu)
        result = ctx.client.backward(ctx.uid, restored_inputs, grad_inputs)
        flatten_grads = unfill_dict(result)

        returns = (DUMMY, None, None, None, *flatten_grads)
        logger.error(f"restored_inputs: : {restored_inputs}")
        return returns