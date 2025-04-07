from sympy import Dummy
from hivemind import DHT
from hivemind.p2p import P2P, PeerID, StubBase

from p2p_module_calling.server.module_servicer import ModuleServicer

from p2p_module_calling.utils import serialize_tensors, deserialize_tensors


from p2p_module_calling.module_service import (
    ModuleForwardRequest, ModuleForwardResponse, ModuleBackwardRequest, ModuleBackwardResponse
)


import torch
from torch.autograd.function import once_differentiable

from typing import Dict, List, Union, Optional, Tuple, Any

import asyncio

DUMMY = torch.Tensor([0])
DUMMY.requires_grad = True

def get_server_stub(p2p: P2P, server_peer_id: PeerID) -> StubBase:
    return ModuleServicer.get_stub(p2p, server_peer_id)




class Client:

    def __init__(self, p2p: P2P, server_peer_id: PeerID):
        self.stub = get_server_stub(p2p, server_peer_id)
    

    def forward(self, module_id: str, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data_bytes: bytes = deserialize_tensors(data)
        message = ModuleForwardRequest(module_id=module_id, input_tensor_bytes=data_bytes)
        result: ModuleForwardResponse = asyncio.run(self.stub.rpc_forward_module(message))
        return serialize_tensors(result.output_tensor_bytes)
    
    def backward(self, module_id: str, data: Dict[str, torch.Tensor], grad: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data_bytes: bytes = deserialize_tensors(data)
        grad_bytes: bytes = deserialize_tensors(grad)
        message = ModuleBackwardRequest(module_id=module_id, input_tensor_bytes=data_bytes, grad_tensor_bytes=grad_bytes)
        result: ModuleBackwardResponse = asyncio.run(self.stub.rpc_backward_module(message))
        return serialize_tensors(result.output_tensor_bytes)
    

class RemoteModule(torch.nn.Module):

    def __init__(self, client: Client, module_id: str):
        super().__init__()
        self.client = client
        self.module_id = module_id
    

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return _RemoteModuleCall.apply(DUMMY, self.module_id, self.client, x)

class _RemoteModuleCall(torch.autograd.Function): #https://github.com/learning-at-home/hivemind/blob/master/hivemind/moe/client/expert.py#L194
    """Internal autograd-friendly call of a remote module. For applications, use RemoteExpert instead."""

    @staticmethod
    def forward(
        ctx,
        dummy: torch.Tensor,
        uid: str,
        client: Client,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        inputs = {input_key: tensor.cpu().detach() for input_key, tensor in inputs.items()}
        inputs_tensors = tuple(inputs.values())
        inputs_keys = tuple(inputs.keys())
        ctx.uid, ctx.client, ctx.keys  = uid, client, inputs_keys
        ctx.save_for_backward(*inputs_tensors)
        result = client.forward(uid, inputs)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:
        grad_outputs_cpu = tuple(tensor.cpu() for tensor in grad_outputs)
        restored_inputs = {key: tensor for key, tensor in zip(ctx.keys, ctx.saved_tensors)}
        result = ctx.client.backward(ctx.uid, restored_inputs, grad_outputs_cpu)
        flatten_grads = [result[key] for key in ctx.keys]

        return (DUMMY, None, None, None, *flatten_grads)