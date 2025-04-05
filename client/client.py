from hivemind import DHT
from hivemind.p2p import P2P, PeerID, StubBase

from p2p_module_calling.server.module_servicer import ModuleServiceServicer

from p2p_module_calling.utils import serialize_tensors, deserialize_tensors


from p2p_module_calling.module_service import (
    ModuleForwardRequest, ModuleForwardResponse
)


import torch

from typing import Dict


def get_server_stub(p2p: P2P, server_peer_id: PeerID) -> StubBase:
    return ModuleServiceServicer.get_stub(p2p, server_peer_id)


class Client:

    def __init__(self, p2p: P2P, server_peer_id: PeerID):
        self.stub = get_server_stub(p2p, server_peer_id)
    

    def forward(self, module_id: str, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data_bytes: bytes = deserialize_tensors(data)
        message = ModuleForwardRequest(module_id=module_id, input_tensor_bytes=data_bytes)
        result: ModuleForwardResponse = self.stub.rpc_forward_module(message)
        return serialize_tensors(result.output_tensor_bytes)

