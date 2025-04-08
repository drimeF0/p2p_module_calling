import torch
from typing import Dict, Tuple, Optional

from p2p_module_calling.client import Client, RemoteModule
from p2p_module_calling.server import ModuleServicer
from p2p_module_calling.constants import PUBLIC_INITIAL_PEERS
from p2p_module_calling.utils import serialize_tensors, deserialize_tensors



from p2p_module_calling.module_service import (
    ModuleForwardRequest, ModuleForwardResponse, ModuleBackwardRequest, ModuleBackwardResponse
)

from hivemind import DHT
from hivemind.p2p import PeerID, P2P



import threading

import asyncio

class TestModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    def forward(self, x):
        result = self.linear(x)
        dictonary = {"result": result}
        return dictonary

dht = DHT(
    start=True,
    initial_peers=PUBLIC_INITIAL_PEERS,
    use_auto_relay=True,
    use_relay=True,
)
p2p = asyncio.run(dht.replicate_p2p())
my_peer_id = PeerID.from_base58(input("Enter peer id: "))
client = Client(p2p, my_peer_id)

data = {"x": torch.randn(1,10)}

print(
    client.forward("test_model", data)
)

