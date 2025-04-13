import torch
from typing import Dict, Tuple, Optional

from p2p_module_calling.client import Client, RemoteModule
from p2p_module_calling.constants import PUBLIC_INITIAL_PEERS



from p2p_module_calling.module_service import (
    ModuleForwardRequest, ModuleForwardResponse, ModuleBackwardRequest, ModuleBackwardResponse
)

from hivemind import DHT
from hivemind.p2p import PeerID, P2P


import logging 

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
my_peer_id = PeerID.from_base58(input("Enter peer id: "))
client = Client(dht, my_peer_id)
remote_module = RemoteModule(client, "test_model")


with torch.enable_grad():
    data = {"x": torch.randn(1,10)}
    result = remote_module(data)
    print(result)
    result["result"].backward()
    print(result)
