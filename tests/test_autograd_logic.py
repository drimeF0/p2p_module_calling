import torch
from torch.autograd import Function
from typing import Dict, Tuple, Optional
from torch.autograd.function import once_differentiable

from p2p_module_calling.client import Client, RemoteModule
from p2p_module_calling.server import ModuleServicer

from hivemind import DHT
from hivemind.p2p import PeerID, P2P


import asyncio

class TestModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    def forward(self, x):
        result = self.linear(x)
        dictonary = {"result": result}
        return dictonary

async def main():
    dht = DHT(
        start=True
    )
    p2p = await dht.replicate_p2p()
    my_peer_id = dht.peer_id
    print(my_peer_id.to_string())

    model = TestModel()
    models = {"test_model": model}

    servicer = ModuleServicer(dht, models)
    servicer.start()
    
    client = Client(p2p, my_peer_id)
    remote_module = RemoteModule(client, "test_model")

    x = torch.ones(10, 10, requires_grad=False)
    result = remote_module({"x": x})["result"]
    result.sum().backward()
    print(result)
    print(result.grads)

asyncio.run(main())
    