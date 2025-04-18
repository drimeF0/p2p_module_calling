import torch
from typing import Dict, Tuple, Optional

from p2p_module_calling.client import Client, RemoteModule
from p2p_module_calling.server import ModuleServicer
from p2p_module_calling.constants import PUBLIC_INITIAL_PEERS

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


def server_main():
    dht = DHT(
        start=True,
        initial_peers=PUBLIC_INITIAL_PEERS,
        use_auto_relay=True,
        use_relay=True,
    )
    my_peer_id = dht.peer_id
    print(my_peer_id.to_string())
    model = TestModel()
    models = {"test_model": model}
    servicer = ModuleServicer(dht, models)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    servicer.run()

def client_main():
    dht = DHT(
        start=True,
        initial_peers=PUBLIC_INITIAL_PEERS,
        use_auto_relay=True,
        use_relay=True,
    )
    p2p = asyncio.run(dht.replicate_p2p())
    my_peer_id = PeerID.from_base58(input("Enter peer id: "))
    client = Client(p2p, my_peer_id)
    remote_module = RemoteModule(client, "test_model")
    x = torch.ones(10, 10, requires_grad=False)
    result = remote_module({"x": x})["result"]
    result.sum().backward()
    print(result)
    print(result.grads)

if __name__ == "__main__":
    server_thread = threading.Thread(target=server_main)
    server_thread.start()
    client_main()
    