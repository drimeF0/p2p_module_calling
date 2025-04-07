from hivemind import DHT, get_dht_time
import time

from p2p_module_calling.server import ModuleServicer
from p2p_module_calling.constants import PUBLIC_INITIAL_PEERS


import asyncio

import torch


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
my_peer_id = dht.peer_id
print(my_peer_id.to_string())
model = TestModel()
models = {"test_model": model}
servicer = ModuleServicer(dht, models)
asyncio.wait_for(servicer.run())