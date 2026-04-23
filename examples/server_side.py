from hivemind import DHT, get_dht_time
import time

from p2p_module_calling.server import ModuleServicer
from p2p_module_calling.constants import PUBLIC_INITIAL_PEERS


import asyncio

import torch

import logging



class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Linear(10,10)
    
    def forward(self, x):
        result = self.mlp(x)
        dictonary = {"result": result}
        return dictonary


dht = DHT(
    start=False,
    #initial_peers=PUBLIC_INITIAL_PEERS,
    use_auto_relay=True,
    use_relay=True,
    use_ipfs=True,
)
dht.run_in_background(timeout=60)
my_peer_id = dht.peer_id
print(my_peer_id.to_string())
print(dht.get_visible_maddrs())
model = TestModel().requires_grad_(False)
models = {"test_model": model}

with torch.enable_grad():
    data = {"x": torch.ones(1, 10)}
    data["x"].requires_grad = True
    result = model(**data)
    result['result'].sum().backward()  # Sum the result to get a scalar for backpropagation
    print(data["x"].grad)

servicer = ModuleServicer(dht, models)
print(servicer.run_in_background())

servicer.join()