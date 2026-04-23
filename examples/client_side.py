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

import torchvision

import asyncio





dht = DHT(
    start=False,
    #initial_peers=PUBLIC_INITIAL_PEERS,
    use_auto_relay=True,
    use_relay=True,
    use_ipfs=True,
)
dht.run_in_background(timeout=60)
my_peer_id = PeerID.from_base58(input("Enter peer id: "))
client = Client(dht, my_peer_id)
remote_module = RemoteModule(client, "test_model")
mse_loss = torch.nn.MSELoss()
remote_module.requires_grad_(True)
with torch.enable_grad():
    data = {"x": torch.ones(1, 10)}
    data["x"].requires_grad = True
    result = remote_module(data)
    print(data)
    result.sum().backward()
    print(data["x"].grad)
