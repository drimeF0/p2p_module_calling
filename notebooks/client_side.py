from hivemind import DHT, get_dht_time
from hivemind.p2p import PeerID
import hivemind

import time

from p2p_module_calling.client.client import Client

import torch

import asyncio

peers =  [
    # IPv4 DNS addresses
    "/ip4/159.89.214.152/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY"
]

table = DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
    initial_peers=peers,
    start=True,
    use_relay=True,
    use_auto_relay=True
)

p2p = asyncio.run(table.replicate_p2p())


peer_id_bytes = table.get("drime_peers").value
peer = PeerID(peer_id_bytes)
client  = Client(p2p,peer)
print(asyncio.run(client.forward("model",{"x":torch.randn(3,10)})))
