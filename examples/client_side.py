from hivemind import DHT, get_dht_time
from hivemind.p2p import PeerID
import hivemind

import time

from p2p_module_calling.client.client import Client

import torch

import asyncio


async def main():
    peers =  [
        # IPv4 DNS addresses
        "/dns/bootstrap1.petals.dev/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY",
    ]
    
    table = DHT(
        initial_peers=peers,
        start=True,
        use_relay=True,
        use_auto_relay=True
    )
    
    p2p = await table.replicate_p2p()

    print(await p2p.list_peers())
    
    
    peer = PeerID.from_base58(input("peerID:"))
    client  = Client(p2p,peer)
    print(await client.test({"x":torch.randn(3,10)}))

asyncio.run(main())
